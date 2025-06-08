#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define X_RESN 4000      /* x resolution */
#define Y_RESN 4000      /* y resolution */
#define NUM_COLORS 256     /* número de cores na paleta */

typedef struct complextype {
    float real, imag;
} Compl;

/* Paleta de cores global */
unsigned long color_palette[NUM_COLORS];

/* Inicializar paleta de cores */
void init_color_palette(Display *display, int screen) {
    XColor color;
    Colormap colormap = DefaultColormap(display, screen);
    
    // Definir cores da paleta (RGB em 16-bit: 0-65535)
    unsigned short palette_rgb[NUM_COLORS][3] = {
        {0,     0,     0},      // Preto
        {65535, 0,     0},      // Vermelho
        {0,     65535, 0},      // Verde
        {0,     0,     65535},  // Azul
        {65535, 65535, 0},      // Amarelo
        {65535, 0,     65535},  // Magenta
        {0,     65535, 65535},  // Ciano
        {65535, 32767, 0}       // Laranja
    };
    
    for (int i = 0; i < NUM_COLORS; i++) {
        color.red = palette_rgb[i][0];
        color.green = palette_rgb[i][1];
        color.blue = palette_rgb[i][2];
        color.flags = DoRed | DoGreen | DoBlue;
        
        if (XAllocColor(display, colormap, &color)) {
            color_palette[i] = color.pixel;
        } else {
            color_palette[i] = BlackPixel(display, screen);
        }
    }
}

/* Função para obter cor da paleta baseada na posição */
unsigned long get_color_from_palette(int i, int j) {
    // Usar coordenadas para selecionar cor da paleta
    int color_index = (i / 100 + j / 100) % NUM_COLORS;
    return color_palette[color_index];
}

void main() {
    Window      win;                            /* initialization for a window */
    unsigned
    int         width, height,                  /* window size */
                x, y,                           /* window position */
                border_width,                   /* border width in pixels */
                display_width, display_height,  /* size of screen */
                screen;                         /* which screen */

    char        *window_name = "Mandelbrot Set Colorido", *display_name = NULL;
    GC          gc;
    unsigned
    long        valuemask = 0;
    XGCValues	values;
    Display     *display;
    XSizeHints	size_hints;
    Pixmap      bitmap;
    XPoint      points[4000];
    FILE        *fp, *fopen();
    char        str[100];
    
    XSetWindowAttributes attr[1];

    /* Mandlebrot variables */
    int		i, j, k;
    Compl	z, c;
    float	lengthsq, temp;
       
    /* connect to Xserver */
    if ((display = XOpenDisplay(display_name)) == NULL) {
        fprintf(stderr, "drawon: cannot connect to X server %s\n",
                XDisplayName(display_name));
        exit(-1);
    }
    
    /* get screen size */
    screen = DefaultScreen(display);
    display_width = DisplayWidth(display, screen);
    display_height = DisplayHeight(display, screen);

    /* set window size */
    width = 4000;
    height = 4000;

    /* set window position */
    x = 0;
    y = 0;

    /* create opaque window */
    border_width = 4;
    win = XCreateSimpleWindow(display, RootWindow(display, screen),
                x, y, width, height, border_width, 
                BlackPixel(display, screen), WhitePixel(display, screen));

    size_hints.flags = USPosition|USSize;
    size_hints.x = x;
    size_hints.y = y;
    size_hints.width = width;
    size_hints.height = height;
    size_hints.min_width = 300;
    size_hints.min_height = 300;
    
    XSetNormalHints(display, win, &size_hints);
    XStoreName(display, win, window_name);

    /* create graphics context */
    gc = XCreateGC(display, win, valuemask, &values);

    XSetBackground(display, gc, WhitePixel(display, screen));
    XSetForeground(display, gc, BlackPixel(display, screen));
    XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);

    attr[0].backing_store = Always;
    attr[0].backing_planes = 1;
    attr[0].backing_pixel = BlackPixel(display, screen);

    XChangeWindowAttributes(display, win, CWBackingStore | CWBackingPlanes | CWBackingPixel, attr);

    XMapWindow(display, win);
    XSync(display, 0);

    /* Inicializar paleta de cores */
    init_color_palette(display, screen);

    double t_inicio = omp_get_wtime();
           
    /* Calculate and draw points */
    #pragma omp parallel for default(none) private(j, z, c, k, temp, lengthsq) shared(display, win, gc, screen) schedule(static)
    for (i = 0; i < X_RESN; i++) {
        for (j = 0; j < Y_RESN; j++) {
            z.real = z.imag = 0.0;
            c.real = ((float) j - 2000.0)/1000.0;          /* scale factors for 4000 x 4000 window */
            c.imag = ((float) i - 2000.0)/1000.0;
            k = 0;

            do {                                           /* iterate for pixel color */
                temp = z.real*z.real - z.imag*z.imag + c.real;
                z.imag = 2.0*z.real*z.imag + c.imag;
                z.real = temp;
                lengthsq = z.real*z.real + z.imag*z.imag;
                k++;
            } while (lengthsq < 4.0 && k < 100);

            if (k == 100) {
                #pragma omp critical
                {
                    unsigned long color = get_color_from_palette(i, j);
                    XSetForeground(display, gc, color);
                    XDrawPoint(display, win, gc, j, i);
                }                
            }
        }
    } // barreira implícita        
    
    double t_fim = omp_get_wtime();
    printf("Simulação concluída!\n");
    printf("Tamanho da imagem: %dx%d\n", X_RESN, Y_RESN);
    printf("Número de Threads: %d\n", omp_get_max_threads());
    printf("Tempo de execução: %.15lf\n", t_fim - t_inicio);
     
    XFlush(display);
    sleep(30);

    /* Program Finished */
}
