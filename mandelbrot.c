/* OpenMP Parallel Colored Mandelbrot program */

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>

#define X_RESN  1600      /* increased resolution for more computation */
#define Y_RESN  1600
#define MAX_ITER 256      /* increased iterations for better color */

typedef struct complextype {
    float real, imag;
} Compl;

/* Global variables for X11 */
Display *g_display;
Window g_win;
GC g_gc;
int g_screen;
unsigned long colors[MAX_ITER];

/* Initialize color palette */
void init_colors(Display *display, int screen) {
    Colormap colormap = DefaultColormap(display, screen);
    XColor color;
    
    for (int i = 0; i < MAX_ITER; i++) {
        if (i == MAX_ITER - 1) {
            colors[i] = BlackPixel(display, screen);
        } else {
            float ratio = (float)i / MAX_ITER;
            color.red = (unsigned short)(65535 * (0.5 + 0.5 * sin(ratio * 3.14159 * 3)));
            color.green = (unsigned short)(65535 * (0.5 + 0.5 * sin(ratio * 3.14159 * 3 + 2)));
            color.blue = (unsigned short)(65535 * (0.5 + 0.5 * sin(ratio * 3.14159 * 3 + 4)));
            color.flags = DoRed | DoGreen | DoBlue;
            
            if (XAllocColor(display, colormap, &color)) {
                colors[i] = color.pixel;
            } else {
                colors[i] = BlackPixel(display, screen);
            }
        }
    }
}

int main() {
    Window win;
    unsigned int width, height, x, y, border_width;
    unsigned int display_width, display_height, screen;
    char *window_name = "OpenMP Parallel Mandelbrot Set", *display_name = NULL;
    GC gc;
    unsigned long valuemask = 0;
    XGCValues values;
    Display *display;
    XSizeHints size_hints;
    XSetWindowAttributes attr[1];
    
    /* Mandelbrot variables */
    int i, j, k;
    Compl z, c;
    float lengthsq, temp;
    double start_time, end_time;
    
    printf("OpenMP Parallel Mandelbrot Set Generator\n");
    printf("Number of threads available: %d\n", omp_get_max_threads());
    
    /* Connect to X server */
    if ((display = XOpenDisplay(display_name)) == NULL) {
        fprintf(stderr, "Cannot connect to X server %s\n", XDisplayName(display_name));
        exit(-1);
    }
    
    /* Get screen size */
    screen = DefaultScreen(display);
    g_screen = screen;
    g_display = display;
    display_width = DisplayWidth(display, screen);
    display_height = DisplayHeight(display, screen);
    
    /* Initialize colors */
    init_colors(display, screen);
    
    /* Set window size and position */
    width = X_RESN;
    height = Y_RESN;
    x = 0;
    y = 0;
    
    /* Create window */
    border_width = 4;
    win = XCreateSimpleWindow(display, RootWindow(display, screen),
                             x, y, width, height, border_width,
                             BlackPixel(display, screen), WhitePixel(display, screen));
    
    g_win = win;
    
    size_hints.flags = USPosition | USSize;
    size_hints.x = x;
    size_hints.y = y;
    size_hints.width = width;
    size_hints.height = height;
    size_hints.min_width = 300;
    size_hints.min_height = 300;
    
    XSetNormalHints(display, win, &size_hints);
    XStoreName(display, win, window_name);
    
    /* Create graphics context */
    gc = XCreateGC(display, win, valuemask, &values);
    g_gc = gc;
    
    XSetBackground(display, gc, WhitePixel(display, screen));
    XSetForeground(display, gc, BlackPixel(display, screen));
    XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);
    
    attr[0].backing_store = Always;
    attr[0].backing_planes = 1;
    attr[0].backing_pixel = BlackPixel(display, screen);
    
    XChangeWindowAttributes(display, win, CWBackingStore | CWBackingPlanes | CWBackingPixel, attr);
    XMapWindow(display, win);
    XSync(display, 0);
    
    printf("Starting calculation...\n");
    printf("Image size: %dx%d, Max iterations: %d\n", X_RESN, Y_RESN, MAX_ITER);
    
    /* Start timing */
    start_time = omp_get_wtime();
    
    /* PARALLEL MANDELBROT CALCULATION WITH OPENMP */
    #pragma omp parallel for private(i, j, k, z, c, lengthsq, temp) schedule(dynamic, 10)
    for(i = 0; i < Y_RESN; i++) {
        
        /* Print progress from thread 0 */
        if (omp_get_thread_num() == 0 && i % 100 == 0) {
            printf("Progress: %.1f%% (Thread %d/%d)\n", 
                   (float)i/Y_RESN*100, omp_get_thread_num(), omp_get_num_threads());
        }
        
        for(j = 0; j < X_RESN; j++) {
            z.real = z.imag = 0.0;
            c.real = ((float) j - X_RESN/2.0) / (X_RESN/4.0);  /* scale to complex plane */
            c.imag = ((float) i - Y_RESN/2.0) / (Y_RESN/4.0);
            k = 0;
            
            /* Mandelbrot iteration */
            do {
                temp = z.real*z.real - z.imag*z.imag + c.real;
                z.imag = 2.0*z.real*z.imag + c.imag;
                z.real = temp;
                lengthsq = z.real*z.real + z.imag*z.imag;
                k++;
            } while (lengthsq < 4.0 && k < MAX_ITER);
            
            /* Critical section: only one thread can draw at a time */
            #pragma omp critical(drawing)
            {
                XSetForeground(display, gc, colors[k]);
                XDrawPoint(display, win, gc, j, i);
            }
        }
        
        /* Periodic flush to show progress */
        if (i % 50 == 0) {
            #pragma omp critical(flushing)
            {
                XFlush(display);
            }
        }
    }
    
    /* End timing */
    end_time = omp_get_wtime();
    
    printf("\nCalculation completed!\n");
    printf("Execution time: %.2f seconds\n", end_time - start_time);
    printf("Threads used: %d\n", omp_get_max_threads());
    
    /* Final flush */
    XFlush(display);
    
    /* Keep window open */
    printf("Window will close in 30 seconds...\n");
    sleep(30);
    
    /* Cleanup */
    XCloseDisplay(display);
    
    return 0;
}