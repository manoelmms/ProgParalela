#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RANGE_BEGIN -100.0
#define RANGE_END 100.0
#define NOISE_MIN -2.5
#define NOISE_MAX 2.5
#define SEED 2025

double f(double x) {
    return 7.0*x + 2.0;
}

int main(int argc, char* argv[]) {
    int n;
    
    // Argument validation
    if (argc != 2) {
        printf("Missing argument or invalid argument!\n");
        printf("Usage: %s <num>\n", argv[0]);
        printf("\t<num> integer > 0\n");
        return 1;
    }
    
    n = atoi(argv[1]);
    if (n <= 0) {
        printf("Error: Number must be positive\n");
        return 1;
    }
    
    // Open file for writing
    FILE* file = fopen("xydata", "w");
    if (file == NULL) {
        printf("Could not open file!\n");
        return 2;
    }
    
    // Write number of points to file
    fprintf(file, "%d ", n);
    
    // Generate data with noise
    srand(SEED);
    double h = (RANGE_END - RANGE_BEGIN) / n;
    
    for (double x = RANGE_BEGIN; x <= RANGE_END; x += h) {
        double y = f(x);
        
        // Add random noise to y value
        double noise = rand() / (double) RAND_MAX;
        noise = NOISE_MIN + noise * (NOISE_MAX - NOISE_MIN);
        y *= 1.0 + noise / 100.0;
        
        fprintf(file, "%lf %lf ", x, y);
    }
    
    fclose(file);
    return 0;
}