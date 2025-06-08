#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define COLUMNS    1022
#define ROWS       1022
#define MAX_TEMP_ERROR 0.01
#define MAX_ITERATIONS 4098
#define INITIAL_TEMP 20.0
#define HEAT_SOURCE_TEMP 100.0
#define HEAT_SOURCE_X 800
#define HEAT_SOURCE_Y 800

double Anew[ROWS+2][COLUMNS+2];
double A[ROWS+2][COLUMNS+2];

void iniciar();

int main(int argc, char *argv[]) {
    int i, j;
    int iteration = 1;
    double dt = 100.0;
    double start_time, end_time;
    
    int thread_counts[] = {1, 8, 16, 32, 64};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    printf("2D Heat Transfer Simulation - Jacobi Method\n\n");
    
    for (int test = 0; test < num_tests; test++) {
        omp_set_num_threads(thread_counts[test]);
        
        iniciar();
        iteration = 1;
        dt = 100.0;
        
        printf("Running with %d thread(s)...\n", thread_counts[test]);
        start_time = omp_get_wtime();
        
        while (dt > MAX_TEMP_ERROR && iteration <= MAX_ITERATIONS) {
            // Calculate new temperatures
            #pragma omp parallel for private(j) schedule(static)
            for (i = 1; i <= ROWS; i++) {
                for (j = 1; j <= COLUMNS; j++) {
                    if (i != HEAT_SOURCE_Y || j != HEAT_SOURCE_X) {
                        Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + 
                                            A[i][j+1] + A[i][j-1]);
                    } else {
                        Anew[i][j] = HEAT_SOURCE_TEMP;
                    }
                }
            }
            
            dt = 0.0;
            
            // Update grid and find maximum change
            #pragma omp parallel for private(j) reduction(max:dt) schedule(static)
            for (i = 1; i <= ROWS; i++) {
                for (j = 1; j <= COLUMNS; j++) {
                    double temp_diff = fabs(Anew[i][j] - A[i][j]);
                    if (temp_diff > dt) {
                        dt = temp_diff;
                    }
                    A[i][j] = Anew[i][j];
                }
            }
            
            iteration++;
            
            if (iteration % 500 == 0) {
                printf("  Iteration %d, max error: %.6f\n", iteration-1, dt);
            }
        }
        
        end_time = omp_get_wtime();
        
        printf("  Time: %.3f seconds\n", end_time - start_time);
        printf("  Iterations: %d\n", iteration-1);
        printf("  Final error: %.6f\n\n", dt);

        // Optional: Print a small section of the grid for verification
        if (ROWS <= 10 && COLUMNS <= 10) {
            printf("  Final grid section:\n");
            for (i = 1; i <= ROWS; i++) {
                for (j = 1; j <= COLUMNS; j++) {
                    printf("%.2f ", A[i][j]);
                }
                printf("\n");
            }
        }
        printf("--------------------------------------------------\n");
    }
    
    return 0;
}

void iniciar() {
    int i, j;
    
    #pragma omp parallel for private(j)
    for (i = 0; i <= ROWS+1; i++) {
        for (j = 0; j <= COLUMNS+1; j++) {
            A[i][j] = INITIAL_TEMP;
            Anew[i][j] = INITIAL_TEMP;
        }
    }
    
    // Set borders
    #pragma omp parallel for
    for (i = 0; i <= ROWS+1; i++) {
        A[i][0] = INITIAL_TEMP;
        A[i][COLUMNS+1] = INITIAL_TEMP;
        Anew[i][0] = INITIAL_TEMP;
        Anew[i][COLUMNS+1] = INITIAL_TEMP;
    }
    
    #pragma omp parallel for
    for (j = 0; j <= COLUMNS+1; j++) {
        A[0][j] = INITIAL_TEMP;
        A[ROWS+1][j] = INITIAL_TEMP;
        Anew[0][j] = INITIAL_TEMP;
        Anew[ROWS+1][j] = INITIAL_TEMP;
    }
    
    // Set heat source
    A[HEAT_SOURCE_Y][HEAT_SOURCE_X] = HEAT_SOURCE_TEMP;
    Anew[HEAT_SOURCE_Y][HEAT_SOURCE_X] = HEAT_SOURCE_TEMP;
}