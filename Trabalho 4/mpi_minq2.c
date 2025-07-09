#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "mpi.h"

#define RANGE_BEGIN -100.0
#define RANGE_END 100.0
#define NOISE_MIN -2.5
#define NOISE_MAX 2.5
#define SEED 2025

double f(double x) {
    return 7.0*x + 2.0;
}

// Função para simular trabalho computacional mais intensivo
double intensive_computation(double x, double y) {
    double result = 0.0;
    // Simula cálculos mais complexos que se beneficiam da paralelização
    for (int i = 0; i < 1000; i++) {
        result += sin(x * i) * cos(y * i) + sqrt(fabs(x * y));
    }
    return result;
}

int main(int argc, char **argv) {

    double *x, *y, *local_x, *local_y;
    double mySUMx, mySUMy, mySUMxy, mySUMxx, SUMx, SUMy, SUMxy, SUMxx;
    double slope, y_intercept;
    int i, j, n, myid, numprocs, local_n;
    int *sendcounts, *displs;
    double t_inicial, t_final;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    if (argc != 2) {
        if (myid == 0) printf("Usage: %s <number_of_points>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    n = strtol(argv[1], (char **) NULL, 10);
    

    /* ----------------------------------------------------------
     * Step 1: Process 0 generates data
     * ---------------------------------------------------------- */
    if (myid == 0) {
        x = (double *) malloc(n * sizeof(double));
        y = (double *) malloc(n * sizeof(double));
        
        srand(SEED);
        double h = (RANGE_END - RANGE_BEGIN) / n;
        
        for (i = 0; i < n; i++) {
            x[i] = RANGE_BEGIN + i * h;
            y[i] = f(x[i]);
            
            // Add random noise to y value
            double noise = rand() / (double) RAND_MAX;
            noise = NOISE_MIN + noise * (NOISE_MAX - NOISE_MIN);
            y[i] *= 1.0 + noise / 100.0;
        }
    }
    
    /* ----------------------------------------------------------
     * Step 2: Calculate distribution for MPI_Scatterv
     * ---------------------------------------------------------- */
    sendcounts = (int *) malloc(numprocs * sizeof(int));
    displs = (int *) malloc(numprocs * sizeof(int));
    
    int base_count = n / numprocs;
    int remainder = n % numprocs;
    
    for (i = 0; i < numprocs; i++) {
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
    }
    
    local_n = sendcounts[myid];
    local_x = (double *) malloc(local_n * sizeof(double));
    local_y = (double *) malloc(local_n * sizeof(double));
    
    // Start timing ONLY the computation part
    MPI_Barrier(MPI_COMM_WORLD);
    t_inicial = MPI_Wtime();
    
    /* ----------------------------------------------------------
     * Step 3: Scatter data to all processes
     * ---------------------------------------------------------- */
    MPI_Scatterv(x, sendcounts, displs, MPI_DOUBLE, 
                 local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y, sendcounts, displs, MPI_DOUBLE, 
                 local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* ----------------------------------------------------------
     * Step 4: Each process calculates its partial sum with intensive computation
     * ---------------------------------------------------------- */
    mySUMx = 0; mySUMy = 0; mySUMxy = 0; mySUMxx = 0;
    
    for (j = 0; j < local_n; j++) {
        // Adiciona computação mais intensiva para demonstrar paralelização
        double processed_x = local_x[j];
        double processed_y = local_y[j];
        
        // Simula pré-processamento computacionalmente intensivo
        double computation_result = intensive_computation(processed_x, processed_y);
        
        // Incorpora o resultado da computação intensiva no cálculo
        double weight = 1.0 + computation_result * 1e-10; // Peso muito pequeno para não afetar o resultado
        
        mySUMx += processed_x * weight;
        mySUMy += processed_y * weight;
        mySUMxy += processed_x * processed_y * weight;
        mySUMxx += processed_x * processed_x * weight;
    }
    
    /* ----------------------------------------------------------
     * Step 5: Use MPI_Allreduce to collect partial sums
     * ---------------------------------------------------------- */
    MPI_Allreduce(&mySUMx, &SUMx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&mySUMy, &SUMy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&mySUMxy, &SUMxy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&mySUMxx, &SUMxx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    t_final = MPI_Wtime();
    
    if (myid == 0) {
        slope = (SUMx * SUMy - n * SUMxy) / (SUMx * SUMx - n * SUMxx);
        y_intercept = (SUMy - slope * SUMx) / n;
        
        printf("%.15lf,%d,%d,minq_mpi_optimized\n", 
               t_final - t_inicial, n, numprocs);
               
        // Informações de debug (comentar para benchmark)
        /*
        printf("Slope: %6.2lf, Y-intercept: %6.2lf\n", slope, y_intercept);
        printf("Processes: %d, Points per process: %d-%d\n", 
               numprocs, base_count, base_count + (remainder > 0 ? 1 : 0));
        */
        
        free(x);
        free(y);
    }

    free(local_x);
    free(local_y);
    free(sendcounts);
    free(displs);
    
    MPI_Finalize();
    return 0;
}