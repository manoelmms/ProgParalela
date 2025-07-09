#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "mpi.h"

int main(int argc, char **argv) {

  double *x, *y, *local_x, *local_y;
  double mySUMx, mySUMy, mySUMxy, mySUMxx, SUMx, SUMy, SUMxy,
         SUMxx, SUMres, res, slope, y_intercept, y_estimate;
  int i,j,n,myid,numprocs,local_n;
  int *sendcounts, *displs;
  double t_inicial, t_final;
  FILE *infile;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &myid);
  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);

  /* ----------------------------------------------------------
   * Step 1: Process 0 reads data and prepares for scatter
   * ---------------------------------------------------------- */
  if (myid == 0) {
    infile = fopen("xydata", "r");
    if (infile == NULL) {
      printf("error opening file\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // printf ("Number of processes used: %d\n", numprocs);
    // printf ("-------------------------------------\n");
    
    fscanf (infile, "%d", &n);
    x = (double *) malloc (n*sizeof(double));
    y = (double *) malloc (n*sizeof(double));
    for (i=0; i<n; i++)
      fscanf (infile, "%lf %lf", &x[i], &y[i]);
    fclose(infile);
  }
  
  // Broadcast n to all processes
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
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
  
  // Synchronize all processes before starting timing
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
   * Step 4: Each process calculates its partial sum
   * ---------------------------------------------------------- */
  mySUMx = 0; mySUMy = 0; mySUMxy = 0; mySUMxx = 0;
  
  for (j = 0; j < local_n; j++) {
    mySUMx += local_x[j];
    mySUMy += local_y[j];
    mySUMxy += local_x[j] * local_y[j];
    mySUMxx += local_x[j] * local_x[j];
  }
  
  /* ----------------------------------------------------------
   * Step 5: Use MPI_Allreduce to collect partial sums
   * ---------------------------------------------------------- */
  MPI_Allreduce(&mySUMx, &SUMx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mySUMy, &SUMy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mySUMxy, &SUMxy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mySUMxx, &SUMxx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  // Synchronize before ending timing
  MPI_Barrier(MPI_COMM_WORLD);
  t_final = MPI_Wtime();

  /* ----------------------------------------------------------
   * Step 6: Calculate final results (all processes can do this)
   * ---------------------------------------------------------- */
  slope = ( SUMx*SUMy - n*SUMxy ) / ( SUMx*SUMx - n*SUMxx );
  y_intercept = ( SUMy - slope*SUMx ) / n;
  
  if (myid == 0) {
    // printf ("\n");
    // printf ("The linear equation that best fits the given data:\n");
    // printf ("       y = %6.2lfx + %6.2lf\n", slope, y_intercept);
    // printf ("--------------------------------------------------\n");
    // printf ("Execution time: %.15lf seconds\n", t_final - t_inicial);
    // printf ("--------------------------------------------------\n");
    // printf ("   Original (x,y)     Estimated y     Residual\n");
    // printf ("--------------------------------------------------\n");
    // Saída para análise de desempenho
    printf("%.15lf,%d,%d,minq_mpi\n", 
           t_final - t_inicial, n, numprocs);

    // SUMres = 0;
    // for (i=0; i<n; i++) {
    //   y_estimate = slope*x[i] + y_intercept;
    //   res = y[i] - y_estimate;
    //   SUMres = SUMres + res*res;
    //   printf ("   (%6.2lf %6.2lf)      %6.2lf       %6.2lf\n", 
    //       x[i], y[i], y_estimate, res);
    // }
    // printf("--------------------------------------------------\n");
    // printf("Residual sum = %6.2lf\n", SUMres);
    
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