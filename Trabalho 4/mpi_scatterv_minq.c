/* ------------------------------------------------------------------------
 * FILE: least-squares-pt2pt.c
 *  
 * PROBLEM DESCRIPTION:
 *  The method of least squares is a standard technique used to find
 *  the equation of a straight line from a set of data. Equation for a
 *  straight line is given by 
 *	 y = mx + b
 *  where m is the slope of the line and b is the y-intercept.
 *
 *  Given a set of n points {(x1,y1), x2,y2),...,xn,yn)}, let
 *      SUMx = x1 + x2 + ... + xn
 *      SUMy = y1 + y2 + ... + yn
 *      SUMxy = x1*y1 + x2*y2 + ... + xn*yn
 *      SUMxx = x1*x1 + x2*x2 + ... + xn*xn
 *
 *  The slope and y-intercept for the least-squares line can then be 
 *  calculated using the following equations:
 *        slope (m) = ( SUMx*SUMy - n*SUMxy ) / ( SUMx*SUMx - n*SUMxx ) 
 *  y-intercept (b) = ( SUMy - slope*SUMx ) / n
 *
 * PROGRAM DESCRIPTION:
 *  o This program computes a linear model for a set of given data.
 *  o Data is read from a file; the first line is the number of data 
 *    points (n), followed by the coordinates of x and y.
 *  o Data points are divided amongst processes such that each process
 *    has naverage = n/numprocs points; remaining data points are
 *    added to the last process. 
 *  o Each process calculates the partial sums (mySUMx,mySUMy,mySUMxy,
 *    mySUMxx) independently, using its data subset. In the final step,
 *    the global sums (SUMx,SUMy,SUMxy,SUMxx) are calculated to find
 *    the least-squares line. 
 *  o For the purpose of this exercise, communication is done strictly 
 *    by using the MPI point-to-point operations, MPI_SEND and MPI_RECV. 
 *  
 * USAGE: Tested to run using 1,2,...,10 processes.
 * 
 * AUTHOR: Dora Abdullah (MPI version, 11/96)
 * LAST REVISED: RYL converted to C (12/11)
 * ---------------------------------------------------------------------- */
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "mpi.h"

#define X 0
#define Y 1
#define XY 2
#define XX 3

int main(int argc, char **argv) {

  double *x, *y;
  double mySUM[4], SUM[4],
         SUMres, res, slope, y_intercept, y_estimate;
  int i,j,n,myid,numprocs,naverage,nremain,mypoints,ishift;
  int* scounts = NULL;
  int* displs = NULL;
  double tempo_inicial = 0.0;
  double tempo_final = 0.0;
  /*int new_sleep (int seconds);*/
  MPI_Status istatus;
  MPI_Request irequest;
  FILE *infile;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &myid);
  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);

  tempo_inicial = MPI_Wtime();

  /* ----------------------------------------------------------
   * Step 1: Process 0 reads data and sends the value of n
   * ---------------------------------------------------------- */
  if (myid == 0) {
    //printf ("Number of processes used: %d\n", numprocs);
    //printf ("-------------------------------------\n");
    //printf ("The x coordinates on worker processes:\n");
    /* this call is used to achieve a consistent output format */
    /* new_sleep (3);*/
    infile = fopen("xydata", "r");
    if (infile == NULL) printf("error opening file\n");
    fscanf (infile, "%d", &n);
    MPI_Ibcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD, &irequest);
    MPI_Wait(&irequest, &istatus);
    x = (double *) malloc (n*sizeof(double));
    y = (double *) malloc (n*sizeof(double));
    for (i=0; i<n; i++)
      fscanf (infile, "%lf %lf", &x[i], &y[i]);
    fclose(infile);
  }
  else {
    MPI_Ibcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD, &irequest);
    MPI_Wait(&irequest, &istatus);
    x = (double *) malloc (n*sizeof(double));
    y = (double *) malloc (n*sizeof(double));
  }
  /* ---------------------------------------------------------- */
  
  naverage = n/numprocs;
  nremain = n % numprocs;

  /* ----------------------------------------------------------
   * Step 2: Process 0 sends subsets of x and y 
!  * ---------------------------------------------------------- */
  MPI_Request pedido[2];
  if (myid == 0) {
    scounts = (int*) malloc(numprocs * sizeof(int));
    displs = (int*) malloc(numprocs * sizeof(int));
    scounts[0] = naverage;
    displs[0] = 0;
    for (i=1; i<numprocs; i++) {
      scounts[i] = (i < numprocs -1) ? naverage : naverage + nremain;
      displs[i] = i*naverage; 
    }
    MPI_Iscatterv(x, scounts, displs, MPI_DOUBLE, MPI_IN_PLACE, mypoints, MPI_DOUBLE, 0, MPI_COMM_WORLD, &pedido[0]);
    MPI_Iscatterv(y, scounts, displs, MPI_DOUBLE, MPI_IN_PLACE, mypoints, MPI_DOUBLE, 0, MPI_COMM_WORLD, &pedido[1]);
  }
  else {
    ishift = myid * naverage;
    mypoints = (myid < numprocs -1) ? naverage : naverage + nremain;
    /* ---------------the other processes receive---------------- */
    MPI_Iscatterv(x, scounts, displs, MPI_DOUBLE, &x[ishift], mypoints, MPI_DOUBLE, 0, MPI_COMM_WORLD, &pedido[0]);
    MPI_Iscatterv(y, scounts, displs, MPI_DOUBLE, &y[ishift], mypoints, MPI_DOUBLE, 0, MPI_COMM_WORLD, &pedido[1]);
    //printf ("id %d: ", myid);
    //for (i=0; i<n; i++) printf("%4.2lf ", x[i]);
    //printf ("\n");
    /* ---------------------------------------------------------- */
  }

  /* ----------------------------------------------------------
   * Step 3: Each process calculates its partial sum
   * ---------------------------------------------------------- */
  mySUM[X] = 0; mySUM[Y] = 0; mySUM[XY] = 0; mySUM[XX] = 0;
  if (myid == 0) {
    ishift = 0;
    mypoints = naverage;
  }

  MPI_Waitall(2, pedido, MPI_STATUSES_IGNORE);
  for (j=0; j<mypoints; j++) {
    mySUM[X] += x[ishift+j];
    mySUM[Y] += y[ishift+j];
    mySUM[XY] += x[ishift+j]*y[ishift+j];
    mySUM[XX] += x[ishift+j]*x[ishift+j];
  }
  
  /* ----------------------------------------------------------
   * Step 4: Process 0 receives partial sums from the others 
   * ---------------------------------------------------------- */
  MPI_Ireduce(&mySUM, &SUM, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &irequest);
  MPI_Wait(&irequest, &istatus);
  
  /* ----------------------------------------------------------
   * Step 5: Process 0 does the final steps
   * ---------------------------------------------------------- */
  if (myid == 0) {
    slope = ( SUM[X]*SUM[Y] - n*SUM[XY] ) / ( SUM[X]*SUM[X] - n*SUM[XX] );
    y_intercept = ( SUM[Y] - slope*SUM[X] ) / n;
    /* this call is used to achieve a consistent output format */
    /*new_sleep (3);*/
    //printf ("\n");
    printf ("The linear equation that best fits the given data:\n");
    printf ("       y = %6.2lfx + %6.2lf\n", slope, y_intercept);
    //printf ("--------------------------------------------------\n");
    //printf ("   Original (x,y)     Estimated y     Residual\n");
    //printf ("--------------------------------------------------\n");
    
    SUMres = 0;
    for (i=0; i<n; i++) {
      y_estimate = slope*x[i] + y_intercept;
      res = y[i] - y_estimate;
      SUMres = SUMres + res*res;
      //printf ("   (%6.2lf %6.2lf)      %6.2lf       %6.2lf\n", 
	    //  x[i], y[i], y_estimate, res);
    }

    tempo_final = MPI_Wtime();

    //printf("--------------------------------------------------\n");
    printf("Residual sum = %6.2lf\n", SUMres);
    printf("--------------------------------------------------\n");
    printf ("Number of processes used: %d\n", numprocs);
    printf("n = %d\n", n);
    printf("Tempo de execução = %6.10lf\n", tempo_final - tempo_inicial);
    printf("--------------------------------------------------\n");
  }

  /* ----------------------------------------------------------	*/
  free(x);
  free(y);
  if (myid == 0) {
    free(scounts);
    free(displs);
  }
  
  MPI_Finalize();
}