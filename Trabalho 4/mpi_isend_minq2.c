#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "mpi.h"

#define X 0
#define Y 1
#define XY 2
#define XX 3

int main(int argc, char **argv) {

  double *x, *y, *my_x, *my_y;
  double mySUM[4], SUM[4],
         SUMres, res, slope, y_intercept, y_estimate;
  int i, j, n, myid, numprocs, naverage, nremain, mypoints;
  double tempo_inicial = 0.0;
  double tempo_final = 0.0;
  MPI_Request *send_requests;
  MPI_Status *send_statuses;
  FILE *infile;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  tempo_inicial = MPI_Wtime();

  /* ----------------------------------------------------------
   * Step 1: Processo 0 lê n e faz broadcast para todos
   * ---------------------------------------------------------- */
  if (myid == 0) {
    infile = fopen("xydata", "r");
    if (infile == NULL) {
      printf("erro ao abrir arquivo\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fscanf(infile, "%d", &n);
  }

  // Broadcast do valor de n para todos os processos
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Calcular distribuição de dados
  naverage = n / numprocs;
  nremain = n % numprocs;

  // Cada processo determina quantos pontos receberá
  mypoints = (myid < numprocs - 1) ? naverage : naverage + nremain;

  // Alocar memória para dados locais
  my_x = (double *) malloc(mypoints * sizeof(double));
  my_y = (double *) malloc(mypoints * sizeof(double));

  /* ----------------------------------------------------------
   * Step 2: Processo 0 lê todos os dados e usa MPI_Isend
   * para distribuir para os outros processos
   * ---------------------------------------------------------- */
  if (myid == 0) {
    // Processo 0 aloca memória para todos os dados
    x = (double *) malloc(n * sizeof(double));
    y = (double *) malloc(n * sizeof(double));
    
    // Lê todos os dados
    for (i = 0; i < n; i++) {
      fscanf(infile, "%lf %lf", &x[i], &y[i]);
    }
    fclose(infile);

    // Preparar arrays para MPI_Isend
    send_requests = (MPI_Request *) malloc(2 * (numprocs - 1) * sizeof(MPI_Request));
    send_statuses = (MPI_Status *) malloc(2 * (numprocs - 1) * sizeof(MPI_Status));
    int req_count = 0;

    // Enviar dados para outros processos usando MPI_Isend
    for (i = 1; i < numprocs; i++) {
      int start_index = i * naverage;
      int send_count = (i < numprocs - 1) ? naverage : naverage + nremain;
      
      // Envio não-bloqueante dos dados x
      MPI_Isend(&x[start_index], send_count, MPI_DOUBLE, i, 100, 
                MPI_COMM_WORLD, &send_requests[req_count]);
      req_count++;
      
      // Envio não-bloqueante dos dados y
      MPI_Isend(&y[start_index], send_count, MPI_DOUBLE, i, 200, 
                MPI_COMM_WORLD, &send_requests[req_count]);
      req_count++;
    }

    // Processo 0 copia seus próprios dados
    for (i = 0; i < naverage; i++) {
      my_x[i] = x[i];
      my_y[i] = y[i];
    }

    // Aguardar conclusão de todos os envios
    MPI_Waitall(req_count, send_requests, send_statuses);
    
    free(send_requests);
    free(send_statuses);
    free(x);
    free(y);
  } else {
    // Outros processos recebem seus dados
    MPI_Recv(my_x, mypoints, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(my_y, mypoints, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  /* ----------------------------------------------------------
   * Step 3: Cada processo calcula suas somas parciais
   * ---------------------------------------------------------- */
  mySUM[X] = 0; 
  mySUM[Y] = 0; 
  mySUM[XY] = 0; 
  mySUM[XX] = 0;

  for (j = 0; j < mypoints; j++) {
    mySUM[X] += my_x[j];
    mySUM[Y] += my_y[j];
    mySUM[XY] += my_x[j] * my_y[j];
    mySUM[XX] += my_x[j] * my_x[j];
  }

  /* ----------------------------------------------------------
   * Step 4: Usar MPI_Reduce para coletar somas parciais
   * ---------------------------------------------------------- */
  MPI_Reduce(mySUM, SUM, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  /* ----------------------------------------------------------
   * Step 5: Processo 0 calcula a linha de mínimos quadrados
   * ---------------------------------------------------------- */
  if (myid == 0) {
    slope = (SUM[X] * SUM[Y] - n * SUM[XY]) / (SUM[X] * SUM[X] - n * SUM[XX]);
    y_intercept = (SUM[Y] - slope * SUM[X]) / n;

    tempo_final = MPI_Wtime();

    // Saída para análise de desempenho
    printf("%.15lf,%ld,%d,isend_mpi\n", 
           tempo_final - tempo_inicial, (long)n, numprocs);
    
    // Saída opcional com resultados detalhados
    /*
    printf("Equação da reta: y = %.6lfx + %.6lf\n", slope, y_intercept);
    printf("Número de processos: %d\n", numprocs);
    printf("Número de pontos: %d\n", n);
    printf("Tempo de execução: %.10lf segundos\n", tempo_final - tempo_inicial);
    */
  }

  /* ----------------------------------------------------------
   * Limpeza e finalização
   * ---------------------------------------------------------- */
  free(my_x);
  free(my_y);
  
  MPI_Finalize();
  return 0;
}
