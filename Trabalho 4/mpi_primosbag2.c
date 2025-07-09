#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#define TAMANHO 10000

int primo (int n) {
    int i;
    for (i = 3; i < (int)(sqrt(n) + 1); i+=2) {
        if(n%i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) { /* mpi_primosbag.c */
    double t_inicial, t_final;
    int cont = 0, total = 0;
    int i, n;
    int meu_ranque, num_procs, inicio, dest, raiz=0, tag=1, stop=0;
    int workers_ativos = 0;  // Contador de workers que receberam trabalho
    MPI_Status estado;
    
    /* Verifica o número de argumentos passados */
    if (argc < 2) {
        printf("Entre com o valor do maior inteiro como parâmetro para o programa.\n");
        return 0;
    } else {
        n = strtol(argv[1], (char **) NULL, 10);
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_ranque);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    /* Se houver menos que dois processos aborta */
    if (num_procs < 2) {
        printf("Este programa deve ser executado com no mínimo dois processos.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return(1);
    }
    
    /* Registra o tempo inicial de execução do programa */
    t_inicial = MPI_Wtime();
    
    /* Envia pedaços com TAMANHO números para cada processo */
    if (meu_ranque == 0) {
        // Fase 1: Distribui trabalho inicial para os workers disponíveis
        for (dest=1, inicio=3; dest < num_procs && inicio < n; dest++, inicio += TAMANHO) {
            MPI_Send(&inicio, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
            workers_ativos++;
        }
        
        // Envia sinal de terminação para workers que não receberam trabalho
        for (; dest < num_procs; dest++) {
            int dummy = 0;
            MPI_Send(&dummy, 1, MPI_INT, dest, 99, MPI_COMM_WORLD);
        }
        
        // Fase 2: Recebe resultados e distribui mais trabalho
        while (stop < workers_ativos) {
            MPI_Recv(&cont, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &estado);
            total += cont;
            dest = estado.MPI_SOURCE;
            
            if (inicio < n) {
                // Ainda há trabalho para distribuir
                MPI_Send(&inicio, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                inicio += TAMANHO;
            } else {
                // Não há mais trabalho, sinaliza terminação
                int dummy = 0;
                MPI_Send(&dummy, 1, MPI_INT, dest, 99, MPI_COMM_WORLD);
                stop++;
            }
        }
    }
    else {
        /* Cada processo escravo recebe o início do espaço de busca */
        while (1) {
            MPI_Recv(&inicio, 1, MPI_INT, raiz, MPI_ANY_TAG, MPI_COMM_WORLD, &estado);
            
            if (estado.MPI_TAG == 99) {
                // Sinal de terminação recebido
                break;
            }
            
            // Processa o pedaço de trabalho
            for (i = inicio, cont=0; i < (inicio + TAMANHO) && i < n; i+=2) {
                if (primo(i) == 1) {
                    cont++;
                }
            }
            
            /* Envia a contagem parcial para o processo mestre */
            MPI_Send(&cont, 1, MPI_INT, raiz, tag, MPI_COMM_WORLD);
        }
    }
    
    if (meu_ranque == 0) {
        t_final = MPI_Wtime();
        total += 1; /* Acrescenta o 2, que é primo */
        printf("%.15lf,%d,%d,primosbag2_mpi\n",
               t_final - t_inicial, n, num_procs);
    }
    
    /* Finaliza o programa */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return(0);
}