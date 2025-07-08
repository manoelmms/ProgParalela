#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#define TAMANHO 500000

int primo (int n) {
    int i;
    for (i = 3; i < (int)(sqrt(n) + 1); i+=2) {
        if(n%i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    double t_inicial, t_final;
    int cont = 0, total = 0;
    int i, n;
    int meu_ranque, num_procs, inicio, dest, raiz=0, tag=1, stop=0;
    MPI_Status estado;
    
    if (argc < 2) {
        printf("Entre com o valor do maior inteiro como parâmetro para o programa.\n");
        return 0;
    } else {
        n = strtol(argv[1], (char **) NULL, 10);
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_ranque);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (num_procs < 2) {
        printf("Este programa deve ser executado com no mínimo dois processos.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return(1);
    }
    
    t_inicial = MPI_Wtime();
    
    if (meu_ranque == 0) {
        // *** ALTERAÇÃO 1: Comentário adicionado ***
        // Envia trabalho inicial para todos os processos
        for (dest=1, inicio=3; dest < num_procs && inicio < n; dest++, inicio += TAMANHO) {
            MPI_Send(&inicio, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        }
        
        // *** ALTERAÇÃO 2: Lógica de distribuição corrigida ***
        // Recebe resultados e distribui mais trabalho
        while (stop < (num_procs-1)) {
            MPI_Recv(&cont, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &estado);
            total += cont;
            dest = estado.MPI_SOURCE;
            
            if (inicio < n) {
                // *** ALTERAÇÃO 3: Condição corrigida (< ao invés de >) ***
                // Ainda há trabalho, envia mais
                MPI_Send(&inicio, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                inicio += TAMANHO;
            } else {
                // *** ALTERAÇÃO 4: Sinalização de parada corrigida ***
                // Não há mais trabalho, sinaliza parada
                MPI_Send(&inicio, 1, MPI_INT, dest, 99, MPI_COMM_WORLD);
                stop++;
            }
        }
        
        t_final = MPI_Wtime();
        total += 1; // Acrescenta o 2, que é primo
        printf("%.15lf,%d,%d,primosbag_mpi\n", t_final - t_inicial, n, num_procs);
        
    } else {
        // *** ALTERAÇÃO 5: Loop dos escravos simplificado ***
        // Processo escravo
        while (1) {
            MPI_Recv(&inicio, 1, MPI_INT, raiz, MPI_ANY_TAG, MPI_COMM_WORLD, &estado);
            
            if (estado.MPI_TAG == 99) {
                break; // *** ALTERAÇÃO 6: Saída explícita do loop ***
            }
            
            // Processa o bloco de números
            for (i = inicio, cont=0; i < (inicio + TAMANHO) && i < n; i+=2) {
                if (primo(i) == 1) {
                    cont++;
                }
            }
            
            // Envia resultado de volta
            MPI_Send(&cont, 1, MPI_INT, raiz, tag, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return(0);
}