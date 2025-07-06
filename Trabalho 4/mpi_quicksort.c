#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "mpi.h"

// #define N 10000000 // Tamanho do vetor a ser ordenado
#define SEED 2025 // Semente para geração de números aleatórios

int correctude(int* vetor, int tamanho) {
    for (int i = 0; i < tamanho - 1; ++i) {
        if (vetor[i] > vetor[i + 1]) {
            return 0; // Vetor não está ordenado
        }
    }
    return 1; // Vetor está ordenado
}

int compara_int(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

void imprime_vetor(int rank, int* vet, int size) {
    printf("[%d]: [ ", rank);
    for (int i = 0; i < size; ++i)
        printf("%d, ", vet[i]);
    printf("]\n");
}

int main(int argc, char* argv[]) {
    int meu_rank, num_proc;
    int* vetor = NULL;
    int* resultado = NULL;
    int* scounts = NULL;
    int* displs = NULL;
    double tempo_inicializacao_inicio;
    double tempo_inicializacao_fim;
    long N;
    MPI_Request pedido;
    
    /* Verifica o número de argumentos passados */
	if (argc < 2) {
        printf("Entre com o valor do tamanho do vetor a ser ordenado como parâmetro para o programa.\n");
       	 return 0;
    } else {
        N = strtol(argv[1], (char **) NULL, 10);
    }
    
    // Inicialização MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    
    double tempo_inicial = MPI_Wtime();
    
    // Configuração da distribuição de trabalho
    scounts = (int*) malloc(num_proc * sizeof(int));
    displs = (int*) malloc(num_proc * sizeof(int));
    
    int div_trab = N / num_proc;
    for (int i = 0; i < num_proc; ++i)
        scounts[i] = div_trab;
    
    // Distribui elementos restantes entre os primeiros processos
    int res = N % num_proc;
    for (int i = 0; i < res; ++i)
        ++scounts[i];
    
    // Calcula deslocamentos para cada processo
    displs[0] = 0;
    for (int i = 1; i < num_proc; ++i)
        displs[i] = displs[i - 1] + scounts[i - 1];
    
    // Processo raiz: inicializa vetor com valores aleatórios
    if (meu_rank == 0) {
        tempo_inicializacao_inicio = MPI_Wtime();
        vetor = (int*) malloc(N * sizeof(int));
        srand(SEED); // Semente para geração de números aleatórios 
        // Gera números aleatórios entre 0 e 50000
        for (int i = 0; i < N; ++i)
            vetor[i] = rand() % 50001;
        tempo_inicializacao_fim = MPI_Wtime();

        printf("Tempo de inicialização do vetor: %1.10f segundos\n", 
               tempo_inicializacao_fim - tempo_inicializacao_inicio);
        
        MPI_Iscatterv(vetor, scounts, displs, MPI_INT, MPI_IN_PLACE, 
                     scounts[meu_rank], MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    } else {
        // Outros processos: aloca espaço para receber dados
        vetor = (int*) malloc(scounts[meu_rank] * sizeof(int));
        MPI_Iscatterv(vetor, scounts, displs, MPI_INT, vetor, 
                     scounts[meu_rank], MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    }
    
    MPI_Wait(&pedido, MPI_STATUS_IGNORE);
    
    // Cada processo ordena sua parte
    qsort(vetor, scounts[meu_rank], sizeof(int), compara_int);
    
    // Coleta os vetores ordenados no processo raiz
    if (meu_rank == 0)
        MPI_Igatherv(MPI_IN_PLACE, scounts[meu_rank], MPI_INT, vetor, 
                    scounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    else
        MPI_Igatherv(vetor, scounts[meu_rank], MPI_INT, vetor, 
                    scounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    
    MPI_Wait(&pedido, MPI_STATUS_IGNORE);
    
    // Merge final: processo raiz combina os vetores ordenados
    if (meu_rank == 0) {
        resultado = (int*) malloc(N * sizeof(int));
        
        for (int i = 0; i < N; ++i) {
            int menor = INT_MAX;
            int p_menor = -1;
            
            // Encontra o menor elemento entre os primeiros de cada processo
            for (int p = 0; p < num_proc; ++p) {
                int end = p * scounts[0] + scounts[p];
                if (displs[p] >= end)
                    continue;
                
                int valor = vetor[displs[p]];
                if (valor < menor) {
                    menor = valor;
                    p_menor = p;
                }
            }
            
            resultado[i] = menor;
            ++displs[p_menor];
        }
    }
    
    double tempo_final = MPI_Wtime();
    
    // Exibe resultados
    if (meu_rank == 0) {
        // printf("Número de processos: %d\n", num_proc);
        // printf("n = %d\n", N);
        // printf("Tempo total de execução: %1.10f\n", tempo_final - tempo_inicial);
        // printf("Tempo de execução (sem inicialização): %1.10f\n", 
        //        (tempo_final - tempo_inicial) - (tempo_inicializacao_fim - tempo_inicializacao_inicio));
        // printf("Vetor ordenado: %s\n", correctude(resultado, N) ? "SIM" : "NÃO");

        //Time to csv (t_end - t_start, n, 1, "strassen_seq") to be used in the plot script
        printf("%.15lf,%ld,%d,quicksort_mpi\n", 
               tempo_final - tempo_inicial - (tempo_inicializacao_fim - tempo_inicializacao_inicio), N, num_proc);
    }
    
    // Liberação de memória
    if (vetor != NULL) free(vetor);
    if (resultado != NULL) free(resultado);
    if (scounts != NULL) free(scounts);
    if (displs != NULL) free(displs);
    
    MPI_Finalize();
    return 0;
}