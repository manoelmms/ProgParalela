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

// Função para mesclar dois subarrays ordenados
void merge(int arr[], int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    int* L = (int*) malloc(n1 * sizeof(int));
    int* R = (int*) malloc(n2 * sizeof(int));
    
    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    i = 0; j = 0; k = left;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
    
    free(L);
    free(R);
}

// Implementação recursiva do MergeSort
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        
        merge(arr, left, mid, right);
    }
}

// Função para mesclar múltiplos arrays ordenados (merge k-way)
void mergeKArrays(int* vetor, int* scounts, int* displs, int num_proc, int* resultado, int total_size) {
    int* indices = (int*) calloc(num_proc, sizeof(int));
    
    for (int pos = 0; pos < total_size; pos++) {
        int menor = INT_MAX;
        int p_menor = -1;
        
        // Encontra o menor elemento entre os primeiros de cada array
        for (int p = 0; p < num_proc; p++) {
            if (indices[p] < scounts[p]) {
                int valor = vetor[displs[p] + indices[p]];
                if (valor < menor) {
                    menor = valor;
                    p_menor = p;
                }
            }
        }
        
        if (p_menor != -1) {
            resultado[pos] = menor;
            indices[p_menor]++;
        }
    }
    
    free(indices);
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
    
    // Cada processo ordena sua parte usando MergeSort
    mergeSort(vetor, 0, scounts[meu_rank] - 1);
    
    // Coleta os vetores ordenados no processo raiz
    if (meu_rank == 0)
        MPI_Igatherv(MPI_IN_PLACE, scounts[meu_rank], MPI_INT, vetor, 
                    scounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    else
        MPI_Igatherv(vetor, scounts[meu_rank], MPI_INT, vetor, 
                    scounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &pedido);
    
    MPI_Wait(&pedido, MPI_STATUS_IGNORE);
    
    // Merge final: processo raiz combina os vetores ordenados usando k-way merge
    if (meu_rank == 0) {
        resultado = (int*) malloc(N * sizeof(int));
        mergeKArrays(vetor, scounts, displs, num_proc, resultado, N);
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
        printf("%.15lf,%ld,%d,mergesort_mpi\n", 
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