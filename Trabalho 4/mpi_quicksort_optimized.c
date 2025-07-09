#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include "mpi.h"

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

// Merge otimizado usando min-heap para k vetores ordenados
typedef struct {
    int value;
    int proc_id;
    int index;
} HeapNode;

void heapify(HeapNode heap[], int n, int i) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && heap[left].value < heap[smallest].value)
        smallest = left;
    
    if (right < n && heap[right].value < heap[smallest].value)
        smallest = right;
    
    if (smallest != i) {
        HeapNode temp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = temp;
        heapify(heap, n, smallest);
    }
}

void build_heap(HeapNode heap[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(heap, n, i);
}

// Merge k-way otimizado usando heap
void merge_k_sorted_arrays(int* vetor, int* scounts, int* displs, int num_proc, int N, int* resultado) {
    HeapNode* heap = (HeapNode*)malloc(num_proc * sizeof(HeapNode));
    int heap_size = 0;
    
    // Inicializa heap com primeiro elemento de cada processo
    for (int i = 0; i < num_proc; i++) {
        if (scounts[i] > 0) {
            heap[heap_size].value = vetor[displs[i]];
            heap[heap_size].proc_id = i;
            heap[heap_size].index = 0;
            heap_size++;
        }
    }
    
    build_heap(heap, heap_size);
    
    // Merge usando heap
    for (int i = 0; i < N; i++) {
        // Pega o menor elemento do heap
        HeapNode min_node = heap[0];
        resultado[i] = min_node.value;
        
        // Avança o índice do processo correspondente
        min_node.index++;
        
        // Se ainda há elementos neste processo, atualiza o heap
        if (min_node.index < scounts[min_node.proc_id]) {
            heap[0].value = vetor[displs[min_node.proc_id] + min_node.index];
            heap[0].index = min_node.index;
            heapify(heap, heap_size, 0);
        } else {
            // Remove este processo do heap
            heap[0] = heap[heap_size - 1];
            heap_size--;
            if (heap_size > 0)
                heapify(heap, heap_size, 0);
        }
    }
    
    free(heap);
}

int main(int argc, char* argv[]) {
    int meu_rank, num_proc;
    int* vetor = NULL;
    int* vetor_local = NULL;
    int* resultado = NULL;
    int* scounts = NULL;
    int* displs = NULL;
    double tempo_inicializacao_inicio = 0;
    double tempo_inicializacao_fim = 0;
    long N;
    
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
    
    // Sincronização antes de iniciar medição
    MPI_Barrier(MPI_COMM_WORLD);
    double tempo_inicial = MPI_Wtime();
    
    // Configuração da distribuição de trabalho
    scounts = (int*) malloc(num_proc * sizeof(int));
    displs = (int*) malloc(num_proc * sizeof(int));
    
    int div_trab = N / num_proc;
    int res = N % num_proc;
    
    for (int i = 0; i < num_proc; ++i) {
        scounts[i] = div_trab + (i < res ? 1 : 0);
    }
    
    // Calcula deslocamentos para cada processo
    displs[0] = 0;
    for (int i = 1; i < num_proc; ++i)
        displs[i] = displs[i - 1] + scounts[i - 1];
    
    // Aloca vetor local para cada processo
    vetor_local = (int*) malloc(scounts[meu_rank] * sizeof(int));
    
    // Processo raiz: inicializa vetor com valores aleatórios
    if (meu_rank == 0) {
        tempo_inicializacao_inicio = MPI_Wtime();
        vetor = (int*) malloc(N * sizeof(int));
        srand(SEED); // Semente para geração de números aleatórios 
        // Gera números aleatórios entre 0 e 50000
        for (int i = 0; i < N; ++i)
            vetor[i] = rand() % 50001;
        tempo_inicializacao_fim = MPI_Wtime();
    }
    
    // Distribui dados usando Scatterv (síncrono e mais eficiente)
    MPI_Scatterv(vetor, scounts, displs, MPI_INT, vetor_local, 
                 scounts[meu_rank], MPI_INT, 0, MPI_COMM_WORLD);
    
    // Cada processo ordena sua parte
    qsort(vetor_local, scounts[meu_rank], sizeof(int), compara_int);
    
    // Coleta os vetores ordenados no processo raiz
    if (meu_rank == 0) {
        // Realoca vetor para receber dados ordenados
        if (vetor) free(vetor);
        vetor = (int*) malloc(N * sizeof(int));
    }
    
    MPI_Gatherv(vetor_local, scounts[meu_rank], MPI_INT, vetor, 
                scounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Merge final otimizado: apenas processo raiz
    if (meu_rank == 0) {
        resultado = (int*) malloc(N * sizeof(int));
        merge_k_sorted_arrays(vetor, scounts, displs, num_proc, N, resultado);
    }
    
    // Sincronização antes de finalizar medição
    MPI_Barrier(MPI_COMM_WORLD);
    double tempo_final = MPI_Wtime();
    
    // Exibe resultados
    if (meu_rank == 0) {
        double tempo_computacao = tempo_final - tempo_inicial;
        double tempo_sem_init = tempo_computacao - (tempo_inicializacao_fim - tempo_inicializacao_inicio);
        
        // Verificação de correção (opcional para debugging)
        int esta_ordenado = correctude(resultado, N);
        
        // Saída formatada para CSV
        printf("%.15lf,%ld,%d,quicksort_mpi_optimized\n", 
               tempo_sem_init, N, num_proc);
        
        // Para debugging (descomente se necessário)
        /*
        printf("Número de processos: %d\n", num_proc);
        printf("n = %ld\n", N);
        printf("Tempo total de execução: %1.10f\n", tempo_computacao);
        printf("Tempo de execução (sem inicialização): %1.10f\n", tempo_sem_init);
        printf("Vetor ordenado: %s\n", esta_ordenado ? "SIM" : "NÃO");
        */
    }
    
    // Liberação de memória
    if (vetor != NULL) free(vetor);
    if (vetor_local != NULL) free(vetor_local);
    if (resultado != NULL) free(resultado);
    if (scounts != NULL) free(scounts);
    if (displs != NULL) free(displs);
    
    MPI_Finalize();
    return 0;
}