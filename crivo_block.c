#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>

#define N 1000000000 // 1 bilhão
#define CACHE_SIZE 32768 // 32KB - tamanho típico de cache L1

// Macro para converter número ímpar para índice no vetor
#define ODD_TO_INDEX(num) ((num - 3) / 2)
// Macro para converter índice no vetor para número ímpar
#define INDEX_TO_ODD(index) (2 * index + 3)

int main(int argc, char** argv)
{
    unsigned long total = 0, n, i, j;
    double start, end;

    if (argc < 2) {
        n = N;
    } else {
        n = strtol(argv[1], (char **) NULL, 10);
    }

    // Calculamos quantos números ímpares existem de 3 até n
    unsigned long max_odd = (n % 2 == 0) ? n - 1 : n;
    unsigned long odd_count = (max_odd - 1) / 2;
    
    // Vetor para armazenar apenas números ímpares >= 3
    bool *vector = malloc(odd_count * sizeof(bool)); 
    
    if (vector == NULL) {
        printf("Erro ao alocar memória\n");
        return 1;
    }

    // Início da medição
    start = omp_get_wtime();
    
    // Inicializando todos como primos (true)
    memset(vector, true, odd_count);

    int limite = (int) sqrt(n);
    
    // Tamanho do bloco baseado no cache (em elementos bool)
    unsigned long block_size = CACHE_SIZE / sizeof(bool);
    
    // Loop principal do crivo com cache blocking
    for (i = 3; i <= limite; i += 2) {
        unsigned long index_i = ODD_TO_INDEX(i);
        
        if (index_i >= odd_count || !vector[index_i]) continue;
        
        // Cache blocking: dividir o range em blocos
        #pragma omp parallel for schedule(dynamic)
        for (unsigned long block_start = i * i; block_start <= max_odd; block_start += block_size * 2 * i) {
            unsigned long block_end = block_start + block_size * 2 * i;
            if (block_end > max_odd) block_end = max_odd;
            
            // Processar o bloco atual
            for (j = block_start; j <= block_end; j += 2 * i) {
                // Garantir que j é múltiplo ímpar de i
                if (j < i * i) continue;
                
                unsigned long index_j = ODD_TO_INDEX(j);
                if (index_j < odd_count) {
                    vector[index_j] = false;
                }
            }
        }
    }
    
    // Contagem dos primos ímpares com cache blocking
    #pragma omp parallel for reduction(+:total) schedule(static, block_size)
    for (i = 0; i < odd_count; i++) {
        if (vector[i]) {
            total++;
        }
    }

    // Fim da medição
    end = omp_get_wtime();

    // Impressão do resultado
    printf("Total de primos: %lu | Tempo total do algoritmo: %.3f segundos\n", total + 1, end - start);
    printf("Total de threads: %d\n", omp_get_max_threads());
    printf("Throughput: %.2f números/segundo\n", (total + 1) / (end - start));
    printf("Eficiência de memória: %.2f MB\n", (odd_count * sizeof(bool)) / (1024.0 * 1024.0));
    printf("Tamanho do bloco: %lu elementos\n", block_size);

    // Libera a memória alocada
    free(vector);
    return 0;
}