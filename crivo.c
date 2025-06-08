#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>

#define N 1000000000 // 1 bilhão

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
    // Números ímpares: 3, 5, 7, 9, ..., n (se ímpar) ou n-1 (se par)
    unsigned long max_odd = (n % 2 == 0) ? n - 1 : n;
    unsigned long odd_count = (max_odd - 1) / 2; // quantidade de ímpares >= 3
    
    // Vetor para armazenar apenas números ímpares >= 3
    // Economiza ~50% da memória original
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
    
    // Loop principal do crivo - apenas sobre números ímpares
    for (i = 3; i <= limite; i += 2) {
        unsigned long index_i = ODD_TO_INDEX(i);
        
        if (index_i >= odd_count || !vector[index_i]) continue;
        
        // Paralelização do loop interno
        #pragma omp parallel for
        for (j = i * i; j <= max_odd; j += 2 * i) {
            unsigned long index_j = ODD_TO_INDEX(j);
            if (index_j < odd_count) {
                vector[index_j] = false;
            }
        }
    }
    
    // Contagem dos primos ímpares
    #pragma omp parallel for reduction(+:total)
    for (i = 0; i < odd_count; i++) {
        if (vector[i]) {
            total++;
        }
    }

    // Fim da medição
    end = omp_get_wtime();

    // Impressão do resultado
    // Somamos 1 ao total porque 2 também é primo
    printf("Total de primos: %lu | Tempo total do algoritmo: %.3f segundos\n", total + 1, end - start);
    printf("Total de threads: %d\n", omp_get_max_threads());
    printf("Throughput: %.2f números/segundo\n", (total + 1) / (end - start));
    printf("Eficiência de memória: %.2f MB\n", (odd_count * sizeof(bool)) / (1024.0 * 1024.0));

    // Libera a memória alocada
    free(vector);
    return 0;
}