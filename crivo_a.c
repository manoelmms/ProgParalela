#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>

#define DEFAULT_N 1000000000 // 1 bilhão

int main(int argc, char** argv) {
    unsigned long total_primes = 0;
    unsigned long n, i, j;
    double start_time, end_time;
    
    // Processamento dos argumentos da linha de comando
    if (argc < 2) {
        printf("Sem parâmetros, assumindo n = %d\n", DEFAULT_N);
        n = DEFAULT_N;
    } else {
        n = strtol(argv[1], NULL, 10);
        if (n <= 2) {
            printf("Erro: n deve ser maior que 2\n");
            return 1;
        }
    }
    
    printf("Executando crivo para n = %lu\n", n);
    
    // Alocação do vetor de booleanos
    bool *is_prime = malloc(n * sizeof(bool));
    if (is_prime == NULL) {
        printf("Erro: falha na alocação de memória\n");
        return 1;
    }
    
    // Início da medição de tempo
    start_time = omp_get_wtime();
    
    // Inicialização: todos os números são considerados primos inicialmente
    memset(is_prime, true, n);
    
    // 0 e 1 não são primos
    if (n > 0) is_prime[0] = false;
    if (n > 1) is_prime[1] = false;
    
    // Crivo de Eratóstenes - apenas números ímpares a partir de 3
    unsigned long limit = (unsigned long) sqrt(n);
    
    for (i = 3; i <= limit; i += 2) {
        if (!is_prime[i]) continue; // Se já foi marcado como não primo, pula
        
        // Paralelização do loop interno para marcar múltiplos
        #pragma omp parallel for schedule(static)
        for (j = i * i; j < n; j += 2 * i) {
            is_prime[j] = false;
        }
    }
    
    // Contagem dos números primos com redução paralela
    // Conta o 2 separadamente
    if (n > 2) total_primes = 1; // o número 2 é primo
    
    #pragma omp parallel for reduction(+:total_primes) schedule(static)
    for (i = 3; i < n; i += 2) {
        if (is_prime[i]) {
            total_primes++;
        }
    }
    
    // Fim da medição de tempo
    end_time = omp_get_wtime();
    
    // Liberação da memória
    free(is_prime);
    
    // Exibição dos resultados
    printf("Total de primos encontrados: %lu\n", total_primes);
    printf("Tempo total de execução: %.3f segundos\n", end_time - start_time);
    printf("Número de threads utilizadas: %d\n", omp_get_max_threads());
    
    return 0;
}