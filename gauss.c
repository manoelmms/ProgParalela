#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Parâmetros do Gauss-Seidel SOR
#define MAX_ITERACOES 4098
#define FATOR_RELAXAMENTO 0.5

// Configuração do problema
#define TAMANHO_GRID 1022
#define TEMP_INICIAL 20.0
#define TEMP_BORDA 20.0
#define TEMP_FONTE 100.0
#define LINHA_FONTE 800
#define COLUNA_FONTE 800

// Funções
void inicializar_grid();
double calcular_novo_valor(int linha, int coluna);

// Grids de temperatura
double* grid_atual = NULL;
double* grid_anterior = NULL;

int main(int argc, char* argv[]) {
    inicializar_grid();
    
    double tempo_inicio = omp_get_wtime();
    
    int iteracao = 0;
    while (iteracao < MAX_ITERACOES) {
        #pragma omp parallel default(none) shared(grid_atual)
        {
            // Padrão xadrez pra evitar problemas de concorrência
            // V: Vermelho    P: Preto
            //
            // V P V P V P V P V P
            // P V P V P V P V P V
            // V P V P V P V P V P
            // P V P V P V P V P V
            // V P V P V P V P V P
            // P V P V P V P V P V
            // V P V P V P V P V P
            // P V P V P V P V P V
            // V P V P V P V P V P
            // P V P V P V P V P V
            //
            // Os pontos vermelhos só leem dos pretos e vice-versa,
            // assim não tem problema de condição de corrida
            
            // Fase vermelha (atualiza vermelhos, lê dos pretos)
            #pragma omp for nowait schedule(static)
            for (int i = 1; i < TAMANHO_GRID - 1; i += 2) {
                for (int j = 1; j < TAMANHO_GRID - 1; j += 2) {
                    if (i != LINHA_FONTE || j != COLUNA_FONTE) {
                        grid_atual[i * TAMANHO_GRID + j] = calcular_novo_valor(i, j);
                    }
                }
            }
            
            #pragma omp for schedule(static)
            for (int i = 2; i < TAMANHO_GRID - 1; i += 2) {
                for (int j = 2; j < TAMANHO_GRID - 1; j += 2) {
                    if (i != LINHA_FONTE || j != COLUNA_FONTE) {
                        grid_atual[i * TAMANHO_GRID + j] = calcular_novo_valor(i, j);
                    }
                }
            } // Barreira automática aqui
            
            // Fase preta (atualiza pretos, lê dos vermelhos)
            #pragma omp for nowait schedule(static)
            for (int i = 1; i < TAMANHO_GRID - 1; i += 2) {
                for (int j = 2; j < TAMANHO_GRID - 1; j += 2) {
                    if (i != LINHA_FONTE || j != COLUNA_FONTE) {
                        grid_atual[i * TAMANHO_GRID + j] = calcular_novo_valor(i, j);
                    }
                }
            }
            
            #pragma omp for schedule(static)
            for (int i = 2; i < TAMANHO_GRID - 1; i += 2) {
                for (int j = 1; j < TAMANHO_GRID - 1; j += 2) {
                    if (i != LINHA_FONTE || j != COLUNA_FONTE) {
                        grid_atual[i * TAMANHO_GRID + j] = calcular_novo_valor(i, j);
                    }
                }
            } // Barreira automática aqui também
        }
        
        // Troca os ponteiros pros grids
        double* temp = grid_atual;
        grid_atual = grid_anterior;
        grid_anterior = temp;
        
        iteracao++;
    }
    
    // Calcula o erro final entre as duas últimas iterações
    double erro_maximo = 0.0;
    #pragma omp parallel for default(none) shared(grid_atual, grid_anterior) reduction(max:erro_maximo) schedule(static)
    for (int i = 1; i < TAMANHO_GRID - 1; i++) {
        for (int j = 1; j < TAMANHO_GRID - 1; j++) {
            double erro_local = fabs(grid_atual[i * TAMANHO_GRID + j] - grid_anterior[i * TAMANHO_GRID + j]);
            erro_maximo = fmax(erro_local, erro_maximo);
        }
    } // Barreira automática
    
    double tempo_fim = omp_get_wtime();
    
    printf("Simulação concluída!\n");
    printf("Tamanho do grid: %d x %d\n", TAMANHO_GRID, TAMANHO_GRID);
    printf("Número máximo de iterações: %d\n", MAX_ITERACOES);
    printf("Thread count: %d\n", omp_get_max_threads());
    printf("Depois de %d iterações, o erro foi: %.15lf\n", iteracao, erro_maximo);
    printf("Tempo de execução: %.15lf segundos\n", tempo_fim - tempo_inicio);
    
    // Libera a memória
    free(grid_atual);
    free(grid_anterior);
    
    return 0;
}

void inicializar_grid() {
    size_t tamanho_memoria = TAMANHO_GRID * TAMANHO_GRID * sizeof(double);
    
    grid_atual = (double*) malloc(tamanho_memoria);
    grid_anterior = (double*) malloc(tamanho_memoria);
    
    if (!grid_atual || !grid_anterior) {
        fprintf(stderr, "Não conseguiu alocar memória pros grids!\n");
        exit(1);
    }
    
    // Bordas laterais
    for (int i = 1; i < TAMANHO_GRID - 1; i++) {
        grid_anterior[i * TAMANHO_GRID] = TEMP_BORDA;                        // Borda esquerda
        grid_anterior[i * TAMANHO_GRID + TAMANHO_GRID - 1] = TEMP_BORDA;     // Borda direita
    }
    
    // Bordas de cima e de baixo
    for (int j = 0; j < TAMANHO_GRID; j++) {
        grid_anterior[j] = TEMP_BORDA;                                       // Borda de cima
        grid_anterior[(TAMANHO_GRID - 1) * TAMANHO_GRID + j] = TEMP_BORDA;  // Borda de baixo
    }
    
    // Pontos internos
    for (int i = 1; i < TAMANHO_GRID - 1; i++) {
        for (int j = 1; j < TAMANHO_GRID - 1; j++) {
            grid_anterior[i * TAMANHO_GRID + j] = TEMP_INICIAL;
        }
    }
    
    // Fonte de calor
    grid_anterior[LINHA_FONTE * TAMANHO_GRID + COLUNA_FONTE] = TEMP_FONTE;
    
    // Copia o estado inicial pro grid atual
    memcpy(grid_atual, grid_anterior, tamanho_memoria);
}

double calcular_novo_valor(int linha, int coluna) {
    // Calcula usando os 4 vizinhos (cima, baixo, esquerda, direita)
    double valor_gauss_seidel = 0.25 * (
        grid_anterior[(linha + 1) * TAMANHO_GRID + coluna] +     // Abaixo
        grid_atual[(linha - 1) * TAMANHO_GRID + coluna] +        // Acima
        grid_anterior[linha * TAMANHO_GRID + coluna + 1] +       // Direita
        grid_atual[linha * TAMANHO_GRID + coluna - 1]            // Esquerda
    );
    
    // Aplica a fórmula do SOR: novo = antigo + w * (gauss_seidel - antigo)
    // Otimizado pra fazer uma multiplicação a menos
    double valor_antigo = grid_anterior[linha * TAMANHO_GRID + coluna];
    return valor_antigo + FATOR_RELAXAMENTO * (valor_gauss_seidel - valor_antigo);
}