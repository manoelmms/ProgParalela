#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // para medir o tempo de execução com 'omp_get_wtime()'

// gcc -DL1D_CACHE_TAM=$(getconf LEVEL1_DCACHE_LINESIZE)
#define BLOCO_TAM (L1D_CACHE_TAM / sizeof(double))

long n = -1;
double* mat1 = NULL;
double* mat2 = NULL;
double* mat2_t = NULL;
double* prod = NULL;

unsigned int seed = 2025;

void transpose_mat(double* mat, double* mat_t) {
    for (long i = 0; i < n; ++i) {
        double* p_mat = mat + i;
        double* p_mat_t = mat_t + i*n;        
        for (long j = 0; j < n; ++j) {
            *p_mat_t = *p_mat;            
            p_mat += n;
            p_mat_t++;
        }
    }        
}

void mult_mat() {
    transpose_mat(mat2, mat2_t);

    long num = n / BLOCO_TAM;
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            for (int k = 0; k < BLOCO_TAM; ++k) {
                double* p_prod = prod + i*BLOCO_TAM*n + j*BLOCO_TAM + k*n; // pointer to the product block
                
                for (int m = 0; m < BLOCO_TAM; ++m) { 
                    double soma = 0.0;
                    
                    for (long r = 0; r < num; ++r) {
                        double* p_mat1 = mat1 + i*BLOCO_TAM*n + r*BLOCO_TAM + k*n; // pointer to the mat1 block
                        double* p_mat2_t = mat2_t + j*BLOCO_TAM*n + r*BLOCO_TAM + m*n; // pointer to the mat2_t block
                        
                        for (int p = 0; p < BLOCO_TAM; ++p) {
                            soma += (*p_mat1) * (*p_mat2_t); 
                            p_mat1++;
                            p_mat2_t++;
                        }
                    }
                    *p_prod = soma;
                    p_prod++;
                }
            }
        }
    }
}

void get_args(int argc, char* argv[]) {
    if (argc == 2) {
        n = atol(argv[1]);
    } else {
        fprintf(stderr, "Invalid Arguments\n");
        exit(1);
    }
}

void memory_allocate(double** mat, size_t size) {
    if (posix_memalign((void**) mat, L1D_CACHE_TAM, size)) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(2);
    }
}

void init() {
    size_t size = n*n * sizeof(double);
    memory_allocate(&mat1, size);
    memory_allocate(&mat2, size);
    memory_allocate(&mat2_t, size);
    memory_allocate(&prod, size);

    // create random matrices for mat1 and mat2
    // using a fixed seed for reproducibility
    srand(seed);
    double start_interval = -100.0;
    double end_interval = 100.0;
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j) {
            mat1[i*n + j] = (((double) rand() / RAND_MAX) * (end_interval - start_interval)) + start_interval;
            mat2[i*n + j] = (((double) rand() / RAND_MAX) * (end_interval - start_interval)) + start_interval;
        }

    // initialize product matrix to zero
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j)
            prod[i*n + j] = 0.0;
}

void exportar_bin() {
    FILE* arquivo;
    arquivo = fopen("mat_prod_op", "wb");
    if (arquivo == NULL) {
        fprintf(stderr, "Cant create file!\n");
        exit(3);
    }

    // dimensions
    fwrite(&n, sizeof(long), 1, arquivo);
    // elements of the product matrix
    fwrite(prod, sizeof(double), (n * n), arquivo);

    fclose(arquivo);
}

void finalize() {
    free(mat1);
    free(mat2);
    free(mat2_t);
    free(prod);
}

int main(int argc, char* argv[]) {
    get_args(argc, argv);
    init();

    double t_start = omp_get_wtime();

    mult_mat();

    double t_end = omp_get_wtime();
    //printf("%ld x %ld in %.15lf seconds\n", n, n, t_end - t_start);
    //Time to csv (t_end - t_start, n, 1, "optimized_seq") to be used in the plot script
    printf("%.15lf,%ld,1,optimized_seq\n", t_end - t_start, n);

    // export the product matrix to a binary file
    exportar_bin();
    finalize();
    return 0;
}