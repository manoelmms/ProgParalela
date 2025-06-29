#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // para medir o tempo de execução com 'omp_get_wtime()'

long n = -1;
double* mat1 = NULL;
double* mat2 = NULL;
double* prod = NULL;

unsigned int seed = 2025;

void mult_mat() {
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < n; ++j)
            for (long k = 0; k < n; ++k)
                prod[i*n + j] += mat1[i*n + k] * mat2[k*n + j];
}

void get_args(int argc, char* argv[]) {
    if (argc == 2) {
        n = atol(argv[1]);
    } else {
        fprintf(stderr, "Invalid Arguments\n");
        exit(1);
    }
}

void init() {
    size_t size = n*n * sizeof(double);
    mat1 = (double*) malloc(size);
    mat2 = (double*) malloc(size);
    prod = (double*) malloc(size);
    if (mat1 == NULL || mat2 == NULL || prod == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(2);
    }

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
    arquivo = fopen("mat_prod_naive", "wb");
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
    free(prod);
}

int main(int argc, char* argv[]) {
    get_args(argc, argv);
    init();

    double t_start = omp_get_wtime();

    mult_mat();

    double t_end = omp_get_wtime();
    //printf("%ld x %ld in %.15lf seconds\n", n, n, t_end - t_start);
    //Time to csv (t_end - t_start, n, 1, "naive_mul") to be used in the plot script
    printf("%.15lf,%ld,1,naive_mul\n", t_end - t_start, n);

    //export the product matrix to a binary file
    //exportar_bin();
    finalize();
    return 0;
}