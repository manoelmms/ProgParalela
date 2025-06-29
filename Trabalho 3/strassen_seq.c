#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // para medir o tempo de execução com 'omp_get_wtime()'

// gcc -DL1D_CACHE_TAM=$(getconf LEVEL1_DCACHE_LINESIZE)
#define BLOCO_TAM (L1D_CACHE_TAM / sizeof(double))

#define STRASSEN_THRESHOLD 1000 // threshold for switching to Strassen's algorithm

long n = -1;
double* mat1 = NULL;
double* mat2 = NULL;
double* prod = NULL;

unsigned int seed = 2025;

void memory_allocate(double** mat, size_t size);
void get_args(int argc, char* argv[]);
void init();
void exportar_bin();
void finalize();
void strassen(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _prod, long _prod_jmp);
void transpose_mat(long _n, double* _mat, long _mat_jmp, double* _mat_t, long _mat_t_jmp);
void add_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _adic, long _adic_jmp);
void sub_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _subt, long _subt_jmp);
void mult_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _prod, long _prod_jmp);

void transpose_mat(long _n, double* _mat, long _mat_jmp, double* _mat_t, long _mat_t_jmp) {
    for (long i = 0; i < _n; ++i) {
        double* p_mat = _mat + i;
        double* p_mat_t = _mat_t + i*_mat_t_jmp;        
        
        for (long j = 0; j < _n; ++j) {
            *p_mat_t = *p_mat;            
            p_mat += _mat_jmp;
            p_mat_t++;
        }
    }        
}

void add_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _adic, long _adic_jmp) {
    for (long i = 0; i < _n; ++i) {
        double* p_mat1 = _mat1 + i*_mat1_jmp;
        double* p_mat2 = _mat2 + i*_mat2_jmp;
        double* p_adic = _adic + i*_adic_jmp;
        
        for (long j = 0; j < _n; ++j) {
            *p_adic = *p_mat1 + *p_mat2;
            p_mat1++;
            p_mat2++;
            p_adic++;
        }        
    }
}

void sub_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _subt, long _subt_jmp) {
    for (long i = 0; i < _n; ++i) {
        double* p_mat1 = _mat1 + i*_mat1_jmp;
        double* p_mat2 = _mat2 + i*_mat2_jmp;
        double* p_subt = _subt + i*_subt_jmp;
        
        for (long j = 0; j < _n; ++j) {
            *p_subt = *p_mat1 - *p_mat2;
            p_mat1++;
            p_mat2++;
            p_subt++;
        }        
    }
}

void mult_mat(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _prod, long _prod_jmp) {
    double* _mat2_t = NULL;
    memory_allocate(&_mat2_t, _n*_n * sizeof(double));

    transpose_mat(_n, _mat2, _mat2_jmp, _mat2_t, _n);

    long num = _n / BLOCO_TAM;
    for (long i = 0; i < num; ++i) {
        for (long j = 0; j < num; ++j) {
            for (int k = 0; k < BLOCO_TAM; ++k) {
                double* p_prod = _prod + i*BLOCO_TAM*_prod_jmp + j*BLOCO_TAM + k*_prod_jmp;
                
                for (int m = 0; m < BLOCO_TAM; ++m) {
                    double soma = 0.0;
                    
                    for (long r = 0; r < num; ++r) {
                        double* p_mat1 = _mat1 + i*BLOCO_TAM*_mat1_jmp + r*BLOCO_TAM + k*_mat1_jmp;
                        double* p_mat2_t = _mat2_t + j*BLOCO_TAM*_n + r*BLOCO_TAM + m*_n;
                        
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
    free(_mat2_t);
}

void strassen(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _prod, long _prod_jmp) {
    
    if (_n <= STRASSEN_THRESHOLD) {
        mult_mat(_n, _mat1, _mat1_jmp, _mat2, _mat2_jmp, _prod, _prod_jmp);
        return;
    }
    
    long half_n = _n / 2;
    size_t half_size = _n * _n * sizeof(double);

    double* mat1_00 = _mat1;
    double* mat1_01 = _mat1 + half_n;
    double* mat1_10 = _mat1 + half_n*_mat1_jmp;
    double* mat1_11 = _mat1 + half_n*_mat1_jmp + half_n;

    double* mat2_00 = _mat2;
    double* mat2_01 = _mat2 + half_n;
    double* mat2_10 = _mat2 + half_n*_mat2_jmp;
    double* mat2_11 = _mat2 + half_n*_mat2_jmp + half_n;

    double* prod_00 = _prod;
    double* prod_01 = _prod + half_n;
    double* prod_10 = _prod + half_n*_prod_jmp;
    double* prod_11 = _prod + half_n*_prod_jmp + half_n;

    double* s0 = NULL;
    double* s1 = NULL;
    double* s2 = NULL;
    double* s3 = NULL;
    double* s4 = NULL;
    double* s5 = NULL;
    double* s6 = NULL;
    double* s7 = NULL;
    double* s8 = NULL;
    double* s9 = NULL;

    double* p0 = NULL;
    double* p1 = NULL;
    double* p2 = NULL;
    double* p3 = NULL;
    double* p4 = NULL;
    double* p5 = NULL;
    double* p6 = NULL;

    // S0 = B01 - B11
    // P0 = A00 * S0
    memory_allocate(&s0, half_size);
    memory_allocate(&p0, half_size);
    sub_mat(half_n, mat2_01, _mat2_jmp, mat2_11, _mat2_jmp, s0, half_n);    
    strassen(half_n, mat1_00, _mat1_jmp, s0, half_n, p0, half_n);
    free(s0);

    // S1 = A00 + A01
    // P1 = S1 * B11
    memory_allocate(&s1, half_size);
    memory_allocate(&p1, half_size);
    add_mat(half_n, mat1_00, _mat1_jmp, mat1_01, _mat1_jmp, s1, half_n);    
    strassen(half_n, s1, half_n, mat2_11, _mat2_jmp, p1, half_n);
    free(s1);

    // S2 = A10 + A11
    // P2 = S2 * B00
    memory_allocate(&s2, half_size);
    memory_allocate(&p2, half_size);
    add_mat(half_n, mat1_10, _mat1_jmp, mat1_11, _mat1_jmp, s2, half_n);    
    strassen(half_n, s2, half_n, mat2_00, _mat2_jmp, p2, half_n);
    free(s2);

    // S3 = B10 - B00
    // P3 = A11 * S3
    memory_allocate(&s3, half_size);
    memory_allocate(&p3, half_size);
    sub_mat(half_n, mat2_10, _mat2_jmp, mat2_00, _mat2_jmp, s3, half_n);    
    strassen(half_n, mat1_11, _mat1_jmp, s3, half_n, p3, half_n);
    free(s3);

    // S4 = A00 + A11
    // S5 = B00 + B11
    // P4 = S4 * S5
    memory_allocate(&s4, half_size);
    memory_allocate(&s5, half_size);
    memory_allocate(&p4, half_size);
    add_mat(half_n, mat1_00, _mat1_jmp, mat1_11, _mat1_jmp, s4, half_n); 
    add_mat(half_n, mat2_00, _mat2_jmp, mat2_11, _mat2_jmp, s5, half_n);   
    strassen(half_n, s4, half_n, s5, half_n, p4, half_n);
    free(s4);
    free(s5);

    // S6 = A01 - A11
    // S7 = B10 + B11
    // P5 = S6 * S7
    memory_allocate(&s6, half_size);
    memory_allocate(&s7, half_size);
    memory_allocate(&p5, half_size);
    sub_mat(half_n, mat1_01, _mat1_jmp, mat1_11, _mat1_jmp, s6, half_n); 
    add_mat(half_n, mat2_10, _mat2_jmp, mat2_11, _mat2_jmp, s7, half_n);   
    strassen(half_n, s6, half_n, s7, half_n, p5, half_n);
    free(s6);
    free(s7);

    // S8 = A00 - A10
    // S9 = B00 + B01
    // P6 = S8 * S9
    memory_allocate(&s8, half_size);
    memory_allocate(&s9, half_size);
    memory_allocate(&p6, half_size);
    sub_mat(half_n, mat1_00, _mat1_jmp, mat1_10, _mat1_jmp, s8, half_n); 
    add_mat(half_n, mat2_00, _mat2_jmp, mat2_01, _mat2_jmp, s9, half_n);   
    strassen(half_n, s8, half_n, s9, half_n, p6, half_n);
    free(s8);
    free(s9);

    // C00 = P4 + P3 - P1 + P5
    add_mat(half_n, p4, half_n, p3, half_n, prod_00, _prod_jmp);
    sub_mat(half_n, prod_00, _prod_jmp, p1, half_n, prod_00, _prod_jmp);
    add_mat(half_n, prod_00, _prod_jmp, p5, half_n, prod_00, _prod_jmp);

    // C01 = P0 + P1
    add_mat(half_n, p0, half_n, p1, half_n, prod_01, _prod_jmp);

    // C10 = P2 + P3
    add_mat(half_n, p2, half_n, p3, half_n, prod_10, _prod_jmp);

    // C11 = P4 + P0 - P2 - P6
    add_mat(half_n, p4, half_n, p0, half_n, prod_11, _prod_jmp);
    sub_mat(half_n, prod_11, _prod_jmp, p2, half_n, prod_11, _prod_jmp);
    sub_mat(half_n, prod_11, _prod_jmp, p6, half_n, prod_11, _prod_jmp);

    free(p0);
    free(p1);
    free(p2);
    free(p3);
    free(p4);
    free(p5);
    free(p6);
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
    arquivo = fopen("mat_prod_strassen_seq", "wb");
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

    // use Strassen's algorithm for matrix multiplication
    strassen(n, (double*) mat1, n, (double*) mat2, n, (double*) prod, n);

    double t_end = omp_get_wtime();
    
    //Time to csv (t_end - t_start, n, 1, "strassen_seq") to be used in the plot script
    printf("%.15lf,%ld,1,strassen_seq\n", t_end - t_start, n);

    // export the product matrix to a binary file
    //exportar_bin();
    finalize();
    return 0;
}