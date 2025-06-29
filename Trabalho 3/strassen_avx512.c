#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <string.h>

// AVX-512 vector width (8 doubles)
#define AVX512_VEC_SIZE 8
#define ALIGNMENT 64

// Cache-aware block size optimized for Xeon Phi
#define BLOCK_SIZE 128
#define STRASSEN_THRESHOLD 1024

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
void transpose_mat_avx512(long _n, double* _mat, long _mat_jmp, double* _mat_t, long _mat_t_jmp);
void add_mat_avx512(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _adic, long _adic_jmp);
void sub_mat_avx512(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _subt, long _subt_jmp);
void mult_mat_avx512(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _prod, long _prod_jmp);

// AVX-512 optimized transpose
void transpose_mat_avx512(long _n, double* _mat, long _mat_jmp, double* _mat_t, long _mat_t_jmp) {
    #pragma omp taskloop grainsize(32) default(none) firstprivate(_n, _mat, _mat_jmp, _mat_t, _mat_t_jmp)
    for (long i = 0; i < _n; i += 8) {
        for (long j = 0; j < _n; j += 8) {
            // 8x8 block transpose using AVX-512
            long max_i = (i + 8 > _n) ? _n : i + 8;
            long max_j = (j + 8 > _n) ? _n : j + 8;
            
            for (long ii = i; ii < max_i; ++ii) {
                double* src = _mat + ii * _mat_jmp + j;
                double* dst = _mat_t + j * _mat_t_jmp + ii;
                
                long vec_len = max_j - j;
                long vec_ops = vec_len / AVX512_VEC_SIZE;
                long remainder = vec_len % AVX512_VEC_SIZE;
                
                // Vectorized part
                for (long v = 0; v < vec_ops; ++v) {
                    __m512d vec = _mm512_loadu_pd(src + v * AVX512_VEC_SIZE);
                    
                    // Store transposed elements
                    for (int k = 0; k < AVX512_VEC_SIZE; ++k) {
                        double val = ((double*)&vec)[k];
                        *(dst + k * _mat_t_jmp) = val;
                    }
                }
                
                // Handle remainder
                for (long r = vec_ops * AVX512_VEC_SIZE; r < vec_len; ++r) {
                    *(dst + r * _mat_t_jmp) = src[r];
                }
            }
        }
    }
}

// AVX-512 optimized addition
void add_mat_avx512(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _adic, long _adic_jmp) {
    #pragma omp taskloop grainsize(32) default(none) firstprivate(_n, _mat1, _mat1_jmp, _mat2, _mat2_jmp, _adic, _adic_jmp)
    for (long i = 0; i < _n; ++i) {
        double* p_mat1 = _mat1 + i * _mat1_jmp;
        double* p_mat2 = _mat2 + i * _mat2_jmp;
        double* p_adic = _adic + i * _adic_jmp;
        
        long vec_ops = _n / AVX512_VEC_SIZE;
        long remainder = _n % AVX512_VEC_SIZE;
        
        // Vectorized addition
        for (long j = 0; j < vec_ops; ++j) {
            __m512d vec1 = _mm512_loadu_pd(p_mat1 + j * AVX512_VEC_SIZE);
            __m512d vec2 = _mm512_loadu_pd(p_mat2 + j * AVX512_VEC_SIZE);
            __m512d result = _mm512_add_pd(vec1, vec2);
            _mm512_storeu_pd(p_adic + j * AVX512_VEC_SIZE, result);
        }
        
        // Handle remainder
        for (long j = vec_ops * AVX512_VEC_SIZE; j < _n; ++j) {
            p_adic[j] = p_mat1[j] + p_mat2[j];
        }
    }
}

// AVX-512 optimized subtraction
void sub_mat_avx512(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _subt, long _subt_jmp) {
    #pragma omp taskloop grainsize(32) default(none) firstprivate(_n, _mat1, _mat1_jmp, _mat2, _mat2_jmp, _subt, _subt_jmp)
    for (long i = 0; i < _n; ++i) {
        double* p_mat1 = _mat1 + i * _mat1_jmp;
        double* p_mat2 = _mat2 + i * _mat2_jmp;
        double* p_subt = _subt + i * _subt_jmp;
        
        long vec_ops = _n / AVX512_VEC_SIZE;
        long remainder = _n % AVX512_VEC_SIZE;
        
        // Vectorized subtraction
        for (long j = 0; j < vec_ops; ++j) {
            __m512d vec1 = _mm512_loadu_pd(p_mat1 + j * AVX512_VEC_SIZE);
            __m512d vec2 = _mm512_loadu_pd(p_mat2 + j * AVX512_VEC_SIZE);
            __m512d result = _mm512_sub_pd(vec1, vec2);
            _mm512_storeu_pd(p_subt + j * AVX512_VEC_SIZE, result);
        }
        
        // Handle remainder
        for (long j = vec_ops * AVX512_VEC_SIZE; j < _n; ++j) {
            p_subt[j] = p_mat1[j] - p_mat2[j];
        }
    }
}

// AVX-512 optimized matrix multiplication with blocking
void mult_mat_avx512(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _prod, long _prod_jmp) {
    // Initialize product matrix
    #pragma omp taskloop grainsize(16) default(none) firstprivate(_n, _prod, _prod_jmp)
    for (long i = 0; i < _n; ++i) {
        double* row = _prod + i * _prod_jmp;
        long vec_ops = _n / AVX512_VEC_SIZE;
        
        for (long j = 0; j < vec_ops; ++j) {
            _mm512_storeu_pd(row + j * AVX512_VEC_SIZE, _mm512_setzero_pd());
        }
        
        for (long j = vec_ops * AVX512_VEC_SIZE; j < _n; ++j) {
            row[j] = 0.0;
        }
    }
    
    // Blocked matrix multiplication with AVX-512
    #pragma omp taskloop grainsize(2) default(none) firstprivate(_n, _mat1, _mat1_jmp, _mat2, _mat2_jmp, _prod, _prod_jmp)
    for (long ii = 0; ii < _n; ii += BLOCK_SIZE) {
        for (long jj = 0; jj < _n; jj += BLOCK_SIZE) {
            for (long kk = 0; kk < _n; kk += BLOCK_SIZE) {
                
                long i_end = (ii + BLOCK_SIZE > _n) ? _n : ii + BLOCK_SIZE;
                long j_end = (jj + BLOCK_SIZE > _n) ? _n : jj + BLOCK_SIZE;
                long k_end = (kk + BLOCK_SIZE > _n) ? _n : kk + BLOCK_SIZE;
                
                // Inner loops for the block
                for (long i = ii; i < i_end; ++i) {
                    for (long k = kk; k < k_end; ++k) {
                        __m512d a_vec = _mm512_set1_pd(_mat1[i * _mat1_jmp + k]);
                        
                        double* b_row = _mat2 + k * _mat2_jmp + jj;
                        double* c_row = _prod + i * _prod_jmp + jj;
                        
                        long vec_ops = (j_end - jj) / AVX512_VEC_SIZE;
                        
                        // Vectorized inner loop
                        for (long j = 0; j < vec_ops; ++j) {
                            __m512d b_vec = _mm512_loadu_pd(b_row + j * AVX512_VEC_SIZE);
                            __m512d c_vec = _mm512_loadu_pd(c_row + j * AVX512_VEC_SIZE);
                            __m512d result = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
                            _mm512_storeu_pd(c_row + j * AVX512_VEC_SIZE, result);
                        }
                        
                        // Handle remainder
                        for (long j = jj + vec_ops * AVX512_VEC_SIZE; j < j_end; ++j) {
                            _prod[i * _prod_jmp + j] += _mat1[i * _mat1_jmp + k] * _mat2[k * _mat2_jmp + j];
                        }
                    }
                }
            }
        }
    }
}

void strassen(long _n, double* _mat1, long _mat1_jmp, double* _mat2, long _mat2_jmp, double* _prod, long _prod_jmp) {
    
    if (_n <= STRASSEN_THRESHOLD) {
        mult_mat_avx512(_n, _mat1, _mat1_jmp, _mat2, _mat2_jmp, _prod, _prod_jmp);
        return;
    }
    
    long half_n = _n / 2;
    size_t half_size = half_n * half_n * sizeof(double);

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

    double* tmp0 = NULL;
    double* tmp1 = NULL;

    #pragma omp taskgroup
    {
        // S0 = B01 - B11
        #pragma omp task depend(out: s0) default(none) firstprivate(half_size, half_n, mat2_01, mat2_11, _mat2_jmp) shared(s0)
        {
            memory_allocate(&s0, half_size);
            sub_mat_avx512(half_n, mat2_01, _mat2_jmp, mat2_11, _mat2_jmp, s0, half_n);
        }
        
        // S1 = A00 + A01
        #pragma omp task depend(out: s1) default(none) firstprivate(half_size, half_n, mat1_00, mat1_01, _mat1_jmp) shared(s1)
        {
            memory_allocate(&s1, half_size);
            add_mat_avx512(half_n, mat1_00, _mat1_jmp, mat1_01, _mat1_jmp, s1, half_n);
        }
        
        // S2 = A10 + A11
        #pragma omp task depend(out: s2) default(none) firstprivate(half_size, half_n, mat1_10, mat1_11, _mat1_jmp) shared(s2)
        {
            memory_allocate(&s2, half_size);
            add_mat_avx512(half_n, mat1_10, _mat1_jmp, mat1_11, _mat1_jmp, s2, half_n);
        }
        
        // S3 = B10 - B00
        #pragma omp task depend(out: s3) default(none) firstprivate(half_size, half_n, mat2_10, mat2_00, _mat2_jmp) shared(s3)
        {
            memory_allocate(&s3, half_size);
            sub_mat_avx512(half_n, mat2_10, _mat2_jmp, mat2_00, _mat2_jmp, s3, half_n);
        }
        
        // S4 = A00 + A11
        #pragma omp task depend(out: s4) default(none) firstprivate(half_size, half_n, mat1_00, mat1_11, _mat1_jmp) shared(s4)
        {
            memory_allocate(&s4, half_size);
            add_mat_avx512(half_n, mat1_00, _mat1_jmp, mat1_11, _mat1_jmp, s4, half_n);
        }
        
        // S5 = B00 + B11
        #pragma omp task depend(out: s5) default(none) firstprivate(half_size, half_n, mat2_00, mat2_11, _mat2_jmp) shared(s5)
        {
            memory_allocate(&s5, half_size);
            add_mat_avx512(half_n, mat2_00, _mat2_jmp, mat2_11, _mat2_jmp, s5, half_n);
        }
        
        // S6 = A01 - A11
        #pragma omp task depend(out: s6) default(none) firstprivate(half_size, half_n, mat1_01, mat1_11, _mat1_jmp) shared(s6)
        {
            memory_allocate(&s6, half_size);
            sub_mat_avx512(half_n, mat1_01, _mat1_jmp, mat1_11, _mat1_jmp, s6, half_n);
        }
        
        // S7 = B10 + B11
        #pragma omp task depend(out: s7) default(none) firstprivate(half_size, half_n, mat2_10, mat2_11, _mat2_jmp) shared(s7)
        {
            memory_allocate(&s7, half_size);
            add_mat_avx512(half_n, mat2_10, _mat2_jmp, mat2_11, _mat2_jmp, s7, half_n);
        }
        
        // S8 = A00 - A10
        #pragma omp task depend(out: s8) default(none) firstprivate(half_size, half_n, mat1_00, mat1_10, _mat1_jmp) shared(s8)
        {
            memory_allocate(&s8, half_size);
            sub_mat_avx512(half_n, mat1_00, _mat1_jmp, mat1_10, _mat1_jmp, s8, half_n);
        }
        
        // S9 = B00 + B01
        #pragma omp task depend(out: s9) default(none) firstprivate(half_size, half_n, mat2_00, mat2_01, _mat2_jmp) shared(s9)
        {
            memory_allocate(&s9, half_size);
            add_mat_avx512(half_n, mat2_00, _mat2_jmp, mat2_01, _mat2_jmp, s9, half_n);
        }        
        
        // P0 = A00 * S0
        #pragma omp task depend(in: s0) depend(out: p0) default(none) firstprivate(half_size, half_n, mat1_00, _mat1_jmp) shared(s0, p0)
        {
            memory_allocate(&p0, half_size);
            strassen(half_n, mat1_00, _mat1_jmp, s0, half_n, p0, half_n);
            free(s0);
        }        

        // P1 = S1 * B11
        #pragma omp task depend(in: s1) depend(out: p1) default(none) firstprivate(half_size, half_n, mat2_11, _mat2_jmp) shared(s1, p1)
        {
            memory_allocate(&p1, half_size);
            strassen(half_n, s1, half_n, mat2_11, _mat2_jmp, p1, half_n);
            free(s1);
        }
        
        // P2 = S2 * B00
        #pragma omp task depend(in: s2) depend(out: p2) default(none) firstprivate(half_size, half_n, mat2_00, _mat2_jmp) shared(s2, p2)
        {
            memory_allocate(&p2, half_size);
            strassen(half_n, s2, half_n, mat2_00, _mat2_jmp, p2, half_n);
            free(s2);
        }

        // P3 = A11 * S3
        #pragma omp task depend(in: s3) depend(out: p3) default(none) firstprivate(half_size, half_n, mat1_11, _mat1_jmp) shared(s3, p3)
        {
            memory_allocate(&p3, half_size);                
            strassen(half_n, mat1_11, _mat1_jmp, s3, half_n, p3, half_n);
            free(s3);
        }        

        // P4 = S4 * S5
        #pragma omp task depend(in: s4, s5) depend(out: p4) default(none) firstprivate(half_size, half_n) shared(s4, s5, p4)
        {
            memory_allocate(&p4, half_size);
            strassen(half_n, s4, half_n, s5, half_n, p4, half_n);
            free(s4);
            free(s5);
        }

        // P5 = S6 * S7
        #pragma omp task depend(in: s6, s7) depend(out: p5) default(none) firstprivate(half_size, half_n) shared(s6, s7, p5)
        {
            memory_allocate(&p5, half_size);
            strassen(half_n, s6, half_n, s7, half_n, p5, half_n);
            free(s6);
            free(s7);
        }        

        // P6 = S8 * S9
        #pragma omp task depend(in: s8, s9) depend(out: p6) default(none) firstprivate(half_size, half_n) shared(s8, s9, p6)
        {        
            memory_allocate(&p6, half_size);               
            strassen(half_n, s8, half_n, s9, half_n, p6, half_n);
            free(s8);
            free(s9);
        }        

        // C00 = P4 + P3 - P1 + P5 = (P4 + P5) + (P3 - P1) 
        #pragma omp task depend(in: p4, p5) depend(out: prod_00) default(none) firstprivate(half_n, _prod_jmp) shared(p4, p5, prod_00)
        {
            add_mat_avx512(half_n, p4, half_n, p5, half_n, prod_00, _prod_jmp);
        }
        #pragma omp task depend(in: p3, p1) depend(out: tmp0) default(none) firstprivate(half_n, half_size) shared(p3, p1, tmp0)
        {
            memory_allocate(&tmp0, half_size);
            sub_mat_avx512(half_n, p3, half_n, p1, half_n, tmp0, half_n);            
        }
        #pragma omp task depend(in: prod_00, tmp0) default(none) firstprivate(half_n, _prod_jmp) shared(prod_00, tmp0)
        {            
            add_mat_avx512(half_n, prod_00, _prod_jmp, tmp0, half_n, prod_00, _prod_jmp);
            free(tmp0);
        }        

        // C01 = P0 + P1
        #pragma omp task depend(in: p0, p1) default(none) firstprivate(half_n, prod_01, _prod_jmp) shared(p0, p1)
        {
            add_mat_avx512(half_n, p0, half_n, p1, half_n, prod_01, _prod_jmp);
        }        

        // C10 = P2 + P3
        #pragma omp task depend(in: p2, p3) default(none) firstprivate(half_n, prod_10, _prod_jmp) shared(p2, p3)
        {
            add_mat_avx512(half_n, p2, half_n, p3, half_n, prod_10, _prod_jmp);
        }        

        // C11 = P4 + P0 - P2 - P6 = (P4 - P2) + (P0 - P6)
        #pragma omp task depend(in: p4, p2) depend(out: prod_11) default(none) firstprivate(half_n, _prod_jmp) shared(p4, p2, prod_11)
        {
            sub_mat_avx512(half_n, p4, half_n, p2, half_n, prod_11, _prod_jmp);
        }
        #pragma omp task depend(in: p0, p6) depend(out: tmp1) default(none) firstprivate(half_n, half_size) shared(p0, p6, tmp1)
        {
            memory_allocate(&tmp1, half_size);
            sub_mat_avx512(half_n, p0, half_n, p6, half_n, tmp1, half_n);
        }
        #pragma omp task depend(in: prod_11, tmp1) default(none) firstprivate(half_n, _prod_jmp) shared(prod_11, tmp1)
        {
            add_mat_avx512(half_n, prod_11, _prod_jmp, tmp1, half_n, prod_11, _prod_jmp);
            free(tmp1);
        }
    } // implicit barrier at the end of taskgroup

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
    if (posix_memalign((void**) mat, ALIGNMENT, size)) {
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
    memset(prod, 0, size);
}

void exportar_bin() {
    FILE* arquivo;
    arquivo = fopen("mat_prod_strassen_avx512", "wb");
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

    int num_threads = -1;
    double t_start = omp_get_wtime();

    // use Strassen's algorithm for matrix multiplication
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            strassen(n, (double*) mat1, n, (double*) mat2, n, (double*) prod, n);
        }    
    }

    double t_end = omp_get_wtime();
    printf("%.15lf,%ld,%d,strassen_avx512\n", t_end - t_start, n, num_threads);

    // export the product matrix to a binary file
    //exportar_bin();
    finalize();
    return 0;
}