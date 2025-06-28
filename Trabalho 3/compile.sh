#!/bin/bash 

# Compile the naive matrix multiplication program using intel compiler
icc -qopenmp -o naive_mul naive_mul.c -lm
# Compile the optimized matrix multiplication program using intel compiler
icc -O3 -qopenmp -o optimized_mul optimized_mul.c -lm
# Compile the Strassen algorithm program using intel compiler
icc -O3 -qopenmp -o strassen_omp_tsk strassen_omp_tsk.c -lm
# Compile the Strassen algorithm program using intel compiler vectorized MIC-AVX512
icc -O3 -qopenmp -xMIC-AVX512 -o strassen_omp_tsk_mic strassen_omp_tsk.c -lm