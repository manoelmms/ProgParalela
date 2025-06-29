#!/bin/bash 

# Compile the naive matrix multiplication program using gcc
gcc -fopenmp -o naive_mul naive_mul.c -lm
# Compile the optimized matrix multiplication program using gcc
gcc -DL1D_CACHE_TAM=$(getconf LEVEL1_DCACHE_LINESIZE) -O3 -fno-tree-loop-vectorize -fno-tree-slp-vectorize -fno-tree-vectorize -fopenmp -o optimized_mul optimized_mul.c -lm
# Compile the Strassen algorithm program using gcc
gcc -DL1D_CACHE_TAM=$(getconf LEVEL1_DCACHE_LINESIZE) -O3 -fno-tree-loop-vectorize -fno-tree-slp-vectorize -fno-tree-vectorize -fopenmp -o strassen_omp_tsk strassen_omp_tsk.c -lm
# Compile the Strassen algorithm program using gcc vectorized MIC-AVX512
gcc -DL1D_CACHE_TAM=$(getconf LEVEL1_DCACHE_LINESIZE) -O3 -mavx512f -mavx512er -mavx512cd -mavx512pf -fopenmp -o strassen_omp_tsk_mic strassen_omp_tsk.c -lm