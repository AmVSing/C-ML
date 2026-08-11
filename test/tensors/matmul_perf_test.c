/*
    Test of the efficiency of different matrix multiplication algorithms
*/

/*
    Example output on CPU:
    256 x 256 * 256 x 256, 100 reps 
    naive: 239.106 ms, optimised: 101.603 ms
    speedup: 2.35x
*/

#include "../../src/tensors/tensor.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MIN_ENTRY (-100.0f)
#define MAX_ENTRY (100.0f)

typedef Tensor* (*matmul_func)(const Tensor* m1, const Tensor* m2, Tensor* out);
// function pointer to a matmul function

static volatile float bench_sink; // acts as a sink, so matmul results appear used though they aren't
// volatile so that compiler doesn't just delete accesses because it's not used anywhere else

static double bench_matmul(
    matmul_func matmul,
    const Tensor* m1, 
    const Tensor* m2,
    Tensor* out,
    int reps
) {
    // benchmarks matrix multiplication by performing the matrix multiplication
    // `reps` times, and returns average time taken to multiply

    (*matmul)(m1, m2, out); // called to warm up cache and allocated

    const clock_t start = clock();

    for (int i = 0; i < reps; i++) {
        (*matmul)(m1, m2, out);
    }

    const clock_t end = clock();

    bench_sink += out->data[1]; // prevent compiler from optimising the matmuls away

    const double time = (double) (end-start) / (double) CLOCKS_PER_SEC;
    // calculates number of clock ticks then divides by clocks per sec
    
    return time * 1000.0 / (double) reps;
    
}

static inline size_t flatten_index(const Tensor* m, size_t i, size_t j) {
    return i * m->strides[0] + j * m->strides[1];
}
static Tensor* naive_matmul(const Tensor* m1, const Tensor* m2, Tensor* out) {
    // performs naive matrix multiplication by setting
    // C[i][j] = sum over k (A[row][k] * B[k][column])

    assert( matmultiplicable(m1, m2) );
    assert(is_matrix(out));
    assert(out != m1 && out != m2);
    assert(out->shape[0] == m1->shape[0]);
    assert(out->shape[1] == m2->shape[1]);

    const size_t rows = m1->shape[0];
    const size_t cols = m2->shape[1];
    const size_t shared = m1->shape[1]; // or m2->shape[0];

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < shared; k++) {
                sum += m1->data[flatten_index(m1, i, k)] * m2->data[flatten_index(m2, k, j)];
            }
            out->data[flatten_index(out, i, j)] = sum;
        }
    }
    return out;
}

static void bench_contiguous(
    size_t rows,
    size_t shared,
    size_t cols,
    int reps
) {
   const size_t m1_shape[] = {rows, shared};
   const size_t m2_shape[] = {shared, cols};
   const size_t out_shape[] = {rows, cols};
   
   Tensor m1 = make_tensor(MATRIX_RANK, m1_shape);
   Tensor m2 = make_tensor(MATRIX_RANK, m2_shape);
   Tensor out = make_tensor(MATRIX_RANK, out_shape);

   tensor_rand(&m1, MIN_ENTRY, MAX_ENTRY);
   tensor_rand(&m2, MIN_ENTRY, MAX_ENTRY);

   const double naive_avg = bench_matmul(&naive_matmul, &m1, &m2, &out, reps);
   const double optimised_avg = bench_matmul(&matmul, &m1, &m2, &out, reps);

   fprintf(stdout, "%zu x %zu * %zu x %zu, %d reps \nnaive: %.3f ms, optimised: %.3f ms\nspeedup: %.2fx\n", 
    rows, shared, shared, cols, reps, naive_avg, optimised_avg, naive_avg/optimised_avg);

   free_tensor(&m1);
   free_tensor(&m2);
   free_tensor(&out);
}

int main(void) {

    bench_contiguous(256,256,256,100);
}

