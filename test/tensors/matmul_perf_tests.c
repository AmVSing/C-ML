/*
    Test of the efficiency of different matrix multiplication algorithms.

    Example CPU output:
    256 x 256 * 256 x 256, 100 reps
    naive: 239.106 ms, optimised: 101.603 ms
    speedup: 2.35x
*/

#include "../../src/tensors/tensor.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>

#define MIN_ENTRY (-100.0f)
#define MAX_ENTRY (100.0f)

typedef Tensor* (*MatmulFunction)(const Tensor* m1, const Tensor* m2, Tensor* out);

static volatile float benchmark_sink;

static double benchmark_matmul(MatmulFunction function, const Tensor* m1, const Tensor* m2,
                               Tensor* out, size_t repetitions) {

    assert(function != NULL);
    assert(repetitions > 0);

    function(m1, m2, out);

    const clock_t start = clock();

    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
        function(m1, m2, out);
    }

    const clock_t end = clock();
    benchmark_sink += out->data[out->no_elems / 2];

    const double elapsed_seconds = (double)(end - start) / (double)CLOCKS_PER_SEC;
    return elapsed_seconds * 1000.0 / (double)repetitions;
}

static inline size_t matrix_index(const Tensor* matrix, size_t row, size_t column) {
    return row * matrix->strides[0] + column * matrix->strides[1];
}

static Tensor* naive_matmul(const Tensor* m1, const Tensor* m2, Tensor* out) {
    assert(matmultiplicable(m1, m2));
    assert(is_matrix(out));
    assert(out != m1 && out != m2);
    assert(out->shape[0] == m1->shape[0]);
    assert(out->shape[1] == m2->shape[1]);

    const size_t rows = m1->shape[0];
    const size_t columns = m2->shape[1];
    const size_t shared = m1->shape[1];

    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            float sum = 0.0f;

            for (size_t k = 0; k < shared; ++k) {
                sum += m1->data[matrix_index(m1, row, k)] * m2->data[matrix_index(m2, k, column)];
            }

            out->data[matrix_index(out, row, column)] = sum;
        }
    }

    return out;
}

static void benchmark_contiguous(size_t rows, size_t shared, size_t columns, size_t repetitions) {
    const size_t m1_shape[] = {rows, shared};
    const size_t m2_shape[] = {shared, columns};
    const size_t out_shape[] = {rows, columns};
    Tensor m1 = make_tensor(MATRIX_RANK, m1_shape);
    Tensor m2 = make_tensor(MATRIX_RANK, m2_shape);
    Tensor out = make_tensor(MATRIX_RANK, out_shape);

    tensor_rand(&m1, MIN_ENTRY, MAX_ENTRY);
    tensor_rand(&m2, MIN_ENTRY, MAX_ENTRY);

    const double naive_average = benchmark_matmul(naive_matmul, &m1, &m2, &out, repetitions);
    const double optimised_average = benchmark_matmul(matmul, &m1, &m2, &out, repetitions);

    printf("%zu x %zu * %zu x %zu, %zu reps\n"
           "naive: %.3f ms, optimised: %.3f ms\n"
           "speedup: %.2fx\n",
           rows, shared, shared, columns, repetitions, naive_average, optimised_average,
           naive_average / optimised_average);

    free_tensor(&m1);
    free_tensor(&m2);
    free_tensor(&out);
}

int main(void) {
    benchmark_contiguous(256, 256, 256, 100);
    return 0;
}
