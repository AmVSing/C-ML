/*
    Test of the efficiency of different matrix multiplication algorithms
*/

#include "../../src/tensors/tensor.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


static inline size_t flatten_index(Tensor* m, size_t i, size_t j) {
    return i * m->strides[0] + j * m->strides[1];
}
static Tensor* naive_matmul(const Tensor* m1, const Tensor* m2, Tensor* out) {
    // performs naive matrix multiplication by setting
    // C[i][j] = sum over k (A[row][k] * B[k][column])

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


int main(void) {
    const size_t a_shape[] = {2, 3};
    const size_t b_shape[] = {3, 2};

    const float a_data[] = {1.0f, 2.0f, 3.0f, 
                            4.0f, 5.0f, 6.0f};
    
    const float b_data[] = {7.0f, 8.0f,
                            9.0f, 10.0f,
                            11.0f, 12.0f};
    
    Tensor a = tensor_from_data(MATRIX_RANK, a_shape, a_data);
    Tensor b = tensor_from_data(MATRIX_RANK, b_shape, b_data);

    const size_t result_shape[] = {a_shape[0], b_shape[1]};
    Tensor result = make_tensor(MATRIX_RANK, result_shape);
    matmul(&a, &b, &result);

    free_tensor(&result);
    free_tensor(&a);
    free_tensor(&b);

    return 0;
}

