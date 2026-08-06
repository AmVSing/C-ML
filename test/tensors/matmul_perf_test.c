/*
    Test of the efficiency of different matrix multiplication algorithms
*/

#include "../../src/tensors/tensor.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
}

