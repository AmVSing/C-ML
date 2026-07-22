/*
    header file for tensors
*/

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>

#define MAX_DIMS 8
#define MAX_ELEMS (1 << 28)
#define MATRIX_RANK 2
/* 
the stride for each tensor is a stack allocated array of integers, where
moving one position in the ith dimension corresponds to moving 
strides[i] positions in the flat data array

e.g.
[
    [1, 2, 3]
    [4, 5, 6]
]
data = [1,2,3,4,5,6]
shape = [2, 3]
strides = [3, 1] (moving down a row increase position by 3, moving across 
a column increases position by 1)
NOTE - OTHER FUNCTIONS DO NOT ALWAYS RESPECT STRIDES
*/


typedef struct tensor_s {
    size_t rank; // no. of dimensions (0 - scalar, 1 - vector, 2 - matrix ..)
    size_t shape[MAX_DIMS]; // shape[n] = size of dimension n
    size_t strides[MAX_DIMS]; // stores how many positions we move in memory if index increases by 1
    size_t no_elems; // total no. elems
    float* data; // malloc-ed block of memory (ALWAYS MALLOC-ED)
    bool owns_data; // if not, dont need to call free
} Tensor;

/* creation + destruction */

// create

Tensor make_tensor(size_t rank, const size_t shape[]); // creates 0 tensor
// Tensor itself is not malloc-d but the tensor's data is
Tensor tensor_from_data(size_t rank, const size_t shape[], const float* data); // create tensor from data
Tensor tensor_copy(const Tensor* t); // create copy of existing tensor
// input array can be stack or heap allocated
void free_tensor(Tensor* t); 

// other helpful funcs

Tensor* tensor_fill(Tensor* t, float value); // Sets all tensor elements to value
Tensor* tensor_rand(Tensor* t, float min, float max); // creates tensor populated with randfloats from [min, max]

/* ops */

// shape comparison funcs

bool same_shape(const Tensor* t1, const Tensor* t2); // true if t1->shape == t2->shape
bool is_matrix(const Tensor* t); // true iff t points to a matrix

// actual ops - (tensor, scalar)

Tensor* tensor_add_scalar(const Tensor* t, float x, Tensor* out); // adds x to all entries
Tensor* tensor_mult_scalar(const Tensor* t, float x, Tensor* out); // multiples all entries by x

// actual ops - (tensor, tensor)

Tensor* tensor_add(const Tensor* t1, const Tensor* t2, Tensor* res);
Tensor* tensor_sub(const Tensor* t1, const Tensor* t2, Tensor* res);
Tensor* tensor_mult(const Tensor* t1, const Tensor* t2, Tensor* res); // elementwise multiplication not dot product
Tensor* tensor_div(const Tensor* t1, const Tensor* t2, Tensor* res);

// matrix ops

bool matmultiplicable(const Tensor* m1, const Tensor* m2);
Tensor* matmul(const Tensor* m1, const Tensor* m2, Tensor* out); 

Tensor transpose_view(const Tensor* m); // works in O(1) time to create a transposed view
// INVALID WHEN ORIGINAL TENSOR IS FREE'D
// NOTE - OTHER FUNCTIONS DO NOT ALWAYS RESPECT STRIDES, PURELY FOR TESTING

Tensor* transpose(const Tensor* m, Tensor* out); // creates full transpose of matrix in O(n)
// this version is more costly to make, but works better with sequential access due to spatial locality

Tensor* transpose_inplace(Tensor* m); // transposes the matrix in place FOR SQUARE ONLY
// O(n/2) instead of O(n)

/* display helpers */

#endif