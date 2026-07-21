/*
    header file for tensors
*/

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>

#define MAX_DIMS 8

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
Tensor tensor_from_data(size_t rank, const size_t shape[], float* data); // create tensor from data
void free_tensor(Tensor* t); 

// other helpful funcs
Tensor* tensor_fill(Tensor* t, float value); // Sets all tensor elements to value

Tensor* tensor_rand(Tensor* t, float min, float max); // creates tensor populated with randfloats from [min, max]

/* ops */

// shape comparison funcs
bool tensor_addable(const Tensor* t1, const Tensor* t2); // true iff t1 and t2 have the same shape
bool tensor_multiplicable(const Tensor* t1, const Tensor* t2); // true iff t1 and t2 are dot product compatible
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
Tensor* matmul(const Tensor* a, const Tensor* b, Tensor* out); 
Tensor* transpose(const Tensor* a, Tensor* out);

/* display helpers */

#endif