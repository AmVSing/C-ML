/*
    header file for tensors
*/

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>

#define MAX_DIMS 8
#define MAX_ELEMS ((size_t)1 << 28)
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
Operations either respect strides or return TENSOR_LAYOUT_E when they require contiguous data.
*/

typedef struct tensor_s {
    size_t rank;              // no. of dimensions (0 - scalar, 1 - vector, 2 - matrix ..)
    size_t shape[MAX_DIMS];   // shape[n] = size of dimension n
    size_t strides[MAX_DIMS]; // stores how many positions we move in memory if index increases by 1
    size_t no_elems;          // total no. elems
    float* data;    // malloc-ed block of memory (ALWAYS MALLOC-ED, might have been malloc-ed by
                    // another tensor)
    bool owns_data; // if not, dont need to call free
} Tensor;

typedef enum tensor_status_e_en {
    TENSOR_OK,
    TENSOR_NULL_E, // null pointer error
    TENSOR_RANK_E, // invalid rank error
    TENSOR_SHAPE_E, // invalid shape error
    TENSOR_SIZE_OVERFLOW_E, // tensor size overflows
    TENSOR_ALLOC_E, // tensor allocation fails
    TENSOR_SHAPE_MISMATCH_E, // tensor shapes misalign
    TENSOR_ALIAS_E, // two tensors share storage, but the op requires not-shared
                    // e.g. matmul(&left, &right, &left), or transpose(&m, &m)
    TENSOR_LAYOUT_E, // function doesn't support strided pattern
    TENSOR_INVALID_RANGE_E, // invalid range, e.g. tensor_rand(&t, 100.0f, 0.0f)
    TENSOR_DEVICE_E // GPU specific failures
} TensorStatus;



/* creation + destruction */

// create

// `out` must point to an empty Tensor and is unchanged if creation fails
TensorStatus tensor_try_make(Tensor* out, size_t rank, const size_t shape[]);
TensorStatus tensor_try_from_data(Tensor* out, size_t rank, const size_t shape[],
                                  const float* data);
TensorStatus tensor_try_copy(Tensor* out, const Tensor* t);

Tensor make_tensor(size_t rank, const size_t shape[]); // creates 0 tensor
// Tensor itself is not malloc-d but the tensor's data is
Tensor tensor_from_data(size_t rank, const size_t shape[],
                        const float* data); // create tensor from data
Tensor tensor_copy(const Tensor* t);        // create copy of existing tensor
// input array can be stack or heap allocated
void free_tensor(Tensor* t);

// other helpful funcs

TensorStatus tensor_try_fill(Tensor* t, float value);
TensorStatus tensor_try_rand(Tensor* t, float min, float max);

Tensor* tensor_fill(Tensor* t, float value); // Sets all tensor elements to value
Tensor* tensor_rand(Tensor* t, float min,
                    float max); // creates tensor populated with randfloats from [min, max]

/* ops */

// shape comparison funcs

bool same_shape(const Tensor* t1, const Tensor* t2); // true if t1->shape == t2->shape
bool is_matrix(const Tensor* t);                     // true iff t points to a matrix
bool tensor_is_contiguous(const Tensor* t);

// actual ops - (tensor, scalar)

TensorStatus tensor_try_add_scalar(const Tensor* t, float x, Tensor* out);
TensorStatus tensor_try_mult_scalar(const Tensor* t, float x, Tensor* out);

Tensor* tensor_add_scalar(const Tensor* t, float x, Tensor* out);  // adds x to all entries
Tensor* tensor_mult_scalar(const Tensor* t, float x, Tensor* out); // multiples all entries by x

// actual ops - (tensor, tensor)

TensorStatus tensor_try_add(const Tensor* t1, const Tensor* t2, Tensor* res);
TensorStatus tensor_try_sub(const Tensor* t1, const Tensor* t2, Tensor* res);
TensorStatus tensor_try_mult(const Tensor* t1, const Tensor* t2, Tensor* res);
TensorStatus tensor_try_div(const Tensor* t1, const Tensor* t2, Tensor* res);

Tensor* tensor_add(const Tensor* t1, const Tensor* t2, Tensor* res);
Tensor* tensor_sub(const Tensor* t1, const Tensor* t2, Tensor* res);
Tensor* tensor_mult(const Tensor* t1, const Tensor* t2,
                    Tensor* res); // elementwise multiplication not dot product
Tensor* tensor_div(const Tensor* t1, const Tensor* t2, Tensor* res);

// matrix ops

bool matmultiplicable(const Tensor* m1, const Tensor* m2);
TensorStatus tensor_try_matmul(const Tensor* m1, const Tensor* m2, Tensor* out);
Tensor* matmul(const Tensor* m1, const Tensor* m2, Tensor* out);

TensorStatus tensor_try_transpose_view(const Tensor* m, Tensor* out);
// `out` must point to an empty Tensor; the view borrows data from m and must not outlive m
Tensor transpose_view(const Tensor* m); // works in O(1) time to create a transposed view
// INVALID WHEN ORIGINAL TENSOR IS FREE'D

TensorStatus tensor_try_transpose(const Tensor* m, Tensor* out);
Tensor* transpose(const Tensor* m, Tensor* out); // creates full transpose of matrix in O(n)
// this version is more costly to make, but works better with sequential access due to spatial
// locality

TensorStatus tensor_try_transpose_inplace(Tensor* m);
Tensor* transpose_inplace(Tensor* m); // transposes the matrix in place FOR SQUARE ONLY
// O(n/2) instead of O(n)

/* display helpers */
const char* tensor_status_string(TensorStatus status);
// returns a string corresponding to errors if any occured

#endif
