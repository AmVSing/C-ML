/*
    implementation of tensors
*/

#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/*
struct from tensor.h

typedef struct tensor_s {
    size_t rank; // no. of dimensions (0 - scalar, 1 - vector, 2 - matrix ..)
    size_t shape[MAX_DIMS]; // shape[n] = size of dimension n
    size_t strides[MAX_DIMS]; // stores how many positions we move in memory if index increases by 1
    size_t no_elems; // total no. elems
    float* data; // malloc-ed block of memory
    bool owns_data; // if not, dont need to call free
} Tensor;
*/

/* tensor try initialisation functions, and static helpers*/

static void compute_strides(Tensor* t) {
    assert(t != NULL);

    if (t->rank == 0)
        return; // scalar strides is empty

    t->strides[t->rank - 1] = 1; // bottom up, final stride is always 1

    // from the next dimension stride and length, work out curr stride
    for (size_t i = t->rank - 1; i > 0; --i) {
        t->strides[i - 1] = t->strides[i] * t->shape[i];
    }
}

static void tensor_panic(const char* function, TensorStatus status) {
    // called when an error occurs, prints it to stderr and aborts
    fprintf(stderr, "%s: %s\n", function, tensor_status_string(status));
    abort();
}

static TensorStatus tensor_try_make_uninit(Tensor* out, size_t rank, const size_t shape[]) {
    // tries to make, if failure then returns corresponding status
    // otherwise returns success and creates tensor at out
    if (out == NULL) { // check output not null
        return TENSOR_NULL_E;
    }
    if (rank > MAX_DIMS) {
        return TENSOR_RANK_E;
    }
    if (shape == NULL && rank > 0) {
        return TENSOR_NULL_E;
    }

    Tensor t = {0}; // set all fields to 0

    t.rank = rank;
    t.no_elems = 1;

    for (size_t i = 0; i < rank; i++) { // doesn't run for scalar
        if (shape[i] == 0) {
            return TENSOR_SHAPE_E;
        }

        if (t.no_elems > MAX_ELEMS / shape[i]) {
            return TENSOR_SIZE_OVERFLOW_E;
        }
        t.no_elems *= shape[i];
        t.shape[i] = shape[i];
    }

    // check if nmemb for malloc can be represented as a size_t
    // technically redundant given current max elems and type of t.data
    if (t.no_elems > SIZE_MAX / sizeof(*t.data)) {
        return TENSOR_SIZE_OVERFLOW_E;
    }

    t.data = malloc(t.no_elems * sizeof(*t.data));
    t.owns_data = true;

    if (t.data == NULL) {
        return TENSOR_ALLOC_E;
    }

    compute_strides(&t);

    *out = t;
    return TENSOR_OK;
}

TensorStatus tensor_try_make(Tensor* out, size_t rank, const size_t shape[]) {
    // wrapper around static try make uninit, initialises vals to 0
    if (out == NULL) {
        return TENSOR_NULL_E;
    }

    Tensor t = {0};
    const TensorStatus status = tensor_try_make_uninit(&t, rank, shape);

    if (status != TENSOR_OK) {
        return status;
    }

    memset(t.data, 0, t.no_elems * sizeof(*(t.data))); // initialised as 0s

    *out = t;
    return TENSOR_OK;
}

TensorStatus tensor_try_from_data(Tensor* out, size_t rank, const size_t shape[],
                                  const float* data) {
    // wrapper around try make uninit which initialises data with data input
    if (out == NULL || data == NULL) {
        return TENSOR_NULL_E;
    }

    Tensor t = {0};
    const TensorStatus status = tensor_try_make_uninit(&t, rank, shape);

    if (status != TENSOR_OK) {
        return status;
    }

    memcpy(t.data, data, t.no_elems * sizeof(*data));

    *out = t;
    return TENSOR_OK;
}

static bool tensor_layout_fits_data(const Tensor* t) {
    if (t == NULL || t->data == NULL || t->rank > MAX_DIMS || t->no_elems == 0) {
        return false;
    }

    size_t max_offset = 0;

    for (size_t dimension = 0; dimension < t->rank; dimension++) {
        if (t->shape[dimension] == 0) {
            return false;
        }

        const size_t extent = t->shape[dimension] - 1;
        const size_t stride = t->strides[dimension];

        if (stride != 0 && extent > (SIZE_MAX - max_offset) / stride) {
            return false;
        }

        max_offset += extent * stride;
    }

    return max_offset < t->no_elems;
}

static size_t logical_offset(const Tensor* t, size_t linear_index) {
    size_t offset = 0;

    for (size_t dimension = t->rank; dimension > 0; dimension--) {
        const size_t index = linear_index % t->shape[dimension - 1];
        linear_index /= t->shape[dimension - 1];
        offset += index * t->strides[dimension - 1];
    }

    return offset;
}

TensorStatus tensor_try_copy(Tensor* out, const Tensor* t) {
    // tries copying to out, on error returns corresponding status
    if (out == NULL || t == NULL || t->data == NULL) {
        return TENSOR_NULL_E;
    }
    if (out == t) {
        return TENSOR_ALIAS_E; // args should be distinct
    }

    Tensor copy = {0};
    const TensorStatus status = tensor_try_make_uninit(&copy, t->rank, t->shape);

    if (status != TENSOR_OK) {
        return status;
    }
    if (copy.no_elems != t->no_elems) {
        free(copy.data);
        return TENSOR_SHAPE_E;
    }
    if (!tensor_layout_fits_data(t)) {
        free(copy.data);
        return TENSOR_LAYOUT_E;
    }

    if (tensor_is_contiguous(t)) {
        memcpy(copy.data, t->data, copy.no_elems * sizeof(*copy.data));
    } else {
        for (size_t i = 0; i < copy.no_elems; i++) {
            copy.data[i] = t->data[logical_offset(t, i)];
        }
    }

    *out = copy;
    return TENSOR_OK;
}

/* Initialisation functions (mainly wrappers around try funcs) */

Tensor make_tensor(size_t rank, const size_t shape[]) {
    // wrapper around try make that aborts if error
    // otherwises returns created Tensor

    Tensor t = {0};
    const TensorStatus status = tensor_try_make(&t, rank, shape);

    if (status != TENSOR_OK) {
        tensor_panic("make_tensor", status);
    }

    return t;
}

Tensor tensor_from_data(size_t rank, const size_t shape[], const float* data) {
    // wrapper around tensor try from data that aborts on error
    // otherwise returns tensor

    Tensor t = {0};
    const TensorStatus status = tensor_try_from_data(&t, rank, shape, data);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_from_data", status);
    }

    return t;
}

Tensor tensor_copy(const Tensor* t) {
    // wrapper around tensor try copy that aborts on error otherwise returns tensor
    Tensor copy = {0};
    const TensorStatus status = tensor_try_copy(&copy, t);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_copy", status);
    }

    return copy;
}

/* Other helpful functions */

static float rand_float(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

TensorStatus tensor_try_fill(Tensor* t, float value) {
    // try fill with value, returns error on failure
    if (t == NULL || t->data == NULL) {
        return TENSOR_NULL_E;
    }
    if (!tensor_is_contiguous(t)) {
        return TENSOR_LAYOUT_E;
    }

    for (size_t i = 0; i < t->no_elems; i++) {
        t->data[i] = value;
    }

    return TENSOR_OK;
}

Tensor* tensor_fill(Tensor* t, float value) {
    // wrapper around try fill aborts on error
    const TensorStatus status = tensor_try_fill(t, value);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_fill", status);
    }

    return t;
}

TensorStatus tensor_try_rand(Tensor* t, float min, float max) {
    // randomised tensor
    if (t == NULL || t->data == NULL) {
        return TENSOR_NULL_E;
    }
    if (!isfinite(min) || !isfinite(max) || min > max) {
        return TENSOR_INVALID_RANGE_E;
    }
    if (!tensor_is_contiguous(t)) {
        return TENSOR_LAYOUT_E;
    }

    for (size_t i = 0; i < t->no_elems; i++) {
        t->data[i] = rand_float(min, max);
    }

    return TENSOR_OK;
}

Tensor* tensor_rand(Tensor* t, float min, float max) {
    // wrapper around try aborts on error
    const TensorStatus status = tensor_try_rand(t, min, max);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_rand", status);
    }

    return t;
}

/* free-ing */
void free_tensor(Tensor* t) {
    if (t == NULL) {
        return;
    }

    if (t->owns_data) {
        free(t->data);
    }

    *t = (Tensor){0}; // reset Tensor to 0
}

/* shape comparison funcs */

bool same_shape(const Tensor* t1, const Tensor* t2) {
    if (t1 == NULL || t2 == NULL) {
        return false;
    }

    if (t1->rank != t2->rank) {
        return false;
    }

    for (size_t i = 0; i < t1->rank; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            return false;
        }
    }
    return true;
}

bool is_matrix(const Tensor* t) {
    return t != NULL && t->rank == MATRIX_RANK;
}

bool tensor_is_contiguous(const Tensor* t) {
    if (t == NULL) {
        return false;
    }

    size_t expected_stride = 1;

    for (size_t dimension = t->rank; dimension > 0; dimension--) {
        const size_t index = dimension - 1;

        if (t->strides[index] != expected_stride) {
            return false;
        }

        expected_stride *= t->shape[index];
    }

    return true;
}

/* operations */
/* tensor-scalar ops*/

static TensorStatus valid_bin_scalar_op(const Tensor* t, const Tensor* out) {
    if (t == NULL || out == NULL || t->data == NULL || out->data == NULL) {
        return TENSOR_NULL_E;
    }
    if (!same_shape(t, out)) {
        return TENSOR_SHAPE_MISMATCH_E;
    }
    if (!tensor_is_contiguous(t) || !tensor_is_contiguous(out)) {
        return TENSOR_LAYOUT_E;
    }

    return TENSOR_OK;
}

TensorStatus tensor_try_add_scalar(const Tensor* t, float x, Tensor* out) {
    const TensorStatus status = valid_bin_scalar_op(t, out);

    if (status != TENSOR_OK) {
        return status;
    }

    for (size_t i = 0; i < t->no_elems; i++) {
        out->data[i] = (t->data[i] + x); // elem `op` scalar
    }

    return TENSOR_OK;
}

Tensor* tensor_add_scalar(const Tensor* t, float x, Tensor* out) {
    const TensorStatus status = tensor_try_add_scalar(t, x, out);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_add_scalar", status);
    }

    return out;
}

TensorStatus tensor_try_mult_scalar(const Tensor* t, float x, Tensor* out) {
    const TensorStatus status = valid_bin_scalar_op(t, out);

    if (status != TENSOR_OK) {
        return status;
    }

    for (size_t i = 0; i < t->no_elems; i++) {
        out->data[i] = (t->data[i] * x); // elem `op` scalar
    }

    return TENSOR_OK;
}

Tensor* tensor_mult_scalar(const Tensor* t, float x, Tensor* out) {
    const TensorStatus status = tensor_try_mult_scalar(t, x, out);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_mult_scalar", status);
    }

    return out;
}

/* tensor-tensor ops */

static TensorStatus valid_bin_tensor_op(const Tensor* t1, const Tensor* t2, const Tensor* res) {
    if (t1 == NULL || t2 == NULL || res == NULL || t1->data == NULL || t2->data == NULL ||
        res->data == NULL) {
        return TENSOR_NULL_E;
    }
    if (!same_shape(t1, t2) || !same_shape(t1, res)) {
        return TENSOR_SHAPE_MISMATCH_E;
    }
    if (!tensor_is_contiguous(t1) || !tensor_is_contiguous(t2) || !tensor_is_contiguous(res)) {
        return TENSOR_LAYOUT_E;
    }

    return TENSOR_OK;
}

// functions below all have the same shape, just copied out purely
// for efficiency
TensorStatus tensor_try_add(const Tensor* t1, const Tensor* t2, Tensor* res) {
    const TensorStatus status = valid_bin_tensor_op(t1, t2, res);

    if (status != TENSOR_OK) {
        return status;
    }

    for (size_t i = 0; i < t1->no_elems; i++) {
        res->data[i] = t1->data[i] + t2->data[i];
    }

    return TENSOR_OK;
}

Tensor* tensor_add(const Tensor* t1, const Tensor* t2, Tensor* res) {
    const TensorStatus status = tensor_try_add(t1, t2, res);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_add", status);
    }

    return res;
}

TensorStatus tensor_try_sub(const Tensor* t1, const Tensor* t2, Tensor* res) {
    const TensorStatus status = valid_bin_tensor_op(t1, t2, res);

    if (status != TENSOR_OK) {
        return status;
    }

    for (size_t i = 0; i < t1->no_elems; i++) {
        res->data[i] = t1->data[i] - t2->data[i];
    }

    return TENSOR_OK;
}

Tensor* tensor_sub(const Tensor* t1, const Tensor* t2, Tensor* res) {
    const TensorStatus status = tensor_try_sub(t1, t2, res);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_sub", status);
    }

    return res;
}

TensorStatus tensor_try_mult(const Tensor* t1, const Tensor* t2, Tensor* res) {
    const TensorStatus status = valid_bin_tensor_op(t1, t2, res);

    if (status != TENSOR_OK) {
        return status;
    }

    for (size_t i = 0; i < t1->no_elems; i++) {
        res->data[i] = t1->data[i] * t2->data[i];
    }

    return TENSOR_OK;
}

Tensor* tensor_mult(const Tensor* t1, const Tensor* t2, Tensor* res) {
    const TensorStatus status = tensor_try_mult(t1, t2, res);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_mult", status);
    }

    return res;
}

TensorStatus tensor_try_div(const Tensor* t1, const Tensor* t2, Tensor* res) {
    const TensorStatus status = valid_bin_tensor_op(t1, t2, res);

    if (status != TENSOR_OK) {
        return status;
    }

    for (size_t i = 0; i < t1->no_elems; i++) {
        res->data[i] = t1->data[i] / t2->data[i];
    }

    return TENSOR_OK;
}

Tensor* tensor_div(const Tensor* t1, const Tensor* t2, Tensor* res) {
    const TensorStatus status = tensor_try_div(t1, t2, res);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_div", status);
    }

    return res;
}

/* unary tensor ops */
static TensorStatus try_sum_elems(const Tensor* t, double* out) {
    // helper to find the sum of all elems in a tensor as a double
    // used for summing all values and finding mean
    // kept separately since summing may cause float overflow, but 
    // could still have a valid mean

    double sum = 0.0;

    for (size_t i = 0; i < t->no_elems; i++) {
        // all elems summed, sum is commutative, and tensors are dense
        // and this provides better spatial locality than taking into account
        // the stride pattern
        sum += (double) t->data[i];  
    }

    if (!isfinite(sum)) {
        return TENSOR_NUMERIC_E;
    }

    *out = sum;
    return TENSOR_OK;
}

static TensorStatus validate_unary_op(const Tensor* t, const Tensor* out) {
    // validation for mean and sum operations
    if (t == NULL || out == NULL || t->data == NULL || out->data == NULL) {
        return TENSOR_NULL_E;
    }

    if (out->rank != 0 || out->no_elems != 1) {
        return TENSOR_SHAPE_MISMATCH_E;
    }

    if (t->data == out->data) {
        return TENSOR_ALIAS_E;
    }

    return TENSOR_OK;
}

TensorStatus tensor_try_sum(const Tensor* t, Tensor* out) {
    const TensorStatus validation_status = validate_unary_op(t, out);

    if (validation_status != TENSOR_OK) {
        return validation_status;
    }
    
    double sum;
    const TensorStatus status = try_sum_elems(t, &sum);

    if (status != TENSOR_OK) {
        return status;
    }

    if (sum > FLT_MAX || sum < -FLT_MAX) {
        return TENSOR_NUMERIC_E;
    }

    out->data[0] = (float) sum;
    return TENSOR_OK;
}

Tensor* tensor_sum(const Tensor* t, Tensor* out) {
    const TensorStatus status = tensor_try_sum(t, out);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_sum", status);
    }
    return out;
}

TensorStatus tensor_try_mean(const Tensor* t, Tensor* out) {
    const TensorStatus validation_status = validate_unary_op(t, out);

    if (validation_status != TENSOR_OK) {
        return validation_status;
    }

    double sum;
    const TensorStatus status = try_sum_elems(t, &sum);

    if (status != TENSOR_OK) {
        return status;
    }

    if (t->no_elems == 0) {
        return TENSOR_SHAPE_E; // prevent division by 0
    }
    
    const double mean = sum / (double) t->no_elems;

    if (!isfinite(mean) || mean > FLT_MAX || mean < -FLT_MAX) {
        return TENSOR_NUMERIC_E;
    }

    out->data[0] = (float) mean;

    return TENSOR_OK;
}

Tensor* tensor_mean(const Tensor* t, Tensor* out) {
    const TensorStatus status = tensor_try_mean(t, out);

    if (status != TENSOR_OK) {
        tensor_panic("tensor_mean", status);
    }
    return out;
}

/* matrix ops */

bool matmultiplicable(const Tensor* m1, const Tensor* m2) {
    if (!is_matrix(m1) || !is_matrix(m2)) {
        return false;
    }

    return m1->shape[1] == m2->shape[0];
}

static TensorStatus valid_matrix(const Tensor* m) {
    // helper checks if a Tensor* is a valid matrix
    // (not null, not null data, correct rank, correct layout)
    if (m == NULL || m->data == NULL) {
        return TENSOR_NULL_E;
    }
    if (!is_matrix(m)) {
        return TENSOR_RANK_E;
    }
    if (!tensor_layout_fits_data(m)) {
        return TENSOR_LAYOUT_E;
    }

    return TENSOR_OK;
}

TensorStatus tensor_try_matmul(const Tensor* m1, const Tensor* m2, Tensor* out) {
    // tries matmul with result in Tensor* out
    // returns error on failure
    
    // check m1, m2, out valid
    TensorStatus status = valid_matrix(m1);

    if (status != TENSOR_OK) {
        return status;
    }

    status = valid_matrix(m2);
    if (status != TENSOR_OK) {
        return status;
    }

    status = valid_matrix(out);
    if (status != TENSOR_OK) {
        return status;
    }
    // check shapes are compatible
    if (!matmultiplicable(m1, m2) || out->shape[0] != m1->shape[0] ||
        out->shape[1] != m2->shape[1]) {
        return TENSOR_SHAPE_MISMATCH_E;
    }
    // check out is distinct from inputs
    if (out->data == m1->data || out->data == m2->data) {
        return TENSOR_ALIAS_E;
    }
    // check contiguity (not strided)
    if (!tensor_is_contiguous(m1) || !tensor_is_contiguous(m2) ||
        !tensor_is_contiguous(out)) {
        return TENSOR_LAYOUT_E;
    }
    // actual matmul
    const size_t m = m1->shape[0];
    const size_t k = m1->shape[1];
    const size_t n = m2->shape[1];

    memset(out->data, 0, out->no_elems * sizeof(*out->data)); // tensor 0

    for (size_t i = 0; i < m; ++i) {
        float* out_row = &out->data[i * n];     // start of out row
        const float* m1_row = &m1->data[i * k]; // start of m1 row

        for (size_t p = 0; p < k; ++p) {
            const float curr = m1_row[p];
            const float* m2_row = &m2->data[p * n]; // out row

            for (size_t j = 0; j < n; ++j) {
                out_row[j] += curr * m2_row[j];
            }
        }
    }

    return TENSOR_OK;
}

Tensor* matmul(const Tensor* m1, const Tensor* m2, Tensor* out) {
    // wrapper around try matmul that aborts on error
    const TensorStatus status = tensor_try_matmul(m1, m2, out);

    if (status != TENSOR_OK) {
        tensor_panic("matmul", status);
    }

    return out;
}

TensorStatus tensor_try_transpose_view(const Tensor* m, Tensor* out) {
    // tries to transspose view, returns error on failure

    if (out == NULL) {
        return TENSOR_NULL_E;
    }

    const TensorStatus status = valid_matrix(m);

    if (status != TENSOR_OK) {
        return status;
    }
    if (m == out) {
        return TENSOR_ALIAS_E;
    }

    Tensor view = *m;

    view.shape[0] = m->shape[1];
    view.shape[1] = m->shape[0];

    view.strides[0] = m->strides[1];
    view.strides[1] = m->strides[0];

    view.owns_data = false;

    *out = view;
    return TENSOR_OK;
}

Tensor transpose_view(const Tensor* m) {
    // wrapper around try transpose view
    Tensor out = {0};
    const TensorStatus status = tensor_try_transpose_view(m, &out);

    if (status != TENSOR_OK) {
        tensor_panic("transpose_view", status);
    }

    return out;
}

static inline size_t flatten_index(const Tensor* m, size_t i, size_t j) {
    // helper to work out physical address offset given logical i, j index
    return i * m->strides[0] + j * m->strides[1];
}

TensorStatus tensor_try_transpose(const Tensor* m, Tensor* out) {
    // tries full transpose, returns error on failure
    
    // check valid matrices
    TensorStatus status = valid_matrix(m);

    if (status != TENSOR_OK) {
        return status;
    }

    status = valid_matrix(out);
    if (status != TENSOR_OK) {
        return status;
    }
    // check distinct m and out
    if (out->data == m->data) {
        return TENSOR_ALIAS_E;
    }
    // check valid shape
    if (out->shape[0] != m->shape[1] || out->shape[1] != m->shape[0]) {
        return TENSOR_SHAPE_MISMATCH_E;
    }
    // actual transpose
    for (size_t i = 0; i < m->shape[0]; i++) {
        for (size_t j = 0; j < m->shape[1]; j++) {
            out->data[flatten_index(out, j, i)] = m->data[flatten_index(m, i, j)];
        }
    }

    return TENSOR_OK;
}

Tensor* transpose(const Tensor* m, Tensor* out) {
    // wrapper around try transpose
    const TensorStatus status = tensor_try_transpose(m, out);

    if (status != TENSOR_OK) {
        tensor_panic("transpose", status);
    }

    return out;
}

TensorStatus tensor_try_transpose_inplace(Tensor* m) {
    // inplace transpose (i.e. original changed to be transposed)
    const TensorStatus status = valid_matrix(m);

    if (status != TENSOR_OK) {
        return status;
    }
    if (m->shape[0] != m->shape[1]) {
        return TENSOR_SHAPE_MISMATCH_E;
    }

    for (size_t i = 0; i < m->shape[0]; i++) {
        for (size_t j = i + 1; j < m->shape[1]; j++) {
            size_t i1 = flatten_index(m, i, j);
            size_t i2 = flatten_index(m, j, i);

            float temp = m->data[i1];
            m->data[i1] = m->data[i2];
            m->data[i2] = temp;
        }
    }

    return TENSOR_OK;
}

Tensor* transpose_inplace(Tensor* m) {
    const TensorStatus status = tensor_try_transpose_inplace(m);

    if (status != TENSOR_OK) {
        tensor_panic("transpose_inplace", status);
    }

    return m;
}

const char* tensor_status_string(TensorStatus status) {
    // return string corresponding to status
    switch (status) {
        case TENSOR_OK:
            return "Success! No errors";
        case TENSOR_NULL_E:
            return "Null pointer error";
        case TENSOR_RANK_E:
            return "Invalid tensor rank";
        case TENSOR_SHAPE_E:
            return "Invalid tensor shape";
        case TENSOR_SIZE_OVERFLOW_E:
            return "Tensor size overflow";
        case TENSOR_ALLOC_E:
            return "Failed to allocate memory for tensor";
        case TENSOR_SHAPE_MISMATCH_E:
            return "Tensor shape mismatch";
        case TENSOR_ALIAS_E:
            return "Unsupported tensor aliasing";
        case TENSOR_LAYOUT_E:
            return "Unsupported tensor stride pattern";
        case TENSOR_INVALID_RANGE_E:
            return "Invalid range provided to tensor function";
        case TENSOR_DEVICE_E:
            return "Device operation failed";
    }
    return "unknown tensor error";
}
