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

TensorStatus tensor_try_from_data(Tensor* out, size_t rank, const size_t shape[], const float* data) {
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

TensorStatus tensor_try_copy(Tensor* out, const Tensor* t) {
    // tries copying to out, on error returns corresponding status
    if (out == NULL || t == NULL) {
        return TENSOR_NULL_E;
    }
    if (out == t) {
        return TENSOR_ALIAS_E; // args should be distinct
    }

    return tensor_try_from_data(out, t->rank, t->shape, t->data);
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
    assert(t != NULL);

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

/* matrix ops */

bool matmultiplicable(const Tensor* m1, const Tensor* m2) {
    assert(is_matrix(m1));
    assert(is_matrix(m2));

    return m1->shape[1] == m2->shape[0];
}

Tensor* matmul(const Tensor* m1, const Tensor* m2, Tensor* out) {
    // assumes contiguous matrices

    assert(matmultiplicable(m1, m2));
    assert(is_matrix(out));
    assert(out != m1 && out != m2);
    assert(out->shape[0] == m1->shape[0]);
    assert(out->shape[1] == m2->shape[1]);

    // m x k `matmul` k x n -> m x n
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
    return out;
}

Tensor transpose_view(const Tensor* m) {
    // INVALID WHEN ORIGINAL TENSOR IS FREE'D
    assert(is_matrix(m)); // already checks for null

    Tensor out = *m;

    out.shape[0] = m->shape[1];
    out.shape[1] = m->shape[0];

    out.strides[0] = m->strides[1];
    out.strides[1] = m->strides[0];

    out.owns_data = false;

    return out;
}

static inline size_t flatten_index(const Tensor* m, size_t i, size_t j) {
    return i * m->strides[0] + j * m->strides[1];
}

Tensor* transpose(const Tensor* m, Tensor* out) {

    // check not null and matrices
    assert(is_matrix(m));
    assert(is_matrix(out));

    // check shapes align
    assert(out->shape[0] == m->shape[1]);
    assert(out->shape[1] == m->shape[0]);

    for (size_t i = 0; i < m->shape[0]; i++) {
        for (size_t j = 0; j < m->shape[1]; j++) {
            out->data[flatten_index(out, j, i)] = m->data[flatten_index(m, i, j)];
        }
    }

    return out;
}

Tensor* transpose_inplace(Tensor* m) {
    // transposes matrix in place (i.e. output Tensor* is provided Tensor*)
    assert(is_matrix(m));
    assert(m->shape[0] == m->shape[1]); // square only

    for (size_t i = 0; i < m->shape[0]; i++) {
        for (size_t j = i + 1; j < m->shape[1]; j++) {
            size_t i1 = flatten_index(m, i, j);
            size_t i2 = flatten_index(m, j, i);

            float temp = m->data[i1];
            m->data[i1] = m->data[i2];
            m->data[i2] = temp;
        }
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
