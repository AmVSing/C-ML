/*
    implementation of tensors
*/

#include "tensor.h"
#include <assert.h>
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

// Tensor creations
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

static Tensor make_tensor_uninit(size_t rank, const size_t shape[]) {
    assert(rank <= MAX_DIMS);           // rank is unsigned so no need for >= 0
    assert(shape != NULL || rank == 0); // if rank = 0, then scalar, shape can be null

    Tensor t = {0}; // set all fields to 0

    t.rank = rank;
    t.no_elems = 1;

    for (size_t i = 0; i < rank; i++) { // doesn't run for scalar
        assert(shape[i] > 0);           // handles dimensions with 0 length

        if (t.no_elems > MAX_ELEMS / shape[i]) {
            fprintf(stderr, "Tensor has too many elements\n");
            exit(EXIT_FAILURE);
        }
        t.no_elems *= shape[i];
        t.shape[i] = shape[i];
    }

    // check if nmemb for malloc can be represented as a size_t
    // technically redundant given current max elems and type of t.data
    if (t.no_elems > SIZE_MAX / sizeof(*t.data)) {
        fprintf(stderr, "Tensor is too large to be malloc-d\n");
        exit(EXIT_FAILURE);
    }

    t.data = malloc(t.no_elems * sizeof(*t.data));
    t.owns_data = true;

    if (t.data == NULL) {
        fprintf(stderr, "Error while malloc-ing data in make_tensor\n");
        exit(EXIT_FAILURE);
    }

    compute_strides(&t);

    return t;
}

Tensor make_tensor(size_t rank, const size_t shape[]) {
    // calls make_tensor_uninit then sets every element to 0
    Tensor t = make_tensor_uninit(rank, shape);

    memset(t.data, 0, t.no_elems * sizeof(*(t.data)));

    return t;
}

Tensor tensor_from_data(size_t rank, const size_t shape[], const float* data) {
    assert(data != NULL);
    Tensor t = make_tensor_uninit(rank, shape);

    memcpy(t.data, data, t.no_elems * sizeof(*data));

    return t;
}

inline Tensor tensor_copy(const Tensor* t) {
    assert(t != NULL);
    return tensor_from_data(t->rank, t->shape, t->data);
}
// helpful functions

static float rand_float(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

Tensor* tensor_fill(Tensor* t, float value) {
    assert(t != NULL);

    for (size_t i = 0; i < t->no_elems; i++) {
        t->data[i] = value;
    }
    return t;
}

Tensor* tensor_rand(Tensor* t, float min, float max) {
    assert(t != NULL);

    for (size_t i = 0; i < t->no_elems; i++) {
        t->data[i] = rand_float(min, max);
    }

    return t;
}

// free-ing
void free_tensor(Tensor* t) {
    assert(t != NULL);

    if (t->owns_data) {
        free(t->data);
    }

    *t = (Tensor){0}; // reset Tensor to 0
}

// shape comparison functions

bool same_shape(const Tensor* t1, const Tensor* t2) {
    assert(t1 != NULL);
    assert(t2 != NULL);

    if (t1->rank != t2->rank)
        return false;

    for (size_t i = 0; i < t1->rank; i++) {
        if (t1->shape[i] != t2->shape[i])
            return false;
    }
    return true;
}

inline bool is_matrix(const Tensor* t) {
    assert(t != NULL);
    return (t->rank == MATRIX_RANK);
}

// operations
// tensor-scalar ops

static void valid_bin_scalar_op(const Tensor* t, const Tensor* out) {
    assert(t != NULL);
    assert(out != NULL);

    assert(same_shape(t, out));
}

Tensor* tensor_add_scalar(const Tensor* t, float x, Tensor* out) {
    valid_bin_scalar_op(t, out);

    for (size_t i = 0; i < t->no_elems; i++) {
        out->data[i] = (t->data[i] + x); // elem `op` scalar
    }

    return out;
}

Tensor* tensor_mult_scalar(const Tensor* t, float x, Tensor* out) {
    valid_bin_scalar_op(t, out);

    for (size_t i = 0; i < t->no_elems; i++) {
        out->data[i] = (t->data[i] * x); // elem `op` scalar
    }

    return out;
}

// Tensor-tensor ops

static void valid_bin_tensor_op(const Tensor* t1, const Tensor* t2, const Tensor* res) {
    assert(t1 != NULL);
    assert(t2 != NULL);
    assert(res != NULL);

    assert(same_shape(t1, t2));
    assert(same_shape(t1, res));
}
// functions below all have the same shape, just copied out purely
// for efficiency
Tensor* tensor_add(const Tensor* t1, const Tensor* t2, Tensor* res) {
    valid_bin_tensor_op(t1, t2, res);

    for (size_t i = 0; i < t1->no_elems; i++) {
        res->data[i] = t1->data[i] + t2->data[i];
    }
    return res;
}
Tensor* tensor_sub(const Tensor* t1, const Tensor* t2, Tensor* res) {
    valid_bin_tensor_op(t1, t2, res);

    for (size_t i = 0; i < t1->no_elems; i++) {
        res->data[i] = t1->data[i] - t2->data[i];
    }
    return res;
}
Tensor* tensor_mult(const Tensor* t1, const Tensor* t2, Tensor* res) {
    valid_bin_tensor_op(t1, t2, res);

    for (size_t i = 0; i < t1->no_elems; i++) {
        res->data[i] = t1->data[i] * t2->data[i];
    }
    return res;
}

Tensor* tensor_div(const Tensor* t1, const Tensor* t2, Tensor* res) {
    valid_bin_tensor_op(t1, t2, res);

    for (size_t i = 0; i < t1->no_elems; i++) {
        res->data[i] = t1->data[i] / t2->data[i];
    }
    return res;
}

// matrix ops

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