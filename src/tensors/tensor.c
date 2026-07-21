#include "tensor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/*
typedef struct tensor_s {
    size_t rank; // no. of dimensions (0 - scalar, 1 - vector, 2 - matrix ..)
    size_t shape[MAX_DIMS]; // shape[n] = size of dimension n
    size_t strides[MAX_DIMS]; // stores how many positions we move in memory if index increases by 1
    size_t no_elems; // total no. elems
    float* data; // malloc-ed block of memory
    bool owns_data; // if not, dont need to call free
} Tensor;
*/

static void compute_strides(Tensor* t) {
    assert( t != NULL );

    if (t->rank == 0) return; // scalar strides is empty

    t->strides[t->rank - 1] = 1; // bottom up, final stride is always 1

    // from the next dimension stride and length, work out curr stride
    for (size_t i = t->rank - 1; i > 0; --i) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }

}


Tensor make_tensor(size_t rank, const size_t shape[]) {
    assert ( rank <= MAX_DIMS ); // rank is unsigned so no need for >= 0
    assert( shape != NULL || rank == 0 ); // if rank = 0, then scalar, shape can be null

    Tensor t = {0}; // set all fields to 0

    t.rank = rank;
    t.no_elems = 1;

    for (size_t i = 0; i < rank; i++) {
        t.no_elems *= shape[i];
        t.shape[i] = shape[i];
    }

    t.data = calloc(t.no_elems, sizeof(*t.data));
    t.owns_data = true;

    if (t.data == NULL) {
        fprintf(stderr, "Error while malloc-ing data in make_tensor\n");
        exit(EXIT_FAILURE);
    }

    compute_strides(&t);

    return t;

}



Tensor make_tensor(size_t rank, const size_t shape[]) {
    
    
    assert( shape != NULL || rank = 0 ); // if rank is 0, shape can be null

    Tensor t = {0}; // by default all fields set to 0

    t.rank = rank;

    t.no_elems = 1;

    for (size_t i = 0; i < rank; i++) {
        t.no_elems *= shape[i];
        t.shape[i] = shape[i];
    }

    t.data = calloc(t.no_elems, sizeof(*t.data));
        t.owns_data = true;

    

    compute_strides(&t);
    return t;
}

void free_tensor(Tensor* t) {
    assert ( t != NULL );

    if (t->owns_data) {
        free(t->data);
    }

    *t = (Tensor){0}; // reset Tensor to 0
}