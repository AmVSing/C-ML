/*
    Tests for tensor initialization functions.
*/

#include "../../src/tensors/tensor.h"
#include "../test_common.h"

#include <stdbool.h>
#include <stdlib.h>

static void correct_metadata(TestContext* context, const Tensor* tensor, size_t rank,
                             const size_t shape[], const size_t strides[], size_t no_elems) {
    // check that a tensor has the expected metadata (not checking data)
    TEST_EXPECT(context, tensor->rank == rank);
    TEST_EXPECT(context, tensor->no_elems == no_elems);
    TEST_EXPECT(context, tensor->data != NULL);
    TEST_EXPECT(context, tensor->owns_data);

    for (size_t dimension = 0; dimension < rank; dimension++) {
        TEST_EXPECT(context, tensor->shape[dimension] == shape[dimension]);
        TEST_EXPECT(context, tensor->strides[dimension] == strides[dimension]);
    }
}

static void test_make_tensor(TestContext* context) {
    // check that make_tensor works as expected
    // scalar case
    Tensor scalar = make_tensor(0, NULL); // scalar, shape is null, rank = 0

    TEST_EXPECT(context, scalar.rank == 0);
    TEST_EXPECT(context, scalar.no_elems == 1);
    TEST_EXPECT(context, scalar.data != NULL);    // check successful malloc
    TEST_EXPECT(context, scalar.data[0] == 0.0f); // check initialised to 0
    TEST_EXPECT(context, scalar.owns_data);       // should own

    // 3D test case
    size_t shape[] = {2, 3, 4};
    const size_t expected_shape[] = {2, 3, 4};
    const size_t expected_strides[] = {12, 4, 1};
    // strides[0] = 4 * 3 * 1
    // strides [1] = 4 * 1
    // strides[2] = 1
    Tensor tensor = make_tensor(3, shape);

    correct_metadata(context, &tensor, 3, expected_shape, expected_strides, 24);

    // each element should be initialised as 0
    for (size_t i = 0; i < tensor.no_elems; ++i) {
        TEST_EXPECT(context, tensor.data[i] == 0.0f);
    }

    // check that make_tensor copies shape, and doesn't rely on the caller array
    shape[0] = 99;
    TEST_EXPECT(context, tensor.shape[0] == 2); // shoueld remain the same

    free_tensor(&tensor);
    free_tensor(&scalar);
}

static void test_tensor_from_data(TestContext* context) {
    // check that tensor_from_data works as intended

    const size_t shape[] = {2, 3};   // 2 x 3 matrix
    const size_t strides[] = {3, 1}; // strides[0] = 3 * 1, strides[1] = 1
    float source[] = {1.0f, -2.5f, 3.25f, 4.0f, 5.5f, -6.0f};
    const float expected[] = {1.0f, -2.5f, 3.25f, 4.0f, 5.5f, -6.0f};
    Tensor tensor = tensor_from_data(MATRIX_RANK, shape, source);

    correct_metadata(context, &tensor, MATRIX_RANK, shape, strides, 6);

    // should copy the array, shouldn't rely on source array
    TEST_EXPECT(context, tensor.data != source);
    source[0] = 1000.0f;

    for (size_t i = 0; i < tensor.no_elems; ++i) {
        TEST_EXPECT(context, tensor.data[i] == expected[i]);
    }

    free_tensor(&tensor);
}

static void test_tensor_copy(TestContext* context) {
    // check tensor_copy creates a deep copy

    const size_t shape[] = {2, 2};
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor original = tensor_from_data(MATRIX_RANK, shape, values);
    Tensor copy = tensor_copy(&original);

    TEST_EXPECT(context, test_tensors_deep_equal(&original, &copy));
    // should copy data without relying on original data
    TEST_EXPECT(context, copy.data != original.data);
    TEST_EXPECT(context, copy.owns_data); // should own its own data

    original.data[0] = 100.0f; // shouldn't change copy
    TEST_EXPECT(context, copy.data[0] == values[0]);

    copy.data[1] = -100.0f; // shouldn't change the original
    TEST_EXPECT(context, original.data[1] == values[1]);

    free_tensor(&copy);
    free_tensor(&original);
}

static void test_checked_constructors(TestContext* context) {
    const size_t shape[] = {2, 2};
    const size_t zero_shape[] = {2, 0};
    const size_t overflow_shape[] = {MAX_ELEMS, 2};
    const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor tensor = {0};

    TEST_EXPECT(context, tensor_try_make(&tensor, MATRIX_RANK, shape) == TENSOR_OK);
    free_tensor(&tensor);

    TEST_EXPECT(context,
                tensor_try_from_data(&tensor, MATRIX_RANK, shape, data) == TENSOR_OK);

    Tensor copy = {0};
    TEST_EXPECT(context, tensor_try_copy(&copy, &tensor) == TENSOR_OK);
    TEST_EXPECT(context, test_tensors_deep_equal(&tensor, &copy));
    free_tensor(&copy);
    free_tensor(&tensor);

    TEST_EXPECT(context, tensor_try_make(NULL, MATRIX_RANK, shape) == TENSOR_NULL_E);
    TEST_EXPECT(context, tensor_try_make(&tensor, 1, NULL) == TENSOR_NULL_E);
    TEST_EXPECT(context, tensor_try_make(&tensor, MAX_DIMS + 1, shape) == TENSOR_RANK_E);
    TEST_EXPECT(context, tensor_try_make(&tensor, 2, zero_shape) == TENSOR_SHAPE_E);
    TEST_EXPECT(context,
                tensor_try_make(&tensor, 2, overflow_shape) == TENSOR_SIZE_OVERFLOW_E);
    TEST_EXPECT(context,
                tensor_try_from_data(&tensor, MATRIX_RANK, shape, NULL) == TENSOR_NULL_E);
    TEST_EXPECT(context, tensor_try_copy(&tensor, NULL) == TENSOR_NULL_E);
    TEST_EXPECT(context, tensor.data == NULL);
}

static void test_tensor_fill(TestContext* context) {
    // check that tensor_fill works as intended

    const size_t shape[] = {2, 3};
    Tensor tensor = make_tensor(MATRIX_RANK, shape);

    // after filling, the same Tensor* that was passed in should be returned
    const float fill_value_1 = -3.25f;
    TEST_EXPECT(context, tensor_fill(&tensor, fill_value_1) == &tensor);

    for (size_t i = 0; i < tensor.no_elems; i++) {
        TEST_EXPECT(context, tensor.data[i] == fill_value_1); // all should be -3.25
    }

    // check another fill also works
    const float fill_value_2 = 7.5;
    // after filling, the same Tensor* that was passed in should be returned
    TEST_EXPECT(context, tensor_fill(&tensor, fill_value_2) == &tensor);

    for (size_t i = 0; i < tensor.no_elems; ++i) {
        TEST_EXPECT(context, tensor.data[i] == fill_value_2);
    }

    free_tensor(&tensor);
}

static void test_tensor_rand(TestContext* context) {
    // test that tensor_rand works as expected
    const size_t shape[] = {16, 16};
    const float min_1 = -4.0f;
    const float max_1 = 6.0f;
    Tensor tensor = make_tensor(MATRIX_RANK, shape);

    srand(31415); // same seed so it's deterministic

    // should return the same Tensor*
    TEST_EXPECT(context, tensor_rand(&tensor, min_1, max_1) == &tensor);

    bool found_different_values = false;
    // check all random values are within [min, max]
    for (size_t i = 0; i < tensor.no_elems; ++i) {
        TEST_EXPECT(context, tensor.data[i] >= min_1);
        TEST_EXPECT(context, tensor.data[i] <= max_1);

        if (i > 0 && tensor.data[i] != tensor.data[0]) {
            found_different_values = true;
        }
    }
    TEST_EXPECT(context, found_different_values);

    const float min_and_max = 2.5f;

    TEST_EXPECT(context,
                tensor_rand(&tensor, min_and_max, min_and_max) == &tensor); // should just fill
    for (size_t i = 0; i < tensor.no_elems; ++i) {
        TEST_EXPECT(context, tensor.data[i] == min_and_max);
    }

    free_tensor(&tensor);
}

static void test_checked_fill_and_rand(TestContext* context) {
    const size_t shape[] = {2, 3};
    Tensor tensor = make_tensor(MATRIX_RANK, shape);
    Tensor transposed_view = transpose_view(&tensor);

    TEST_EXPECT(context, tensor_try_fill(&tensor, 3.0f) == TENSOR_OK);
    TEST_EXPECT(context, tensor_try_rand(&tensor, -1.0f, 1.0f) == TENSOR_OK);

    TEST_EXPECT(context, tensor_try_fill(NULL, 3.0f) == TENSOR_NULL_E);
    TEST_EXPECT(context, tensor_try_rand(NULL, -1.0f, 1.0f) == TENSOR_NULL_E);
    TEST_EXPECT(context, tensor_try_rand(&tensor, 1.0f, -1.0f) == TENSOR_INVALID_RANGE_E);
    TEST_EXPECT(context, tensor_try_fill(&transposed_view, 3.0f) == TENSOR_LAYOUT_E);
    TEST_EXPECT(context,
                tensor_try_rand(&transposed_view, -1.0f, 1.0f) == TENSOR_LAYOUT_E);

    free_tensor(&transposed_view);
    free_tensor(&tensor);
}

int main(void) {
    // run all the tests
    TestContext context = {0}; // initialise context

    test_run(&context, "make_tensor", test_make_tensor);
    test_run(&context, "tensor_from_data", test_tensor_from_data);
    test_run(&context, "tensor_copy", test_tensor_copy);
    test_run(&context, "checked constructors", test_checked_constructors);
    test_run(&context, "tensor_fill", test_tensor_fill);
    test_run(&context, "tensor_rand", test_tensor_rand);
    test_run(&context, "checked fill and rand", test_checked_fill_and_rand);

    return test_summary(&context);
}
