/*
    Tests for matrix operations.
*/

#include "../../src/tensors/tensor.h"
#include "../test_common.h"

#include <stdlib.h>

static const float ABSOLUTE_TOLERANCE = 1.0e-6f;
static const float RELATIVE_TOLERANCE = 1.0e-6f;

static void expect_matrix_values(TestContext* context, const Tensor* actual,
                                 const float expected_data[]) {
    // check that the matrix data matches the expected data

    Tensor expected = tensor_from_data(actual->rank, actual->shape, expected_data);
    TEST_EXPECT(context,
                test_tensors_close(&expected, actual, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE));

    free_tensor(&expected);
}

static void test_matmultiplicable(TestContext* context) {
    // test that matmultiplicable function wroks as intended

    // create shapes
    const size_t left_shape[] = {2, 3};
    const size_t compatible_shape[] = {3, 4};
    const size_t incompatible_shape[] = {2, 4};

    Tensor left = make_tensor(MATRIX_RANK, left_shape);
    Tensor compatible = make_tensor(MATRIX_RANK, compatible_shape);
    Tensor incompatible = make_tensor(MATRIX_RANK, incompatible_shape);

    TEST_EXPECT(context, matmultiplicable(&left, &compatible));
    TEST_EXPECT(context, !matmultiplicable(&left, &incompatible));

    free_tensor(&incompatible);
    free_tensor(&compatible);
    free_tensor(&left);
}

static void test_matmul(TestContext* context) {
    // test that matmul works as intended

    // initialise shapes and data
    const size_t left_shape[] = {2, 3};
    const size_t right_shape[] = {3, 2};
    const size_t out_shape[] = {2, 2};
    const float left_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float right_data[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const float expected_data[] = {58.0f, 64.0f, 139.0f, 154.0f};

    // creates tensors
    Tensor left = tensor_from_data(MATRIX_RANK, left_shape, left_data);
    Tensor right = tensor_from_data(MATRIX_RANK, right_shape, right_data);
    Tensor original_left = tensor_copy(&left);
    Tensor original_right = tensor_copy(&right);
    Tensor out = make_tensor(MATRIX_RANK, out_shape);

    tensor_fill(&out, 100.0f); // fill out matrix, to check out contents overwritten

    // separate output matmul (inplace not allowed)
    // matmul should return same Tensor* as provided Tensor* out argument
    TEST_EXPECT(context, matmul(&left, &right, &out) == &out);

    // check correct values and original unchanged
    expect_matrix_values(context, &out, expected_data);
    TEST_EXPECT(context, test_tensors_deep_equal(&left, &original_left));
    TEST_EXPECT(context, test_tensors_deep_equal(&right, &original_right));

    free_tensor(&out);
    free_tensor(&original_right);
    free_tensor(&original_left);
    free_tensor(&right);
    free_tensor(&left);
}

static void test_transpose_view(TestContext* context) {
    // test that transposing view (i.e. changing strides + shape) works as intended

    // shapes + tensor initialisation
    const size_t shape[] = {2, 3};
    const float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float expected_data[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    Tensor original = tensor_from_data(MATRIX_RANK, shape, data);
    Tensor view = transpose_view(&original);

    // check shape items swap
    TEST_EXPECT(context, view.shape[0] == 3);
    TEST_EXPECT(context, view.shape[1] == 2);

    // check views swap (would be 3, 1)
    TEST_EXPECT(context, view.strides[0] == 1);
    TEST_EXPECT(context, view.strides[1] == 3);

    // check data refers to the same array, and is not owned
    TEST_EXPECT(context, view.no_elems == original.no_elems);
    TEST_EXPECT(context, view.data == original.data);
    TEST_EXPECT(context, !view.owns_data);
    expect_matrix_values(context, &view, expected_data);

    Tensor restored_view = transpose_view(&view); // (A^T)^T = A
    TEST_EXPECT(context, test_tensors_deep_equal(&original, &restored_view));
    TEST_EXPECT(context, restored_view.data == original.data);
    TEST_EXPECT(context, !restored_view.owns_data); // still doesn't own data

    free_tensor(&restored_view);
    free_tensor(&view);

    TEST_EXPECT(context, original.data != NULL); // original should be unchanged
    expect_matrix_values(context, &original, data);

    free_tensor(&original);
}

static void test_transpose(TestContext* context) {
    // checks that transpose function (which creates a full transpose in a new
    // matrix) works as intended

    // shape, data + tensor initialisation
    const size_t input_shape[] = {2, 3};
    const size_t out_shape[] = {3, 2};
    const float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float expected_data[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    Tensor input = tensor_from_data(MATRIX_RANK, input_shape, input_data);
    Tensor original = tensor_copy(&input);
    Tensor out = make_tensor(MATRIX_RANK, out_shape);

    // check transposing returns same Tensor* as the provided Tensor* out argument
    TEST_EXPECT(context, transpose(&input, &out) == &out);

    // check data matches expected
    expect_matrix_values(context, &out, expected_data);

    // check out matrix is distinct (i.e. different in memory and owned by out)
    TEST_EXPECT(context, out.data != input.data);
    TEST_EXPECT(context, out.owns_data);

    // check original unchanged
    TEST_EXPECT(context, test_tensors_deep_equal(&input, &original));

    Tensor restored = make_tensor(MATRIX_RANK, input_shape);
    TEST_EXPECT(context, transpose(&out, &restored) == &restored);    // (A^T)^T = A
    TEST_EXPECT(context, test_tensors_deep_equal(&input, &restored)); // should have same data

    free_tensor(&restored);
    free_tensor(&out);
    free_tensor(&original);
    free_tensor(&input);
}

static void test_transpose_inplace(TestContext* context) {
    // check that an inplace transpose works for square matrices

    const size_t shape[] = {3, 3};
    const float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const float expected_data[] = {1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f};
    Tensor matrix = tensor_from_data(MATRIX_RANK, shape, input_data);
    float* const original_data = matrix.data;

    // check that the same Tensor* that's provided is returned
    TEST_EXPECT(context, transpose_inplace(&matrix) == &matrix);

    expect_matrix_values(context, &matrix, expected_data);

    // should have the same location in memory (NOT NECESSARILY SAME DATA)
    TEST_EXPECT(context, matrix.data == original_data);

    // same shape + strides as before
    TEST_EXPECT(context, matrix.shape[0] == 3 && matrix.shape[1] == 3);
    TEST_EXPECT(context, matrix.strides[0] == 3 && matrix.strides[1] == 1);

    // restore and check that it's as expected
    TEST_EXPECT(context, transpose_inplace(&matrix) == &matrix);
    expect_matrix_values(context, &matrix, input_data);

    free_tensor(&matrix);
}

int main(void) {
    TestContext context = {0};

    test_run(&context, "matmultiplicable", test_matmultiplicable);
    test_run(&context, "matmul", test_matmul);
    test_run(&context, "transpose_view", test_transpose_view);
    test_run(&context, "transpose", test_transpose);
    test_run(&context, "transpose_inplace", test_transpose_inplace);

    return test_summary(&context);
}
