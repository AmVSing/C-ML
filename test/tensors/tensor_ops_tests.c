/*
    Tests for tensor shape helpers, scalar operations, and elementwise operations.
*/

#include "../test_common.h"

#include <stdlib.h>

static const float ABSOLUTE_TOLERANCE = 1.0e-6f;
static const float RELATIVE_TOLERANCE = 1.0e-6f;

typedef Tensor* (*TensorBinaryFunction)(const Tensor* left, const Tensor* right, Tensor* out);

typedef Tensor* (*TensorScalarFunction)(const Tensor* input, float scalar, Tensor* out);

static void test_expected_actual_close(TestContext* context, const Tensor* actual,
                                       const float expected_data[]) {
    // wrapper around test_tensor_close that creates a tensor representing
    // the expected tensor and then tests if the actual and expected are
    // close
    Tensor expected = tensor_from_data(actual->rank, actual->shape, expected_data);

    TEST_EXPECT(context,
                test_tensors_close(&expected, actual, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE));

    free_tensor(&expected);
}

static void test_same_shape(TestContext* context) {
    // test same_shape works as intended

    // these two should be the same as each other
    const size_t matrix_shape[] = {2, 3};
    const size_t same_matrix_shape[] = {2, 3};

    // should be different when compared to every other:
    const size_t reordered_shape[] = {3, 2};
    const size_t vector_shape[] = {6};

    Tensor matrix = make_tensor(MATRIX_RANK, matrix_shape);
    Tensor same_matrix = make_tensor(MATRIX_RANK, same_matrix_shape);
    Tensor reordered = make_tensor(MATRIX_RANK, reordered_shape);
    Tensor vector = make_tensor(1, vector_shape);

    // check same and commutative
    TEST_EXPECT(context, same_shape(&matrix, &matrix));
    TEST_EXPECT(context, same_shape(&matrix, &same_matrix));

    // check not same
    TEST_EXPECT(context, !same_shape(&matrix, &reordered));
    TEST_EXPECT(context, !same_shape(&matrix, &vector));

    free_tensor(&vector);
    free_tensor(&reordered);
    free_tensor(&same_matrix);
    free_tensor(&matrix);
}

static void test_is_matrix(TestContext* context) {
    // check is_matrix works as intended

    const size_t matrix_shape[] = {2, 3};
    const size_t vector_shape[] = {6};

    Tensor scalar = make_tensor(0, NULL);
    Tensor vector = make_tensor(1, vector_shape);
    Tensor matrix = make_tensor(MATRIX_RANK, matrix_shape);

    TEST_EXPECT(context, !is_matrix(&scalar));
    TEST_EXPECT(context, !is_matrix(&vector));
    TEST_EXPECT(context, is_matrix(&matrix));

    free_tensor(&matrix);
    free_tensor(&vector);
    free_tensor(&scalar);
}

static void test_tensor_op_scalar_helper(TestContext* context, TensorScalarFunction op,
                                         Tensor* tensor, float scalar,
                                         const float expected_data[]) {
    // helper for scalar addition and scalar multiplication operations
    Tensor out = make_tensor(tensor->rank, tensor->shape);
    Tensor original_copy = tensor_copy(tensor);

    // check return out Tensor*
    TEST_EXPECT(context, (*op)(tensor, scalar, &out) == &out);

    // check close to expected
    test_expected_actual_close(context, &out, expected_data);

    // original tensor should be unmodified
    TEST_EXPECT(context, test_tensors_deep_equal(tensor, &original_copy));
    // test where output = input
    TEST_EXPECT(context, (*op)(tensor, scalar, tensor) == tensor);
    test_expected_actual_close(context, tensor, expected_data);

    free_tensor(&original_copy);
    free_tensor(&out);
}

static void test_tensor_add_scalar(TestContext* context) {
    // checks tensor_add_scalar works as intended
    // should add constant to every element in tensor

    // initialise input data
    const size_t shape[] = {2, 3};
    const float input_data[6] = {1.0f, -2.5f, 0.0f, 4.25f, -8.0f, 10.5f};
    const float add_value = 2.0f;
    float expected_data[6];

    for (size_t i = 0; i < 6; i++) {
        expected_data[i] = input_data[i] + add_value;
    }
    // create tensor and test
    Tensor tensor = tensor_from_data(MATRIX_RANK, shape, input_data);
    test_tensor_op_scalar_helper(context, tensor_add_scalar, &tensor, add_value, expected_data);

    free_tensor(&tensor);
}

static void test_tensor_mult_scalar(TestContext* context) {
    // tests that tensor_mult_scalar works as intended
    // should multiply every element by constant

    // initialise input data
    const size_t shape[] = {2, 3};
    const float input_data[6] = {1.0f, -2.5f, 0.0f, 4.25f, -8.0f, 10.5f};
    const float mult_value = -0.5f;
    float expected_data[6];

    for (size_t i = 0; i < 6; i++) {
        expected_data[i] = input_data[i] * mult_value;
    }
    // create tensor and test
    Tensor tensor = tensor_from_data(MATRIX_RANK, shape, input_data);
    test_tensor_op_scalar_helper(context, tensor_mult_scalar, &tensor, mult_value, expected_data);

    free_tensor(&tensor);
}

static void test_binary_op_helper(TestContext* context, TensorBinaryFunction function, Tensor* left,
                                  Tensor* right, const float expected_data[]) {
    // helper for tensor-tensor ops

    // create out Tensor
    Tensor out = make_tensor(left->rank, left->shape);
    Tensor original_left = tensor_copy(left);
    Tensor original_right = tensor_copy(right);

    // should return Tensor* corresponding to out
    TEST_EXPECT(context, function(left, right, &out) == &out);
    // should match expected
    test_expected_actual_close(context, &out, expected_data);

    // input tensors should be unchanged
    TEST_EXPECT(context, test_tensors_deep_equal(left, &original_left));
    TEST_EXPECT(context, test_tensors_deep_equal(right, &original_right));

    // inplace, should modify left
    TEST_EXPECT(context, function(left, right, left) == left);
    test_expected_actual_close(context, left, expected_data);

    // right should be unchanged
    TEST_EXPECT(context, test_tensors_deep_equal(right, &original_right));

    free_tensor(&original_left);
    free_tensor(&original_right);
    free_tensor(&out);
}

static void test_tensor_add(TestContext* context) {
    const size_t shape[] = {2, 3};
    const float left_data[] = {8.0f, -6.0f, 4.0f, 2.0f, -1.5f, 0.5f};
    const float right_data[] = {2.0f, 3.0f, -4.0f, 0.5f, -0.5f, 2.0f};
    const float expected[] = {10.0f, -3.0f, 0.0f, 2.5f, -2.0f, 2.5f};

    Tensor left = tensor_from_data(MATRIX_RANK, shape, left_data);
    Tensor right = tensor_from_data(MATRIX_RANK, shape, right_data);
    test_binary_op_helper(context, tensor_add, &left, &right, expected);

    free_tensor(&left);
    free_tensor(&right);
}

static void test_tensor_sub(TestContext* context) {
    const size_t shape[] = {2, 3};
    const float left_data[] = {8.0f, -6.0f, 4.0f, 2.0f, -1.5f, 0.5f};
    const float right_data[] = {2.0f, 3.0f, -4.0f, 0.5f, -0.5f, 2.0f};
    const float expected[] = {6.0f, -9.0f, 8.0f, 1.5f, -1.0f, -1.5f};

    Tensor left = tensor_from_data(MATRIX_RANK, shape, left_data);
    Tensor right = tensor_from_data(MATRIX_RANK, shape, right_data);

    test_binary_op_helper(context, tensor_sub, &left, &right, expected);

    free_tensor(&left);
    free_tensor(&right);
}

static void test_tensor_mult(TestContext* context) {
    const size_t shape[] = {2, 3};
    const float left_data[] = {8.0f, -6.0f, 4.0f, 2.0f, -1.5f, 0.5f};
    const float right_data[] = {2.0f, 3.0f, -4.0f, 0.5f, -0.5f, 2.0f};
    const float expected[] = {16.0f, -18.0f, -16.0f, 1.0f, 0.75f, 1.0f};

    Tensor left = tensor_from_data(MATRIX_RANK, shape, left_data);
    Tensor right = tensor_from_data(MATRIX_RANK, shape, right_data);

    test_binary_op_helper(context, tensor_mult, &left, &right, expected);

    free_tensor(&left);
    free_tensor(&right);
}

static void test_tensor_div(TestContext* context) {
    const size_t shape[] = {2, 3};
    const float left_data[] = {8.0f, -6.0f, 4.0f, 2.0f, -1.5f, 0.5f};
    const float right_data[] = {2.0f, 3.0f, -4.0f, 0.5f, -0.5f, 2.0f};
    const float expected[] = {4.0f, -2.0f, -1.0f, 4.0f, 3.0f, 0.25f};

    Tensor left = tensor_from_data(MATRIX_RANK, shape, left_data);
    Tensor right = tensor_from_data(MATRIX_RANK, shape, right_data);

    test_binary_op_helper(context, tensor_div, &left, &right, expected);

    free_tensor(&left);
    free_tensor(&right);
}

int main(void) {
    TestContext context = {0};

    test_run(&context, "same_shape", test_same_shape);
    test_run(&context, "is_matrix", test_is_matrix);
    test_run(&context, "tensor_add_scalar", test_tensor_add_scalar);
    test_run(&context, "tensor_mult_scalar", test_tensor_mult_scalar);
    test_run(&context, "tensor_add", test_tensor_add);
    test_run(&context, "tensor_sub", test_tensor_sub);
    test_run(&context, "tensor_mult", test_tensor_mult);
    test_run(&context, "tensor_div", test_tensor_div);

    return test_summary(&context);
}
