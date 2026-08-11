/*
    A collection of helpers used in testing
*/

#include "test_common.h"

#include <math.h>
#include <stdio.h>

static const int FIELD_WIDTH = 20; // used for aligning test results

void test_expect(
    TestContext* context, 
    bool condition, 
    const char* file, 
    int line, 
    const char* expression) {

    (context->assertions)++;

    if (!condition) {
        fprintf(stderr, "FAIL: %s:%d: %s\n", file, line, expression);
        (context->failures)++;
    }
}

void test_run(TestContext* context, const char* name, TestFunction function) {
    const int failures_before = context->failures;
    function(context);
    printf("%-*s %s\n", FIELD_WIDTH, name, 
        context->failures == failures_before ? "PASS" : "FAIL");
}

int test_summary(const TestContext* context) {
    printf("\n%d assertions, %d failure(s)\n", context->assertions, context->failures);
    return context->failures == 0 ? 0 : 1;
}

bool test_floats_close(
    float expected,
    float actual,
    float absolute_tolerance,
    float relative_tolerance) {

    // early return if they are considered equal under ==
    if (expected == actual) {
        return true;
    }
    
    // if either are +/- inf or NaN then early return
    if (!isfinite(expected) || !isfinite(actual)) {
        return false; 
    }

    const float difference = fabsf(expected - actual);
    const float scale = fmaxf(fabsf(expected), fabsf(actual));
    return difference <= absolute_tolerance + (relative_tolerance * scale);
}

static size_t logical_offset(const Tensor* tensor, size_t linear_index) {
    // treating the linear_index as the flattened logical index
    // used so that checking for equality is indeepndent of the stride pattern
    size_t offset = 0;

    for (size_t dimension = tensor->rank; dimension > 0; dimension--) {
        const size_t index = linear_index % tensor->shape[dimension - 1];
        linear_index /= tensor->shape[dimension - 1];
        offset += index * tensor->strides[dimension - 1];
    }

    return offset;
}

bool test_tensors_close(const Tensor* expected,
                        const Tensor* actual,
                        float absolute_tolerance,
                        float relative_tolerance) {
    if (expected == NULL || actual == NULL || expected->rank != actual->rank ||
        expected->no_elems != actual->no_elems) {
        return false;
    }

    for (size_t dimension = 0; dimension < expected->rank; dimension++) {
        if (expected->shape[dimension] != actual->shape[dimension]) {
            return false;
        }
    }

    for (size_t i = 0; i < expected->no_elems; ++i) {
        const float expected_value = expected->data[logical_offset(expected, i)];
        const float actual_value = actual->data[logical_offset(actual, i)];

        if (!test_floats_close(
                expected_value, actual_value, absolute_tolerance, relative_tolerance)) {
            return false;
        }
    }

    return true;
}

bool test_tensors_deep_equal(const Tensor* expected, const Tensor* actual) {
    return test_tensors_close(expected, actual, 0.0f, 0.0f);
}
