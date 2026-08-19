/*
    header for test_common.
*/

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "../src/tensors/tensor.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct test_context_s {
    int assertions; // the number of tests run (not just functions, but individual tests)
    int failures;
} TestContext; // stores metadata about tests including

typedef void (*TestFunction)(TestContext* context);

#define TEST_EXPECT(context, condition)                                                            \
    test_expect((context), (condition), __FILE__, __LINE__, #condition)
// macro calls test_expect with the current line number file name and
// the condition (as a string)

void test_expect(TestContext* context, bool condition, const char* file, int line,
                 const char* expression);

void test_run(TestContext* context, const char* name, TestFunction function);
// runs the test and prints whether it passes or fails

int test_summary(const TestContext* context); // summarises tests

bool test_floats_close(float expected, float actual, float absolute_tolerance,
                       float relative_tolerance);
// checks that two floating poitn integers are close enough to each other
// given the absolute and relative tolerances

bool test_tensors_close(const Tensor* expected, const Tensor* actual, float absolute_tolerance,
                        float relative_tolerance);
// tests that two tensors are
// have the same rank, shape, and that the data
// values are close enough together
// owns_data and whether the data is malloc'd
// needn't be the same

bool test_tensors_deep_equal(const Tensor* expected, const Tensor* actual);
// same as above but relative and absolute tolerances == 0

#endif
