CC := gcc

.DEFAULT_GOAL := all

CPPFLAGS := -Isrc/tensors
CFLAGS := -std=c11 -Wall -Werror -Wpedantic -g -fsanitize=address,undefined
LDFLAGS := -fsanitize=address,undefined
LDLIBS := -lm

BUILD_DIR := build

ifeq ($(OS),Windows_NT)
EXE_EXT := .exe
else
EXE_EXT :=
endif

MATMUL_PERF_TEST := $(BUILD_DIR)/matmul_perf_test$(EXE_EXT)
MATMUL_PERF_TEST_SOURCES := test/tensors/matmul_perf_tests.c src/tensors/tensor.c
MATMUL_PERF_TEST_HEADERS := src/tensors/tensor.h

TENSOR_INIT_TESTS := $(BUILD_DIR)/tensor_init_tests$(EXE_EXT)
TENSOR_INIT_TESTS_SOURCES := test/tensors/tensor_init_tests.c test/test_common.c src/tensors/tensor.c
TENSOR_INIT_TESTS_HEADERS := test/test_common.h src/tensors/tensor.h

.PHONY: all matmul_perf_test tensor_init_tests

all: matmul_perf_test tensor_init_tests

matmul_perf_test: $(MATMUL_PERF_TEST)

tensor_init_tests: $(TENSOR_INIT_TESTS)

$(MATMUL_PERF_TEST): $(MATMUL_PERF_TEST_SOURCES) $(MATMUL_PERF_TEST_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MATMUL_PERF_TEST_SOURCES) $(LDFLAGS) -o $@ $(LDLIBS)

$(TENSOR_INIT_TESTS): $(TENSOR_INIT_TESTS_SOURCES) $(TENSOR_INIT_TESTS_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(TENSOR_INIT_TESTS_SOURCES) $(LDFLAGS) -o $@ $(LDLIBS)

$(BUILD_DIR):
	mkdir $(BUILD_DIR)
