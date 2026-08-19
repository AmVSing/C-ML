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

MATMUL_PERF_TESTS := $(BUILD_DIR)/matmul_perf_tests$(EXE_EXT)
MATMUL_PERF_TESTS_SOURCES := test/tensors/matmul_perf_tests.c src/tensors/tensor.c
MATMUL_PERF_TESTS_HEADERS := src/tensors/tensor.h

TENSOR_INIT_TESTS := $(BUILD_DIR)/tensor_init_tests$(EXE_EXT)
TENSOR_INIT_TESTS_SOURCES := test/tensors/tensor_init_tests.c test/test_common.c src/tensors/tensor.c
TENSOR_INIT_TESTS_HEADERS := test/test_common.h src/tensors/tensor.h

TENSOR_OPS_TESTS := $(BUILD_DIR)/tensor_ops_tests$(EXE_EXT)
TENSOR_OPS_TESTS_SOURCES := test/tensors/tensor_ops_tests.c test/test_common.c src/tensors/tensor.c
TENSOR_OPS_TESTS_HEADERS := test/test_common.h src/tensors/tensor.h

.PHONY: all matmul_perf_tests tensor_init_tests tensor_ops_tests

all: matmul_perf_tests tensor_init_tests tensor_ops_tests

matmul_perf_tests: $(MATMUL_PERF_TESTS)

tensor_init_tests: $(TENSOR_INIT_TESTS)

tensor_ops_tests: $(TENSOR_OPS_TESTS)

$(MATMUL_PERF_TESTS): $(MATMUL_PERF_TESTS_SOURCES) $(MATMUL_PERF_TESTS_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MATMUL_PERF_TESTS_SOURCES) $(LDFLAGS) -o $@ $(LDLIBS)

$(TENSOR_INIT_TESTS): $(TENSOR_INIT_TESTS_SOURCES) $(TENSOR_INIT_TESTS_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(TENSOR_INIT_TESTS_SOURCES) $(LDFLAGS) -o $@ $(LDLIBS)

$(TENSOR_OPS_TESTS): $(TENSOR_OPS_TESTS_SOURCES) $(TENSOR_OPS_TESTS_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(TENSOR_OPS_TESTS_SOURCES) $(LDFLAGS) -o $@ $(LDLIBS)

$(BUILD_DIR):
	mkdir $(BUILD_DIR)
