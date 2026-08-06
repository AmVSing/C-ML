CC := gcc

CPPFLAGS := -Isrc/tensors
CFLAGS := -std=c11 -Wall -Werror -Wpedantic -g -fsanitize=address,undefined
LDFLAGS := -fsanitize=address,undefined

BUILD_DIR := build

ifeq ($(OS),Windows_NT)
EXE_EXT := .exe
else
EXE_EXT :=
endif

MATMUL_PERF_TEST := $(BUILD_DIR)/matmul_perf_test$(EXE_EXT)
MATMUL_PERF_TEST_SOURCES := test/tensors/matmul_perf_test.c src/tensors/tensor.c
MATMUL_PERF_TEST_HEADERS := src/tensors/tensor.h

.PHONY: matmul_perf_test

matmul_perf_test: $(MATMUL_PERF_TEST)

$(MATMUL_PERF_TEST): $(MATMUL_PERF_TEST_SOURCES) $(MATMUL_PERF_TEST_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MATMUL_PERF_TEST_SOURCES) $(LDFLAGS) -o $@

$(BUILD_DIR):
	mkdir $(BUILD_DIR)
