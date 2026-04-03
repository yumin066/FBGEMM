NVCC ?= nvcc

ROOT := ..

ARCH ?= sm_120a
NVCCFLAGS := -std=c++17 -g -G -O0
DEFINES_COMMON := -D__CUDA_ARCH_FEAT_SM120_ALL

# ---- tma_copy_b (B-operand TMA unit test) ----
SRC_B := tma_copy_b_sm120_blockscaled_unit_test.cu
BIN_B := tma_copy_b_sm120_blockscaled_unit_test_dbg_pf0_stage_bar

UNIT_TEST_B_K ?= 512
UNIT_TEST_ENABLE_TMA_PREFETCH ?= 0
UNIT_TEST_ENABLE_EMPTY_WAIT ?= 0

DEFINES_B := \
	-DUNIT_TEST_B_K=$(UNIT_TEST_B_K) \
	-DUNIT_TEST_ENABLE_TMA_PREFETCH=$(UNIT_TEST_ENABLE_TMA_PREFETCH) \
	-DUNIT_TEST_ENABLE_EMPTY_WAIT=$(UNIT_TEST_ENABLE_EMPTY_WAIT) \
	$(DEFINES_COMMON)
INCLUDES_B := \
	-I$(ROOT)/6KD_fp8_block_scale/3rdparty/cutlass/include \
	-I$(ROOT)/6KD_fp8_block_scale/kernels/include

# ---- tma_copy_k (K WS TMA unit test) ----
SRC_K := tma_copy_k_sm120_ws_unit_test.cu
BIN_K := tma_copy_k_sm120_ws_unit_test

DEFINES_K := $(DEFINES_COMMON)
INCLUDES_K := \
	-I$(ROOT)/external/cutlass/include \
	-I$(ROOT)/6KD_fp8_block_scale/3rdparty/cutlass/include

# ---- tma_copy_vt (Vt WS TMA unit test) ----
SRC_VT := tma_copy_vt_sm120_ws_unit_test.cu
BIN_VT := tma_copy_vt_sm120_ws_unit_test

DEFINES_VT := $(DEFINES_COMMON)
INCLUDES_VT := \
	-I$(ROOT)/external/cutlass/include \
	-I$(ROOT)/6KD_fp8_block_scale/3rdparty/cutlass/include

.PHONY: all build build-b build-k build-vt run run-b run-k run-vt sant sant-b sant-k sant-vt clean

all: build

build: build-b build-k build-vt

build-b:
	$(NVCC) $(NVCCFLAGS) $(DEFINES_B) $(INCLUDES_B) $(SRC_B) -o $(BIN_B) -arch=$(ARCH)

build-k:
	$(NVCC) $(NVCCFLAGS) $(DEFINES_K) $(INCLUDES_K) $(SRC_K) -o $(BIN_K) -arch=$(ARCH)

build-vt:
	$(NVCC) $(NVCCFLAGS) $(DEFINES_VT) $(INCLUDES_VT) $(SRC_VT) -o $(BIN_VT) -arch=$(ARCH)

run: run-b run-k run-vt

run-b: build-b
	./$(BIN_B)

run-k: build-k
	./$(BIN_K)

run-vt: build-vt
	./$(BIN_VT)

sant: sant-b sant-k sant-vt

sant-b: build-b
	compute-sanitizer --tool memcheck ./$(BIN_B)

sant-k: build-k
	compute-sanitizer --tool memcheck ./$(BIN_K)

sant-vt: build-vt
	compute-sanitizer --tool memcheck ./$(BIN_VT)

clean:
	rm -f $(BIN_B) $(BIN_K) $(BIN_VT)
