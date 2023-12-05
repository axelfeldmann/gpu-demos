# Talk/Demos: GPUs and Kernel Fusion

## Overview

This repo is written to accompany a talk on GPUs and kernel fusion.
It contains 7 demos that taken together (ideally) illustrate the following:
* Basics of CUDA kernels
* Why kernel fusion is desirable yet non-trivial
* How to somewhat-easily do it using Triton

## Demos

### `1_matmul_basic.cu`

This file demonstrates the basics of CUDA programming. The goal is to multiply two
`N x N` matrices, `A` and `B`.

While the GPU ("device") is connected to the CPU ("host"), the two do not share memory.
Therefore, while we are able to construct `A_host` and `B_host` in normal C++ on the
CPU, they *are not* present in GPU memory until they are explicitly copied over.

So, after constructing `A_host` and `B_host`, we first allocate memory for them
(and their product, `C`) on the device using `cudaMalloc`.
Once `A_device`, `B_device`, and `C_device` have been allocated on the GPU,
we use `cudaMemcpy` to transfer over the actual values of `A` and `B`.
Note that this `cudaMemcpy` is very slow compared to most other GPU operations.

TODOS:
* how the CUDA grid works
* threads
* output stationarity

### `2_matmul_sharedmem.cu`

TODOS:
* tiling
* what shared memory is

### `3_matmul_tensorcore.cu`

This is a counterexample-- for certain precisions (fp16, int8, on newer GPUs even fp64),
there are *tensor cores* that provide high throughput matrix multiplications.

However, they are really hard to use! This is my attempt at using them, and it doesn't go super well.
I will dedicate more time to this, but the point is that writing peak performance dense matrix
multiplications is not totally trivial.

### `4_matmul_cublas.cu`

But calling libraries is! cuBLAS achieves basically peak throughput with very little code.

### `5_gating_pytorch.py`

TODOS:
* explain why we want kernel fusion
* kernel fusion is hard because cuBLAS provides a full kernel! Not just a call within a kernel.

### `6_gating_fused.py`

TODOS:
* explain Triton

### `bonus_matmul_A_stationary.cu`

### `bonus_benchmark_fused.py`

