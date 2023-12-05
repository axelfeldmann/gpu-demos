# Talk/Demos: GPUs and Kernel Fusion

## Overview

This repo is written to accompany a talk on GPUs and kernel fusion.
It contains 7 demos that taken together (ideally) illustrate the following:
* Basics of CUDA kernels
* Why kernel fusion is desirable yet non-trivial
* How to somewhat-easily do it using Triton

## Building/Running

* To build the CUDA examples, use SCons. This means insatalling Scons if you don't already have it, then just running `scons` inside this directory.
* You need to have a somewhat modern Nvidia GPU with all the CUDA drivers installed.
* The Python examples require PyTorch and Triton. Both can be installed with `pip` or equivalent stuff.
* Running the CUDA examples is as simple as `./1_matmul_basic <N>`. For simplicity, `N` must be a multiple of 32. I think 2048, 4096, 8192, and 16384
are nice numbers to try.

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

OK, so far we've seen a decent amount of CUDA. We've (hopefully) learned the following:
* It is posible to write your own CUDA kernels and get good but not optimal performance (without a ton of tuning)
* Or, you can call Nvidia provided libraries and get near-optimal performance out of the box

A reasonable question is: "what's the point of writing your own kernels if the Nvidia provided ones are so good?"
Is it only for things that are actually un-supported by the libraries?
To answer these questions, let's look at `5_gating_pytorch.py`.

Here, we are computing the following function:

```
def forward(x, W1, W2, b1, b2):
    result = (x @ W1) + b1
    result.sigmoid_()
    result *= (x @ W2) + b2
    return result
```

PyTorch will dispatch all of the functions in here to carefully optimized Nvidia libraries.
However, the overall performance is not great! On my V100, this gets just 11 TFLOP/s (fp16) 
out of a possible 112 TFLOP/s (fp16). What gives?

The problem is that this single function calls *multiple kernels*. I haven't looked at exactly what
kernels are launched, but it may be something like this:
```
t0 = gemm(x, W1, b1)
t1 = add(t0, b1)
t2 = sigmoid(t1)
t3 = gemm(x, W2)
t4 = add(t3, b2)
t5 = mul(t2, t4)
```

This is not great! Even if there were no overheads to launching kernels, this implementation
causes a ton of *excess data movement*. Think back to `2_matmul_sharedmem.cu`: 
in that kernel, we go through the trouble of loading various tiles of the matrix from main memory into shared memory,
multiplying them, then at the end, storing our final result back to main memory.
Why don't we just add the `b1` and apply the sigmoid to our output tile *before* storing it back to shared memory?
Why don't we just load `x` *once* and multiply it by both `W1` *and* `W2`?

The problem is the kernel abstraction: Nvidia provides libraries that "do a thing", but they provide them as a
full kernel that you call into. If you call `cublasGemmEx`, you *cannot* say "oh while you're doing that, also apply a sigmoid,"
because it's just operating at *a different level of abstraction*.

What we really want to do here is called *kernel fusion*. By designing a kernel to do the entire `forward`, function,
we are able to reduce the number of kernel launches, and more importantly, *the total data movement*. 
And that's just what we'll do in the next example.

### `6_gating_fused.py`

TODOS:
* explain Triton

### `bonus_matmul_A_stationary.cu`

### `bonus_benchmark_fused.py`

