#!/usr/bin/env python3

import torch
import triton
import triton.language as tl
from itertools import product
import pandas as pd

BENCHMARK_SIZES = [32, 64, 128, 256, 512, 1024]

sizes = [16, 32, 64]
configs = []
for m, c1, c2 in product(sizes, sizes, sizes):
    configs.append(
        triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_C1": c1, "BLOCK_SIZE_C2": c2})
    )

@triton.autotune(configs=configs, key=["M", "C2", "C1"])
@triton.jit
def fused_kernel(
    X_ptr,
    W1_ptr,
    W2_ptr,
    b1_ptr,
    b2_ptr,
    O_ptr,
    M,
    C1,
    C2,
    stride_Xm,
    stride_Xc1,
    stride_W1c1,
    stride_W1c2,
    stride_W2c1,
    stride_W2c2,
    stride_Om,
    stride_Oc2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C2: tl.constexpr,
    BLOCK_SIZE_C1: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c2 = tl.program_id(1)

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c2 = pid_c2 * BLOCK_SIZE_C2 + tl.arange(0, BLOCK_SIZE_C2)
    offs_c1 = tl.arange(0, BLOCK_SIZE_C1)

    x_ptrs = X_ptr + offs_xm[:, None] * stride_Xm + offs_c1[None, :] * stride_Xc1
    w1_ptrs = W1_ptr + offs_c1[:, None] * stride_W1c1 + offs_c2[None, :] * stride_W1c2
    w2_ptrs = W2_ptr + offs_c1[:, None] * stride_W2c1 + offs_c2[None, :] * stride_W2c2

    accum_w1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C2), dtype=tl.float32)
    accum_w2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C2), dtype=tl.float32)

    x_ptr_step = BLOCK_SIZE_C1 * stride_Xc1
    w1_ptr_step = BLOCK_SIZE_C1 * stride_W1c1
    w2_ptr_step = BLOCK_SIZE_C1 * stride_W2c1

    for _ in range(0, tl.cdiv(C1, BLOCK_SIZE_C1)):
        x = tl.load(x_ptrs)
        w1 = tl.load(w1_ptrs)
        w2 = tl.load(w2_ptrs)

        accum_w1 += tl.dot(x, w1, allow_tf32=False)
        accum_w2 += tl.dot(x, w2, allow_tf32=False)

        x_ptrs += x_ptr_step
        w1_ptrs += w1_ptr_step
        w2_ptrs += w2_ptr_step

    # Stuff with biases
    b1 = tl.load(b1_ptr + offs_c2)
    b2 = tl.load(b2_ptr + offs_c2)
    accum_w1 += b1[None, :]
    accum_w2 += b2[None, :]

    # Finalize output
    o = (tl.sigmoid(accum_w1) * accum_w2)
    o_ptrs = O_ptr + offs_xm[:, None] * stride_Om + offs_c2[None, :] * stride_Oc2
    tl.store(o_ptrs, o)

class gating(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, W1, W2, b1, b2):
        assert X.is_contiguous()
        assert W1.is_contiguous()
        assert W2.is_contiguous()
        assert b1.is_contiguous()
        assert b2.is_contiguous()

        M, C1 = X.shape
        C1, C2 = W1.shape
        O = torch.empty((M, C2), device=X.device, dtype=X.dtype)

        assert W2.shape == (C1, C2)
        assert O.is_contiguous()

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(C2, META["BLOCK_SIZE_C2"]),
        )
        fused_kernel[grid](
            X,
            W1,
            W2,
            b1,
            b2,
            O,
            M,
            C1,
            C2,
            X.stride(0),
            X.stride(1),
            W1.stride(0),
            W1.stride(1),
            W2.stride(0),
            W2.stride(1),
            O.stride(0),
            O.stride(1),
        )
        return O

def fused_fwd(x, W1, W2, b1, b2):
    result = gating.apply(x, W1, W2, b1, b2)
    return result

def unfused_fwd(x, W1, W2, b1, b2):
    result = (x @ W1) + b1
    result.sigmoid_()
    result *= (x @ W2) + b2
    return result

@torch.jit.script
def unfused_fwd_jit(x, W1, W2, b1, b2):
    result = (x @ W1) + b1
    result.sigmoid_()
    result *= (x @ W2) + b2
    return result

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=BENCHMARK_SIZES,
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "triton",
            "torch-jit",
            "torch",
        ],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch-jit", "Torch"],  # Label name for the lines.
        plot_name="performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark_fwd(size, provider):
    B = 4
    C = 128
    H = 32

    x = torch.randn((B * size * size, C), device="cuda", dtype=torch.float16, requires_grad=False)
    W1 = torch.randn((C, 2 * H), device="cuda", dtype=torch.float16, requires_grad=False)
    W2 = torch.randn((C, 2 * H), device="cuda", dtype=torch.float16, requires_grad=False)
    b1 = torch.randn((2 * H,), device="cuda", dtype=torch.float16, requires_grad=False)
    b2 = torch.randn((2 * H,), device="cuda", dtype=torch.float16, requires_grad=False)

    if provider == "triton":
        ms = triton.testing.do_bench(lambda: fused_fwd(x, W1, W2, b1, b2))
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: unfused_fwd(x, W1, W2, b1, b2))
    if provider == "torch-jit":
        ms = triton.testing.do_bench(lambda: unfused_fwd_jit(x, W1, W2, b1, b2))
    return ms

def memory(f, *args):
    for i in range(10):
        torch.cuda.empty_cache()
        curr_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"] / (
            1024**3
        )

        f(*args)

        torch.cuda.synchronize()
        peak_bytes = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / (1024**3)
        if i == 9:
            peak_memory = peak_bytes - curr_bytes

        torch.cuda.reset_peak_memory_stats()

    return peak_memory

def benchmark_memory():

    B = 4
    C = 128
    H = 32

    data = []

    for size in BENCHMARK_SIZES:

        x = torch.randn((B * size * size, C), device="cuda", dtype=torch.float16, requires_grad=False)
        W1 = torch.randn((C, 2 * H), device="cuda", dtype=torch.float16, requires_grad=False)
        W2 = torch.randn((C, 2 * H), device="cuda", dtype=torch.float16, requires_grad=False)
        b1 = torch.randn((2 * H,), device="cuda", dtype=torch.float16, requires_grad=False)
        b2 = torch.randn((2 * H,), device="cuda", dtype=torch.float16, requires_grad=False)

        memory_fused = memory(fused_fwd, x, W1, W2, b1, b2)
        memory_unfused = memory(unfused_fwd, x, W1, W2, b1, b2)
        memory_unfused_jit = memory(unfused_fwd_jit, x, W1, W2, b1, b2)

        data.append({
            "size": size,
            "fused_GB": memory_fused,
            "unfused_GB": memory_unfused,
            "unfused_jit_GB": memory_unfused_jit
        })

    df = pd.DataFrame(data)
    print("Memory Utilization")
    print(df)

if __name__ == "__main__":
    # Setup
    torch.manual_seed(0)

    # Compute performance
    torch.set_grad_enabled(False)
    print("Fwd performance")
    benchmark_fwd.run(print_data=True, show_plots=False)

    print()
    benchmark_memory()




