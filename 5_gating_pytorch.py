#!/usr/bin/env python3

import torch

B = 4
C = 128
H = 32
size = 1024
M = B * size * size

x = torch.randn((M, C), device="cuda", dtype=torch.float16, requires_grad=False)
W1 = torch.randn((C, 2 * H), device="cuda", dtype=torch.float16, requires_grad=False)
W2 = torch.randn((C, 2 * H), device="cuda", dtype=torch.float16, requires_grad=False)
b1 = torch.randn((2 * H,), device="cuda", dtype=torch.float16, requires_grad=False)
b2 = torch.randn((2 * H,), device="cuda", dtype=torch.float16, requires_grad=False)

def forward(x, W1, W2, b1, b2):
    result = (x @ W1) + b1
    result.sigmoid_()
    result *= (x @ W2) + b2
    return result

# Warm up
for _ in range(10):
    _ = forward(x, W1, W2, b1, b2)

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
res = forward(x, W1, W2, b1, b2)
end_event.record()
torch.cuda.synchronize()

time_elapsed = start_event.elapsed_time(end_event)

print(f"Time: {time_elapsed} ms")

# Calculating FLOPs - first order approximation, it's just the matmuls
flops = (x.shape[0] * x.shape[1] * W1.shape[1] * 2) + (x.shape[0] * x.shape[1] * W2.shape[1] * 2)
print(f"GFLOP/s: {flops / (time_elapsed * 1e6)}")
