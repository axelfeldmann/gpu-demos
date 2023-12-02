#include <cstdio>
#include "utils.h"
#include <vector>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Might as well do this on the CPU if we're already randomly generating the data
// on the CPU
std::vector<half> float_to_half(const std::vector<float>& input) {
    std::vector<half> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = __float2half(input[i]);
    }
    return output;
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("Usage: %s <N> \n", argv[0]);
        return 1;
    }

    size_t N = atoi(argv[1]);

    auto A_host = random_vector<float>(N * N);
    auto B_host = random_vector<float>(N * N);

    auto A_host_half = float_to_half(A_host);
    auto B_host_half = float_to_half(B_host);

    half *A_device_half, *B_device_half;
    float *C_device_float;

    // Allocate too many arrays
    cudaMalloc(&A_device_half, N * N * sizeof(half));
    cudaMalloc(&B_device_half, N * N * sizeof(half));
    cudaMalloc(&C_device_float, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(A_device_half, A_host_half.data(), N * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device_half, B_host_half.data(), N * N * sizeof(half), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Time kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
                 A_device_half, CUDA_R_16F, N, 
                 B_device_half, CUDA_R_16F, N, &beta, 
                 C_device_float, CUDA_R_32F, N, 
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    size_t FLOPs = 2 * N * N * N;
    printf("Time: %f ms, GFLOPs: %f\n", milliseconds, FLOPs / milliseconds / 1e6);

    // Copy result back to host
    auto C_host = std::vector<float>(N * N);
    cudaMemcpy(C_host.data(), C_device_float, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_device_half);
    cudaFree(B_device_half);
    cudaFree(C_device_float);
}   