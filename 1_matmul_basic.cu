#include <cstdio>
#include "utils.h"
#include <vector>

constexpr size_t BLOCK_DIM = 32;

__global__ void matrix_multiply(float* A, float* B, float* C, size_t N) {

    // Calculate row and column of C to work on
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate C[row][col]
    float accum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        accum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = accum;
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("Usage: %s <N> \n", argv[0]);
        return 1;
    }

    size_t N = atoi(argv[1]);

    // This is just a demo, so enforce this for simplicity
    if (N % BLOCK_DIM != 0) {
        printf("N must be a multiple of %zu\n", BLOCK_DIM);
        return 1;
    }

    auto A_host = random_vector<float>(N * N);
    auto B_host = random_vector<float>(N * N);

    // Allocate memory on GPU
    float *A_device, *B_device, *C_device;
    cudaMalloc(&A_device, N * N * sizeof(float));
    cudaMalloc(&B_device, N * N * sizeof(float));
    cudaMalloc(&C_device, N * N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(A_device, A_host.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim(N / BLOCK_DIM, N / BLOCK_DIM);

    printf("(%d x %d) grid of (%d x %d) blocks\n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);

    // Time kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrix_multiply<<<grid_dim, block_dim>>>(A_device, B_device, C_device, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    size_t FLOPs = 2 * N * N * N;
    printf("Time: %f ms, GFLOPs: %f\n", milliseconds, FLOPs / milliseconds / 1e6);

    // Copy result back to host
    auto C_host = std::vector<float>(N * N);
    cudaMemcpy(C_host.data(), C_device, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}   