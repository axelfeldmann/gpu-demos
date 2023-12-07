#include <cstdio>
#include "utils.h"
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>

__launch_bounds__(512) __global__ void half_precision_gemm(half* A, half* B, float* C, size_t N) {

    int global_block_row = blockIdx.y;
    int global_block_col = blockIdx.x;

    int local_warp_row = threadIdx.x / 32;
    int local_warp_col = threadIdx.y;

    int global_warp_row = global_block_row + local_warp_row;
    int global_warp_col = global_block_col + local_warp_col;

    int lane = threadIdx.x % 32;

    extern __shared__ half buffer[];
    half* A_tile = buffer;
    half* B_tile = buffer + (128 * 128);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag[2][2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag[2][2];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    for (int my_row = 0; my_row < 2; my_row++) {
        for (int my_col = 0; my_col < 2; my_col++) {
            nvcuda::wmma::fill_fragment(c_frag[my_row][my_col], 0.0f);
        }
    }

    for (int k = 0; k < N; k += 128) {

        for (int j = 0; j < 32; j++) {
            // load row of A tile
            int local_A_row = local_warp_row * 32 + j;
            int local_A_col = local_warp_col * 32 + lane;

            int global_A_row = global_block_row * 32 + j;
            int global_A_col = k + local_A_col;

            A_tile[local_A_row * 128 + local_A_col] = A[global_A_row * N + global_A_col];
            B_tile[local_A_row * 128 + local_A_col] = B[global_A_row * N + global_A_col];
        }
        __syncthreads();

        for (int my_row = 0; my_row < 2; my_row++) {
            for (int my_col = 0; my_col < 2; my_col++) {

                for (int kk = 0; kk < 8; kk++) {
                    int local_A_row = local_warp_row * 32 + my_row * 16;
                    int local_A_col = kk * 16;

                    int local_B_row = kk * 16;
                    int local_B_col = local_warp_col * 32 + my_col * 16;

                    nvcuda::wmma::load_matrix_sync(a_frag[my_row][my_col], &A_tile[local_A_row * 128 + local_A_col], 128);
                    nvcuda::wmma::load_matrix_sync(b_frag[my_row][my_col], &B_tile[local_B_row * 128 + local_B_col], 128);
                    nvcuda::wmma::mma_sync(c_frag[my_row][my_col], a_frag[my_row][my_col], b_frag[my_row][my_col], c_frag[my_row][my_col]);
                }
            }
        }
        __syncthreads();
    }

    for (int my_row = 0; my_row < 2; my_row++) {
        for (int my_col = 0; my_col < 2; my_col++) {
            int c_index = (global_warp_row * 32 + my_row * 16) * N + global_warp_col * 32 + my_col * 16;
            nvcuda::wmma::store_matrix_sync(&C[c_index], c_frag[my_row][my_col], N, nvcuda::wmma::mem_row_major);
        }
    }
}

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

    dim3 block_dim(128, 4);
    dim3 grid_dim(N / 128, N / 128);

    int sharedmem_size = (64 << 10);
    cudaFuncSetAttribute(half_precision_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedmem_size);

    // Time kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    half_precision_gemm<<<grid_dim, block_dim, sharedmem_size>>>(A_device_half, B_device_half, C_device_float, N);

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