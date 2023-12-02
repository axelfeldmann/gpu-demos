#include <cstdio>
#include "utils.h"
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>

constexpr uint32_t WARPSIZE = 32;
constexpr uint32_t WMMA_DIM = 16;
constexpr uint32_t WMMA_TILE_DIM = 4;
constexpr uint32_t TILE_DIM = WMMA_DIM * WMMA_TILE_DIM;

// Perform matrix multiplication C = A * B using the tensor cores
// This is not a good implementation!
__global__ void half_precision_gemm(half* A, half* B, float* C, size_t N) {

    // Warp x and warp y describe the position of the warp's C tile in C
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_x = x / WARPSIZE;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t warp_y = y;
    uint32_t lane = threadIdx.x % WARPSIZE;

    // Local warp x and local warp y describe the position of the thread's C fragment
    // in the warp's C tile
    uint32_t local_warp_x = threadIdx.x / WARPSIZE;
    uint32_t local_warp_y = threadIdx.y;

    __shared__ half A_tile[TILE_DIM * TILE_DIM];
    __shared__ half B_tile[TILE_DIM * TILE_DIM];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_DIM, WMMA_DIM, WMMA_DIM, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_DIM, WMMA_DIM, WMMA_DIM, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_DIM, WMMA_DIM, WMMA_DIM, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    uint32_t load_col = local_warp_x / 2;
    uint32_t load_row = 2 * local_warp_y + local_warp_x % 2;

    for (uint32_t k = 0; k < N; k += TILE_DIM) {
        
        // Each thread is responsible for helping load 8 elements of A and B
        // into its warp's shared memory tile
        for (uint32_t i = 0; i < 8; i++) {
            // Load A[warp_x + load_row * 8 + i][k + load_col * WARP_SIZE + lane] into
            // A_tile[load_row * 8 + i][load_col * WARP_SIZE + lane]
            uint32_t a_row = warp_x + load_row * 8 + i;
            uint32_t a_col = k + load_col * WARPSIZE + lane;
            uint32_t a_index = a_row * N + a_col;

            uint32_t a_tile_row = load_row * 8 + i;
            uint32_t a_tile_col = load_col * WARPSIZE + lane;
            uint32_t a_tile_index = a_tile_row * TILE_DIM + a_tile_col;

            A_tile[a_tile_index] = A[a_index];

            // Load B[k + load_row * 8 + i][warp_y + load_col * WARP_SIZE + lane] into
            // B_tile[load_row * 8 + i][load_col * WARP_SIZE + lane]
            uint32_t b_row = k + load_row * 8 + i;
            uint32_t b_col = warp_y + load_col * WARPSIZE + lane;
            uint32_t b_index = b_row * N + b_col;

            uint32_t b_tile_row = load_row * 8 + i;
            uint32_t b_tile_col = load_col * WARPSIZE + lane;
            uint32_t b_tile_index = b_tile_row * TILE_DIM + b_tile_col;

            B_tile[b_tile_index] = B[b_index];
        }

        __syncthreads();

        // Load fragments from shared memory
        for (uint32_t kk = 0; kk < WMMA_TILE_DIM; kk++) {

            uint32_t A_tile_index = local_warp_y * TILE_DIM * WMMA_DIM + WMMA_DIM * kk;
            uint32_t B_tile_index = kk * TILE_DIM * WMMA_DIM + WMMA_DIM * local_warp_x;
            
            nvcuda::wmma::load_matrix_sync(a_frag, &A_tile[A_tile_index], TILE_DIM);
            nvcuda::wmma::load_matrix_sync(b_frag, &B_tile[B_tile_index], TILE_DIM);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    uint32_t c_index = warp_y * (WMMA_DIM * N) + warp_x * WMMA_DIM;
    nvcuda::wmma::store_matrix_sync(&C[c_index], c_frag, N, nvcuda::wmma::mem_row_major);
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

    dim3 block_dim(4 * WARPSIZE, 4); // Each block has 4x4 warps
    dim3 grid_dim(N / (4 * WMMA_DIM), N / (4 * WMMA_DIM)); // Each block has 4x4 warps

    // Time kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    half_precision_gemm<<<grid_dim, block_dim>>>(A_device_half, B_device_half, C_device_float, N);

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