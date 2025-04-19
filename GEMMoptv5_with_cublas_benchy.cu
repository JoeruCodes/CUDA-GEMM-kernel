#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda;

#define CUBLAS_CHECK(call)                                               \
  do {                                                                   \
    cublasStatus_t status = call;                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      fprintf(stderr, "cuBLAS error: %d at %s:%d\n", status, __FILE__,   \
              __LINE__);                                                 \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

constexpr int M = 1 << 10;
constexpr int N = 1 << 11;
constexpr int K = 1 << 12;

constexpr int TILE_M = 64;
constexpr int TILE_N = 128;
constexpr int TILE_K = 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_THREADS = 128;
constexpr int NUM_ITERATIONS = 10;

__global__ void convertFloatToHalfKernel(const float* input, __half* output, int num_elements);
__global__ void wmmaGemmKernel(const __half* __restrict__ a, const __half* __restrict__ b, float* __restrict__ c_float_accum, __half* d_c_ptr_unused, int M_param, int N_param, int K_param);

void verify_result(const std::vector<__half>& a, const std::vector<__half>& b, const std::vector<__half>& c, const std::string& label) {
  std::cout << "Verifying results for " << label << "..." << std::endl;
  double max_error = 0.0;
  int error_count = 0;
  float tolerance = 1e-1;

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i) {
        if (row * K + i < a.size() && i * N + col < b.size()) {
            tmp += static_cast<float>(a[row * K + i]) * static_cast<float>(b[i * N + col]);
        }
      }

      float c_val = 0.0f;
      if (row * N + col < c.size()) {
         c_val = static_cast<float>(c[row * N + col]);
      }

      float diff = abs(tmp - c_val);
      bool error_condition = (tmp == 0.0f) ? (diff > tolerance) : (diff / abs(tmp) > tolerance);

      if (error_condition && error_count < 10) {
        std::cerr << "Verification failed at (" << row << ", " << col << ") [" << label << "]: expected " << tmp << ", got " << c_val << ", diff " << diff << std::endl;
        error_count++;
      }
      if (diff > max_error) {
        max_error = diff;
      }
    }
  }
  if (error_count > 0) {
    std::cerr << "Verification encountered " << error_count << " errors for " << label << "." << std::endl;
    std::cerr << "Maximum error [" << label << "]: " << max_error << std::endl;
  } else {
    std::cout << "Verification successful for " << label << ". Max error: " << max_error << std::endl;
  }
}

int main() {
  size_t bytes_a = static_cast<size_t>(M) * K * sizeof(__half);
  size_t bytes_b = static_cast<size_t>(K) * N * sizeof(__half);
  size_t bytes_c = static_cast<size_t>(M) * N * sizeof(__half);
  size_t bytes_c_accum = static_cast<size_t>(M) * N * sizeof(float);

  std::vector<__half> h_a(static_cast<size_t>(M) * K);
  std::vector<__half> h_b(static_cast<size_t>(K) * N);
  std::vector<__half> h_c(static_cast<size_t>(M) * N);
  std::vector<__half> h_c_cublas(static_cast<size_t>(M) * N);

  std::srand(std::time(nullptr));
  std::generate(h_a.begin(), h_a.end(), []() { return static_cast<__half>(static_cast<float>(std::rand() % 5) - 2.0f); });
  std::generate(h_b.begin(), h_b.end(), []() { return static_cast<__half>(static_cast<float>(std::rand() % 5) - 2.0f); });

  __half* d_a, *d_b, *d_c;
  float* d_c_accum;
  cudaMallocManaged(&d_a, bytes_a);
  cudaMallocManaged(&d_b, bytes_b);
  cudaMallocManaged(&d_c, bytes_c);
  cudaMallocManaged(&d_c_accum, bytes_c_accum);

  cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

  dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  dim3 threads(BLOCK_THREADS);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "\n--- Running WMMA Kernel (" << NUM_ITERATIONS << " iterations) ---" << std::endl;
  std::cout << "Grid: (" << blocks.x << ", " << blocks.y << "), Block: (" << threads.x << ")" << std::endl;
  std::vector<float> times_wmma;
  times_wmma.reserve(NUM_ITERATIONS);

  cudaMemset(d_c_accum, 0, bytes_c_accum);
  wmmaGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c_accum, d_c, M, N, K);
  cudaDeviceSynchronize();

  for (int i = 0; i < NUM_ITERATIONS; ++i) {
      cudaMemset(d_c_accum, 0, bytes_c_accum); 
      cudaDeviceSynchronize();

      cudaEventRecord(start);
      wmmaGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c_accum, d_c, M, N, K);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float ms = 0;
      cudaEventElapsedTime(&ms, start, stop);
      times_wmma.push_back(ms);
  }

  float total_time_wmma = std::accumulate(times_wmma.begin(), times_wmma.end(), 0.0f);
  float avg_time_wmma = total_time_wmma / NUM_ITERATIONS;
  double avg_gflops_wmma = (2.0 * M * N * K * 1e-9) / (avg_time_wmma * 1e-3);
  std::cout << "WMMA Kernel Avg Execution Time: " << avg_time_wmma << " ms" << std::endl;
  std::cout << "WMMA Kernel Avg GFLOPs: " << avg_gflops_wmma << std::endl;

  std::cout << "\n--- Running cuBLAS Kernel (" << NUM_ITERATIONS << " iterations) ---" << std::endl;
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  std::vector<float> times_cublas;
  times_cublas.reserve(NUM_ITERATIONS);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaMemset(d_c_accum, 0, bytes_c_accum);
  CUBLAS_CHECK(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                            &alpha, d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta,
                            d_c_accum, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  cudaDeviceSynchronize();

  for (int i = 0; i < NUM_ITERATIONS; ++i) {
      cudaMemset(d_c_accum, 0, bytes_c_accum);
      cudaDeviceSynchronize();

      cudaEventRecord(start);
      CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                &alpha,
                                d_b, CUDA_R_16F, N,
                                d_a, CUDA_R_16F, K,
                                &beta,
                                d_c_accum, CUDA_R_32F, N,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float ms = 0;
      cudaEventElapsedTime(&ms, start, stop);
      times_cublas.push_back(ms);
  }

  float total_time_cublas = std::accumulate(times_cublas.begin(), times_cublas.end(), 0.0f);
  float avg_time_cublas = total_time_cublas / NUM_ITERATIONS;
  double avg_gflops_cublas = (2.0 * M * N * K * 1e-9) / (avg_time_cublas * 1e-3);
  std::cout << "cuBLAS Kernel Avg Execution Time: " << avg_time_cublas << " ms" << std::endl;
  std::cout << "cuBLAS Kernel Avg GFLOPs: " << avg_gflops_cublas << std::endl;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_c_accum);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  CUBLAS_CHECK(cublasDestroy(cublas_handle));

  std::cout << "\nCOMPLETED SUCCESSFULLY" << std::endl;
  return 0;
}

__global__ void convertFloatToHalfKernel(const float* input, __half* output, int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = static_cast<__half>(input[idx]);
  }
}

__global__ void wmmaGemmKernel(const __half* __restrict__ a, const __half* __restrict__ b, float* __restrict__ c_float_accum, __half* d_c_ptr_unused, int M_param, int N_param, int K_param) {
  __shared__ __half s_a[TILE_M][TILE_K + 8];
  __shared__ __half s_b[TILE_K][TILE_N + 8];

  const unsigned int warpId = threadIdx.x / 32;

  const int block_tile_m_idx = blockIdx.y;
  const int block_tile_n_idx = blockIdx.x;
  const int block_start_row_c = block_tile_m_idx * TILE_M;
  const int block_start_col_c = block_tile_n_idx * TILE_N;

  const int warp_quadrant_m_offset = (warpId / 2) * (TILE_M / 2);
  const int warp_quadrant_n_offset = (warpId % 2) * (TILE_N / 2);

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2][2];
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[2];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag[2];

  const int num_k_tiles = (K_param + TILE_K - 1) / TILE_K;

  const int num_m_region = (TILE_M / 2) / (WMMA_M * 2);
  const int num_n_region = (TILE_N / 2) / (WMMA_N * 2);

  for (int m_reg_idx = 0; m_reg_idx < num_m_region; ++m_reg_idx) {
    for (int n_reg_idx = 0; n_reg_idx < num_n_region; ++n_reg_idx) {
      int current_region_row_in_quadrant = m_reg_idx * (WMMA_M * 2);
      int current_region_col_in_quadrant = n_reg_idx * (WMMA_N * 2);

      wmma::fill_fragment(acc_frag[0][0], 0.0f);
      wmma::fill_fragment(acc_frag[0][1], 0.0f);
      wmma::fill_fragment(acc_frag[1][0], 0.0f);
      wmma::fill_fragment(acc_frag[1][1], 0.0f);

      for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        const int a_tile_start_row = block_start_row_c;
        const int a_tile_start_col = k_tile_idx * TILE_K;

        for (int i = 0; i < (TILE_M * TILE_K) / BLOCK_THREADS; ++i) {
          int idx = threadIdx.x + i * BLOCK_THREADS;
          int row = idx / TILE_K;
          int col = idx % TILE_K;
          int global_row = a_tile_start_row + row;
          int global_col = a_tile_start_col + col;

          if (global_row < M_param && global_col < K_param) {
            s_a[row][col] = a[global_row * K_param + global_col];
          } else {
            s_a[row][col] = __float2half(0.0f);
          }
        }

        const int b_tile_start_row = k_tile_idx * TILE_K;
        const int b_tile_start_col = block_start_col_c;

        for (int i = 0; i < (TILE_K * TILE_N) / BLOCK_THREADS; ++i) {
          int idx = threadIdx.x + i * BLOCK_THREADS;
          int row_k = idx / TILE_N;
          int col_n = idx % TILE_N;
          int global_row = b_tile_start_row + row_k;
          int global_col = b_tile_start_col + col_n;

          if (global_row < K_param && global_col < N_param) {
            s_b[row_k][col_n] = b[global_row * N_param + global_col];
          } else {
            s_b[row_k][col_n] = __float2half(0.0f);
          }
        }

        __syncthreads();

        #pragma unroll
        for (int k_wmma_idx = 0; k_wmma_idx < (TILE_K / WMMA_K); ++k_wmma_idx) {
          int s_a_load_row0 = warp_quadrant_m_offset + current_region_row_in_quadrant;
          int s_a_load_row1 = s_a_load_row0 + WMMA_M;
          int s_a_load_col = k_wmma_idx * WMMA_K;

          int s_b_load_row0 = warp_quadrant_n_offset + current_region_col_in_quadrant;
          int s_b_load_row1 = s_b_load_row0 + WMMA_N;
          int s_b_load_col = k_wmma_idx * WMMA_K;

          wmma::load_matrix_sync(a_frag[0], &s_a[s_a_load_row0][s_a_load_col], TILE_K + 8);
          wmma::load_matrix_sync(a_frag[1], &s_a[s_a_load_row1][s_a_load_col], TILE_K + 8);
          wmma::load_matrix_sync(b_frag[0], &s_b[s_b_load_col][s_b_load_row0], TILE_N + 8);
          wmma::load_matrix_sync(b_frag[1], &s_b[s_b_load_col][s_b_load_row1], TILE_N + 8);

          wmma::mma_sync(acc_frag[0][0], a_frag[0], b_frag[0], acc_frag[0][0]);
          wmma::mma_sync(acc_frag[0][1], a_frag[0], b_frag[1], acc_frag[0][1]);
          wmma::mma_sync(acc_frag[1][0], a_frag[1], b_frag[0], acc_frag[1][0]);
          wmma::mma_sync(acc_frag[1][1], a_frag[1], b_frag[1], acc_frag[1][1]);
        }

        __syncthreads();
      }

      int c_store_base_row = block_start_row_c + warp_quadrant_m_offset + current_region_row_in_quadrant;
      int c_store_base_col = block_start_col_c + warp_quadrant_n_offset + current_region_col_in_quadrant;

      int c_store_row_0 = c_store_base_row;
      int c_store_col_0 = c_store_base_col;
      if (c_store_row_0 < M_param && c_store_col_0 < N_param) {
        wmma::store_matrix_sync(&c_float_accum[c_store_row_0 * N_param + c_store_col_0], acc_frag[0][0], N_param, wmma::mem_row_major);
      }

      int c_store_row_1 = c_store_base_row;
      int c_store_col_1 = c_store_base_col + WMMA_N;
      if (c_store_row_1 < M_param && c_store_col_1 < N_param) {
        wmma::store_matrix_sync(&c_float_accum[c_store_row_1 * N_param + c_store_col_1], acc_frag[0][1], N_param, wmma::mem_row_major);
      }

      int c_store_row_2 = c_store_base_row + WMMA_M;
      int c_store_col_2 = c_store_base_col;
      if (c_store_row_2 < M_param && c_store_col_2 < N_param) {
        wmma::store_matrix_sync(&c_float_accum[c_store_row_2 * N_param + c_store_col_2], acc_frag[1][0], N_param, wmma::mem_row_major);
      }

      int c_store_row_3 = c_store_base_row + WMMA_M;
      int c_store_col_3 = c_store_base_col + WMMA_N;
      if (c_store_row_3 < M_param && c_store_col_3 < N_param) {
        wmma::store_matrix_sync(&c_float_accum[c_store_row_3 * N_param + c_store_col_3], acc_frag[1][1], N_param, wmma::mem_row_major);
      }
    }
  }
}
