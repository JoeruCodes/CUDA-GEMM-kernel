#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr int M = 1 << 10;
constexpr int N = 1 << 11;
constexpr int K = 1 << 12;
constexpr int TILE_SIZE = 32;

__global__ void matrixMul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float s_a[TILE_SIZE][TILE_SIZE];
  __shared__ float s_b[TILE_SIZE][TILE_SIZE];

  float tmp = 0.0f;

  for (int t = 0; t < K / TILE_SIZE; ++t) {
    s_a[threadIdx.y][threadIdx.x] = a[row * K + t * TILE_SIZE + threadIdx.x];
    s_b[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * N + col];

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      tmp += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
    }

    __syncthreads();
  }

  c[row * N + col] = tmp;
}

void verify_result(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c) {
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i) {
        tmp += a[row * K + i] * b[i * N + col];
      }

      assert(tmp == c[row * N + col]);
    }
  }
}

int main() {
  size_t bytes_a = M * K * sizeof(float);
  size_t bytes_b = K * N * sizeof(float);
  size_t bytes_c = M * N * sizeof(float);

  std::vector<float> h_a(M * K);
  std::vector<float> h_b(K * N);
  std::vector<float> h_c(M * N);

  std::generate(h_a.begin(), h_a.end(), []() { return static_cast<float>(std::rand() % 100); });
  std::generate(h_b.begin(), h_b.end(), []() { return static_cast<float>(std::rand() % 100); });

  float* d_a, *d_b, *d_c;
  cudaMallocManaged(&d_a, bytes_a);
  cudaMallocManaged(&d_b, bytes_b);
  cudaMallocManaged(&d_c, bytes_c);

  cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

  int THREADS = 32;
  int BLOCKS_X = N / THREADS;
  int BLOCKS_Y = M / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_X, BLOCKS_Y);

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  matrixMul<<<blocks, threads, 0, stream1>>>(d_a, d_b, d_c);
  cudaMemcpyAsync(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost, stream1);

  matrixMul<<<blocks, threads, 0, stream2>>>(d_a, d_b, d_c);
  cudaMemcpyAsync(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost, stream2);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  verify_result(h_a, h_b, h_c);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Calculate GFLOPs
  auto start = std::chrono::steady_clock::now();

  matrixMul<<<blocks, threads, 0, stream1>>>(d_a, d_b, d_c);
  cudaStreamSynchronize(stream1);

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double gflops = (2.0 * M * N * K) / (duration.count() * 1e3);

  std::cout << "GFLOPs: " << gflops << std::endl;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  return 0;
}
