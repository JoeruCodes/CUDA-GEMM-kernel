# CUDA-GEMM-kernel
My attempt of making of a GEMM kernel...

 ## Tiled Matrix Multiplication: 
    The code uses a tiled matrix multiplication algorithm, which improves data locality and reduces global memory accesses. It divides the input matrices into smaller tiles and performs matrix multiplication on each tile, utilizing shared memory to cache data.

 ## Shared Memory: 
    The code declares two shared memory arrays, s_a and s_b, which are used to cache tiles of matrices a and b, respectively. The use of shared memory reduces global memory accesses and enables faster access to the input data.

 ## Loop Unrolling: 
    The innermost loop that performs the matrix multiplication has been unrolled using #pragma unroll. Loop unrolling reduces loop overhead and enables the compiler to generate more efficient code by exploiting instruction-level parallelism.

 ## Asynchronous Memory Copies: 
    The code uses cudaMemcpyAsync to perform asynchronous memory copies between the host and the device. This allows overlapping of memory transfers with kernel execution, reducing the overall execution time.

 ## Multiple CUDA Streams: 
     The code creates two CUDA streams, stream1 and stream2, to overlap the execution of kernel invocations and memory copies. By using multiple streams, the program can perform computations and data transfers concurrently, improving overall throughput.

 ## Verification of Results: 
    The verify_result function checks the correctness of the computed matrix multiplication results by comparing them with a sequentially computed reference solution. This verification ensures the accuracy of the parallel computation.

 ## Performance Measurement: 
    The code measures the execution time of the matrix multiplication kernel using std::chrono library and calculates the achieved GFLOPs (Giga Floating-Point Operations per second) based on the elapsed time and the problem size.
