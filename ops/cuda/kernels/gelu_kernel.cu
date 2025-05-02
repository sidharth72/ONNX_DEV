#include <cuda_runtime.h>
#include <cmath> // For erfcf or tanhf depending on the GELU formula used

// Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
template <typename T>
__global__ void GeluApproximationKernel(const T* x, T* z, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        const T val = x[idx];
        const T kSqrt2OverPi = static_cast<T>(0.7978845608028654); // sqrt(2.0 / M_PI)
        const T kMagic = static_cast<T>(0.044715);
        const T kHalf = static_cast<T>(0.5);

        T x_cube = val * val * val;
        T inner = kSqrt2OverPi * (val + kMagic * x_cube);

        // Use tanhf for float, tanh for double if needed
        z[idx] = kHalf * val * (static_cast<T>(1.0) + tanhf(inner));
    }
}

// Host function to launch the Gelu kernel
// This function is callable from C++ code (cuda_ops.cc)
template <typename T>
void launch_gelu_kernel(int64_t N, T* z, const T* x, cudaStream_t stream) {
    if (N <= 0) return;

    // Simple 1D grid/block configuration
    // Adjust block size based on target architecture if needed, 256/512 are common starts
    const int threads_per_block = 256;
    // Calculate grid size ensuring all elements are covered
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch the chosen kernel (using the approximation here)
    GeluApproximationKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(x, z, N);

    // *** IMPORTANT: Add CUDA error checking in production code! ***
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    // }
    // Or use a macro wrapper for checking kernel launch and other CUDA calls.
}


template void launch_gelu_kernel<float>(int64_t N, float* z, const float* x, cudaStream_t stream);