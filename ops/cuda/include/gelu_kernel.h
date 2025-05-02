#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
void RunGeluCuda(const float* in, float* out, int64_t N, cudaStream_t stream);
#ifdef __cplusplus
}
#endif