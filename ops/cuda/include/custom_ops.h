#pragma once

namespace CustomCUDAOps {

#if defined(USE_CUDA) // Keep guard consistent, removed !ENABLE_TRAINING unless specifically needed
// Function to register all CUDA custom Ops for a given domain
void RegisterOps(Ort::CustomOpDomain& domain);

#else
// Define an empty function when CUDA is not enabled to avoid linker errors
inline void RegisterOps(Ort::CustomOpDomain& /*domain*/) {} // Use inline for empty definition in header
#endif // USE_CUDA

}  // namespace CustomCUDAOps