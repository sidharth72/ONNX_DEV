// custom_ops.cc (or cuda_ops.cc)

// Use the build flag defined in your build system (e.g., USE_CUDA)
#if defined(USE_CUDA)

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "core/providers/cuda/cuda_context.h" // Provides Ort::Custom::CudaContext
#include "onnxruntime_lite_custom_op.h" // Provides OrtLiteCustomOp and creation functions

#include <cuda_runtime.h> // For cudaStream_t, cudaSuccess, cudaGetErrorString, cudaGetLastError
#include <memory>         // For std::unique_ptr
#include <stdexcept>      // For std::runtime_error
#include <string>         // For std::string
#include <vector>         // For Ort::Custom::Tensor::Shape() -> std::vector
#include <cstdio>         // For snprintf if needed, but std::string is safer

// --- Helper Macro (Revised for safer string handling) ---
#define CUSTOM_ENFORCE(cond, ...) \
  if (!(cond)) { \
    /* Use std::string for formatting to avoid buffer overflows easily */ \
    std::string error_msg = "CustomOp Error: "; \
    /* You might need a helper function for complex formatting, */ \
    /* but for simple concatenation, this works. */ \
    /* For the CUDA error case: */ \
    error_msg += std::string(__VA_ARGS__); \
    throw std::runtime_error(error_msg); \
  }
// Note: The above macro assumes __VA_ARGS__ resolves to something convertible to std::string
// or a const char*. A more robust macro might use snprintf or dedicated logging.
// Let's try a slightly different approach directly using string concatenation in the call site.

// --- Revised Helper Macro ---
#define CUSTOM_ENFORCE(condition, message) \
  if (!(condition)) { \
    throw std::runtime_error(std::string("CustomOp Error: ") + (message)); \
  }


// --- Declaration of the CUDA kernel launcher ---
// Defined in gelu_impl.cu or similar .cu file
// Declared in the GLOBAL namespace
template <typename T>
void launch_gelu_kernel(int64_t N, T* z, const T* x, cudaStream_t stream);

// --- Explicit Declaration of External Template Instantiations ---
// Also in the GLOBAL namespace, matching the template function declaration
extern template void launch_gelu_kernel<float>(int64_t, float*, const float*, cudaStream_t);
// extern template void launch_gelu_kernel<__half>(int64_t, __half*, const __half*, cudaStream_t); // If supporting half


// --- Namespace for CUDA Ops (using the name from your error messages) ---
namespace CustomCUDAOps { // <-- Use the namespace from your error messages

// --- Kernel Wrapper as a Free Function Template ---
template <typename T>
void KernelGelu(const Ort::Custom::CudaContext& cuda_ctx,
                const Ort::Custom::Tensor<T>& X, // Input tensor
                Ort::Custom::Tensor<T>& Z) {     // Output tensor

    cudaStream_t stream = cuda_ctx.cuda_stream;
    CUSTOM_ENFORCE(stream != nullptr, "Failed to get CUDA stream from CudaContext.");

    const T* x_data = X.Data();
    const std::vector<int64_t>& shape = X.Shape();
    int64_t num_elements = X.NumberOfElement();

    CUSTOM_ENFORCE(x_data != nullptr, "Input tensor X data is null.");

    T* z_data = Z.Allocate(shape);
    CUSTOM_ENFORCE(z_data != nullptr, "Failed to allocate memory for output tensor Z.");

    // Launch the CUDA kernel (function is in global namespace)
    ::launch_gelu_kernel<T>(num_elements, z_data, x_data, stream); // Added :: prefix for clarity

    // Check for CUDA errors after kernel launch
    cudaError_t cuda_err = cudaGetLastError();
    // Use the revised CUSTOM_ENFORCE macro correctly
    CUSTOM_ENFORCE(cuda_err == cudaSuccess, std::string("CUDA kernel launch failed: ") + cudaGetErrorString(cuda_err));

    // cudaError_t sync_err = cudaStreamSynchronize(stream); // Optional debug sync
    // CUSTOM_ENFORCE(sync_err == cudaSuccess, std::string("CUDA stream synchronization failed: ") + cudaGetErrorString(sync_err));
}


// --- Registration Function ---
void RegisterOps(Ort::CustomOpDomain& domain) {
    // --- Register Gelu Op ---

    // REMOVED: static const Gelu<float> c_Gelu_float; // This line was wrong

    // Create the OrtLiteCustomOp object for float using the function pointer
    // Use the fully qualified name for OrtLiteCustomOp
    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_CustomOpGelu_float{
        Ort::Custom::CreateLiteCustomOp(
            "CustomCUDAGELU",                   // Op name in ONNX model
            "CUDAExecutionProvider",  // Target Execution Provider
            KernelGelu<float>         // PASS THE FUNCTION POINTER HERE
            // Optional: Add shape inference function pointer if needed
        )
    };
    // Add the float version to the domain
    // Check if the unique_ptr was created successfully before calling get()
    CUSTOM_ENFORCE(c_CustomOpGelu_float != nullptr, "Failed to create CustomOp 'Gelu' for float.");
    domain.Add(c_CustomOpGelu_float.get());


    // --- Register CustomOpOne (from original example, if needed) ---
    /*
    // Assuming KernelOne is declared and defined appropriately:
    // extern void KernelOne(const Ort::Custom::CudaContext&, const Ort::Custom::Tensor<float>&, const Ort::Custom::Tensor<float>&, Ort::Custom::Tensor<float>&);

    static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_CustomOpOne_float{
        Ort::Custom::CreateLiteCustomOp(
             "CustomOpOne",
             "CUDAExecutionProvider",
             KernelOne // Assuming KernelOne is the function pointer
        )
    };
    CUSTOM_ENFORCE(c_CustomOpOne_float != nullptr, "Failed to create CustomOp 'CustomOpOne' for float.");
    domain.Add(c_CustomOpOne_float.get());
    */
}

}  // namespace CustomCUDAOps

#endif // USE_CUDA