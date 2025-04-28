#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "onnxruntime_lite_custom_op.h"
#include <cmath>

#define CUSTOM_ENFORCE(cond, msg)                                \
  if (!(cond)) {                                                 \
    ORT_CXX_API_THROW(msg, OrtErrorCode::ORT_RUNTIME_EXCEPTION); \
  }


using namespace Ort::Custom;

namespace CustomOps {

    struct CustomMatMul {
        CustomMatMul(const OrtApi*, const OrtKernelInfo*) {}

        Ort::Status Compute(
            const Ort::Custom::Tensor<float>& A,
            const Ort::Custom::Tensor<float>& B,
            Ort::Custom::Tensor<float>& C

        ){

            // Implement the operator Logic here

            // Get shapes
            auto shape_A = A.Shape();
            auto shape_B = B.Shape();
            // Get the dimensions

            int64_t rows_A = shape_A[0];
            int64_t cols_A = shape_A[1];
            int64_t rows_B = shape_B[0];
            int64_t cols_B = shape_B[1];

            auto A_data = A.Data();
            auto B_data = B.Data();

            // Allocate output tensor
            auto C_data = C.Allocate({rows_A, cols_B});

            // Fill the output tensor with zeros
            // Note: This is not necessary in ONNX Runtime as it initializes the output tensor to zero
            std::fill_n(C_data, rows_A * cols_B, 0.0f);


            for (int64_t i = 0; i < rows_A; ++i) {
                for (int64_t k = 0; k < cols_A; ++k) {
                    float a_val = A_data[i * cols_A + k];
                    for (int64_t j = 0; j < cols_B; ++j) {
                        C_data[i * cols_B + j] += a_val * B_data[k * cols_B + j];
                    }
                }
            }

            return Ort::Status{nullptr};
        }


        // Shape inference function
        static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {

            // Get the input shapes
            auto A_shape = ctx.GetInputShape(0);
            auto B_shape = ctx.GetInputShape(1);

            // Check if the shapes are compatible for matrix multiplication
            if (A_shape.size() != 2 || B_shape.size() != 2) {
                return Ort::Status("Matrix multiplication requires 2D tensors", OrtErrorCode::ORT_INVALID_ARGUMENT);
            }
            // if (A_shape[1] != B_shape[0]) {
            //     return Ort::Status("Matrix multiplication requires compatible dimensions", OrtErrorCode::ORT_INVALID_ARGUMENT);
            // }

            return Ort::Status{nullptr};
            
        }
    };

    // Custom GELU Operator

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    struct CustomGELU{
        CustomGELU(const OrtApi*, const OrtKernelInfo*) {}

        Ort::Status Compute(
            const Ort::Custom::Tensor<float>& A, // Input tensor, supports arbitrary shape (e.g., [batch, seq_len, hidden_dim])
            Ort::Custom::Tensor<float>& B // Output tensor, same shape as input
        ){
            // Retrieve the shape of the input tensor A.
            // Example: For GPT-2 FFN, shape_A might be [batch,seq_len, 3072]
            auto shape_A = A.Shape();
    
            // Compute the total number of elements in the input tensor.
            // This allows us to process the tensor as a flat array, regardless of its original shape.
            int64_t numel = 1;
            for (auto d : shape_A) numel *= d;
    
            // Get a pointer to the raw input data (flattened by default).
            const float* A_data = A.Data();
    
            // Allocate the output tensor B with the same shape as A.
            // B_data will point to the memory where we write the GELU results.
            float* B_data = B.Allocate(shape_A);
    
            // Precompute constants for the GELU formula:
            // sqrt_2_over_pi = sqrt(2/pi) ≈ 0.7978845608
            // coeff = 0.044715 (empirical constant for GELU)
            constexpr float sqrt_2_over_pi = 0.7978845608f;
            constexpr float coeff = 0.044715f;
    
            // Iterate over every element in the input tensor (flattened view).
            // For each element, apply the GELU transformation:
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            for (int64_t i = 0; i < numel; ++i) {
                float x = A_data[i];            // Current input value
                float x_cubed = x * x * x;      // Compute x^3 for the formula
                float inner = sqrt_2_over_pi * (x + coeff * x_cubed); // Argument to tanh
                float tanh_inner = std::tanh(inner);                  // Nonlinear transformation
                B_data[i] = 0.5f * x * (1.0f + tanh_inner);           // Final GELU output
                // Shape: Each output element corresponds to the input element at the same index.
            }
    
            // Output tensor B now contains the GELU-activated values, with the same shape as input A.
            return Ort::Status{nullptr};
        }
        

    };



// Registration of the custom operator
void RegisterOps(Ort::CustomOpDomain& domain) {

    static const std::unique_ptr<OrtLiteCustomOp> c_CustomMatMul{
        Ort::Custom::CreateLiteCustomOp<CustomMatMul>(
            "CustomMatMul",
            "CPUExecutionProvider"
        )
    };

    static const std::unique_ptr<OrtLiteCustomOp> c_CustomGELU{
        Ort::Custom::CreateLiteCustomOp<CustomGELU>(
            "CustomGELU",
            "CPUExecutionProvider"
        )
    };

    // Register the custom operator with the domain
    domain.Add(c_CustomMatMul.get());
    domain.Add(c_CustomGELU.get());

}

} // namespace CustomOps