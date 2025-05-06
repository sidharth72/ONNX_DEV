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

namespace CustomCPUOps {

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


    // Custom 2D Convolution Operator
struct CustomConv2D {

    /**
     * @brief Constructor for the CustomConv2D kernel instance.
     *
     * @param api Pointer to the ONNX Runtime API structure. Provided by ORT.
     * @param info Pointer to the kernel info structure. Can be used to access attributes
     * passed from the ONNX model node (e.g., stride, padding).
     *
     * NOTE: In this implementation, the constructor is empty as stride, padding, etc.,
     * are hardcoded in the Compute method. A more flexible implementation
     * would parse these attributes from `info` here and store them as members.
     */
    CustomConv2D(const OrtApi* /*api*/, const OrtKernelInfo* /*info*/) {
        // Attributes like stride, padding, dilation could be read from 'info' here
        // Example:
        // stride_ = info->GetAttribute<int64_t>("stride"); // Assuming 'stride' attribute exists
        // pad_ = info->GetAttribute<int64_t>("padding"); // Assuming 'padding' attribute exists
        // dilation_ = info->GetAttribute<int64_t>("dilation"); // Assuming 'dilation' attribute exists
    }

    /**
     * @brief Executes the 2D convolution operation.
     *
     * This method performs the core computation of the convolution.
     * It takes the input tensor (X) and weight tensor (W) and computes
     * the output tensor (Y).
     *
     * @param X Input tensor. Expected shape: [N, C_in, H_in, W_in]
     * N: Batch size
     * C_in: Number of input channels
     * H_in: Input height
     * W_in: Input width
     * @param W Weight tensor (convolution kernel). Expected shape: [C_out, C_in, kH, kW]
     * C_out: Number of output channels (number of filters)
     * C_in: Number of input channels (must match X)
     * kH: Kernel height
     * kW: Kernel width
     * @param Y Output tensor. This tensor will be allocated and filled by this function.
     * Resulting shape: [N, C_out, H_out, W_out]
     * H_out, W_out are calculated based on input dimensions, kernel size,
     * stride, padding, and dilation.
     * @return Ort::Status Indicates success (nullptr) or failure.
     *
     * NOTE: The output tensor parameter name was 'C' in the original signature but used
     * as 'Y' internally. It has been corrected to 'Y' for consistency here. If your
     * registration uses 'C', you should rename the parameter back to 'C'.
     */
    Ort::Status Compute(
        const Ort::Custom::Tensor<float>& X, // Input Tensor (const reference)
        const Ort::Custom::Tensor<float>& W, // Weight Tensor (const reference)
        Ort::Custom::Tensor<float>& Y  // Output Tensor (mutable reference)
    ) {
        // --- 1. Get Tensor Shapes ---
        // Retrieve the dimensions of the input tensor X.
        auto x_shape = X.Shape();
        // Retrieve the dimensions of the weight tensor W.
        auto w_shape = W.Shape();

        // --- 2. Input Shape Validation ---
        // Ensure the input tensor X has exactly 4 dimensions.
        CUSTOM_ENFORCE(x_shape.size() == 4, "Input must be 4D [N, C_in, H_in, W_in]");
        // Ensure the weight tensor W has exactly 4 dimensions.
        CUSTOM_ENFORCE(w_shape.size() == 4, "Weights must be 4D [C_out, C_in, kH, kW]");

        // --- 3. Extract Dimensions ---
        // Extract batch size (N) from the input shape (dimension 0).
        int64_t N = x_shape[0];
        // Extract number of input channels (C_in) from the input shape (dimension 1).
        int64_t C_in = x_shape[1];
        // Extract input height (H_in) from the input shape (dimension 2).
        int64_t H_in = x_shape[2];
        // Extract input width (W_in) from the input shape (dimension 3).
        int64_t W_in = x_shape[3];

        // Extract number of output channels (C_out) from the weight shape (dimension 0).
        int64_t C_out = w_shape[0];
        // Extract number of input channels from the weight shape (dimension 1). Let's call it kC for kernel channels.
        int64_t kC = w_shape[1];
        // Extract kernel height (kH) from the weight shape (dimension 2).
        int64_t kH = w_shape[2];
        // Extract kernel width (kW) from the weight shape (dimension 3).
        int64_t kW = w_shape[3];

        // --- 4. Channel Compatibility Check ---
        // Verify that the number of input channels in X matches the number of input channels in W.
        CUSTOM_ENFORCE(C_in == kC, "Input channels (X dim 1) must match weight channels (W dim 1)");

        // --- 5. Define Convolution Parameters ---
        // Define the stride for the convolution operation (how many pixels the kernel shifts).
        int64_t stride = 1; // Hardcoded: Should ideally be an attribute
        // Define the padding applied to the input tensor's height and width dimensions.
        int64_t pad = 0;    // Hardcoded: Should ideally be an attribute
        // Define the dilation factor for the kernel (spacing between kernel elements).
        int64_t dilation = 1; // Hardcoded: Should ideally be an attribute

        // --- 6. Calculate Output Dimensions ---
        // Calculate the height of the output tensor using the standard convolution formula:
        // H_out = floor((H_in + 2 * pad - dilation * (kH - 1) - 1) / stride) + 1
        int64_t H_out = (H_in + 2 * pad - dilation * (kH - 1) - 1) / stride + 1;
        // Calculate the width of the output tensor using the standard convolution formula:
        // W_out = floor((W_in + 2 * pad - dilation * (kW - 1) - 1) / stride) + 1
        int64_t W_out = (W_in + 2 * pad - dilation * (kW - 1) - 1) / stride + 1;

        // --- 7. Validate Output Dimensions ---
        // Ensure the calculated output dimensions are positive.
        CUSTOM_ENFORCE(H_out > 0 && W_out > 0, "Calculated output dimensions (H_out, W_out) must be positive.");

        // --- 8. Allocate Output Tensor ---
        // Define the shape of the output tensor Y.
        std::vector<int64_t> y_shape = {N, C_out, H_out, W_out};
        // Allocate memory for the output tensor Y with the calculated shape.
        // This returns a raw pointer to the newly allocated buffer.
        float* y_data = Y.Allocate(y_shape);

        // --- 9. Get Data Pointers ---
        // Get a constant raw pointer to the input tensor X's data buffer.
        const float* x_data = X.Data();
        // Get a constant raw pointer to the weight tensor W's data buffer.
        const float* w_data = W.Data();

        // --- 10. Initialize Output Tensor ---
        // Calculate the total number of elements in the output tensor.
        size_t y_total_elements = static_cast<size_t>(N * C_out * H_out * W_out);
        // Fill the entire allocated output buffer with 0.0f before starting the convolution sums.
        std::fill_n(y_data, y_total_elements, 0.0f);

        // --- 11. Core Convolution Computation (Naive Nested Loops) ---
        // This section implements the convolution using nested loops.
        // This is generally *not* performant and serves as a basic reference implementation.
        // Optimized implementations often use techniques like im2col + GEMM.

        // Loop through each item in the batch.
        for (int64_t n = 0; n < N; ++n) {
            // Loop through each output channel (each filter).
            for (int64_t co = 0; co < C_out; ++co) {
                // Loop through each spatial position (height) in the output feature map.
                for (int64_t ho = 0; ho < H_out; ++ho) {
                    // Loop through each spatial position (width) in the output feature map.
                    for (int64_t wo = 0; wo < W_out; ++wo) {
                        // Initialize the sum for the current output element (n, co, ho, wo).
                        float sum = 0.0f;
                        // Loop through each input channel. The contribution from each input channel is summed up.
                        for (int64_t ci = 0; ci < C_in; ++ci) {
                            // Loop through the kernel's height dimension.
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                // Loop through the kernel's width dimension.
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    // --- Calculate Input Indices ---
                                    // Calculate the corresponding input height index (hi) based on output height (ho),
                                    // kernel height position (kh), stride, dilation, and padding.
                                    int64_t hi = ho * stride + kh * dilation - pad;
                                    // Calculate the corresponding input width index (wi) based on output width (wo),
                                    // kernel width position (kw), stride, dilation, and padding.
                                    int64_t wi = wo * stride + kw * dilation - pad;

                                    // --- Boundary Check ---
                                    // Check if the calculated input indices (hi, wi) fall within the
                                    // valid bounds of the input tensor's spatial dimensions (H_in, W_in).
                                    // If indices are out of bounds (due to kernel size/padding), skip this element.
                                    if (hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                        // --- Calculate Flattened Data Indices ---
                                        // Calculate the flattened 1D index for the input tensor X (NCHW format).
                                        // Formula: n * C_in*H_in*W_in + ci * H_in*W_in + hi * W_in + wi
                                        int64_t x_idx = n * (C_in * H_in * W_in) +
                                                        ci * (H_in * W_in) +
                                                        hi * W_in +
                                                        wi;

                                        // Calculate the flattened 1D index for the weight tensor W ([C_out, C_in, kH, kW] format).
                                        // Formula: co * C_in*kH*kW + ci * kH*kW + kh * kW + kw
                                        int64_t w_idx = co * (C_in * kH * kW) +
                                                        ci * (kH * kW) +
                                                        kh * kW +
                                                        kw;

                                        // --- Perform Multiplication and Accumulation ---
                                        // Multiply the input element by the corresponding weight element
                                        // and add the result to the running sum for the current output element.
                                        sum += x_data[x_idx] * w_data[w_idx];
                                    }
                                } // End loop: kernel width (kw)
                            } // End loop: kernel height (kh)
                        } // End loop: input channels (ci)

                        // --- Assign Result to Output Tensor ---
                        // Calculate the flattened 1D index for the output tensor Y (NCHW format).
                        // Formula: n * C_out*H_out*W_out + co * H_out*W_out + ho * W_out + wo
                        int64_t y_idx = n * (C_out * H_out * W_out) +
                                        co * (H_out * W_out) +
                                        ho * W_out +
                                        wo;
                        // Assign the final computed sum to the corresponding element in the output tensor.
                        y_data[y_idx] = sum;
                    } // End loop: output width (wo)
                } // End loop: output height (ho)
            } // End loop: output channels (co)
        } // End loop: batch (n)

        // --- 12. Return Success Status ---
        // Indicate that the computation completed successfully.
        // A nullptr Ort::Status signifies success in the C++ API.
        return Ort::Status{nullptr};
    } // End Compute method

    /**
     * @brief Infers the output shape of the CustomConv2D operator.
     *
     * This *static* method is called by ONNX Runtime during model loading or
     * graph optimization to determine the shape of the output tensor(s)
     * based on the shapes of the input tensors and any attributes, *without*
     * performing the actual computation.
     *
     * @param ctx The shape inference context object provided by ORT. It allows
     * accessing input shapes and setting the inferred output shape.
     * @return Ort::Status Indicates success (nullptr) or failure (e.g., invalid shapes).
     */
    static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
        // --- 1. Get Input Shapes from Context ---
        // Get the shape of the first input (X).
        auto x_shape = ctx.GetInputShape(0);
        // Get the shape of the second input (W).
        auto w_shape = ctx.GetInputShape(1);

        // --- 2. Input Shape Validation ---
        // Validate the dimensionality of the input tensor X.
        if (x_shape.size() != 4)
            return Ort::Status("Input must be 4D [N, C_in, H_in, W_in]", OrtErrorCode::ORT_INVALID_ARGUMENT);
        // Validate the dimensionality of the weight tensor W.
        if (w_shape.size() != 4)
            return Ort::Status("Weights must be 4D [C_out, C_in, kH, kW]", OrtErrorCode::ORT_INVALID_ARGUMENT);
        return Ort::Status{nullptr};
    } // End InferOutputShape method

};


struct CustomReLU {
    CustomReLU(const OrtApi*, const OrtKernelInfo*) {}

    Ort::Status Compute(
        const Ort::Custom::Tensor<float>& A,
        Ort::Custom::Tensor<float>& B
    ) {
        auto shape_A = A.Shape();
        int64_t numel = 1;
        for (auto d : shape_A) numel *= d;
        float* B_data = B.Allocate(shape_A);
        const float* A_data = A.Data();
        for (int64_t i = 0; i < numel; ++i) {
            B_data[i] = std::max(0.0f, A_data[i]);
        }
        return Ort::Status{nullptr};
    }

    // Optional: Shape inference
    static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
        auto in_shape = ctx.GetInputShape(0);
        ctx.SetOutputShape(0, in_shape);
        return Ort::Status{nullptr};
    }
};


struct CustomConcat {
    CustomConcat(const OrtApi*, const OrtKernelInfo*) {}

    Ort::Status Compute(
        const Ort::Custom::Tensor<float>& A,
        const Ort::Custom::Tensor<float>& B,
        Ort::Custom::Tensor<float>& C
    ) {
        // Concatenate along the last axis
        auto shape_A = A.Shape();
        auto shape_B = B.Shape();

        // To ensure the tensors have the same rank
        CUSTOM_ENFORCE(shape_A.size() == shape_B.size(), "Concat: Rank mismatch");
        for (size_t i = 0; i < shape_A.size() - 1; ++i)
            CUSTOM_ENFORCE(shape_A[i] == shape_B[i], "Concat: Shape mismatch except last dim");

        std::vector<int64_t> out_shape = shape_A;
        out_shape.back() += shape_B.back();
        float* C_data = C.Allocate(out_shape);

        const float* A_data = A.Data();
        const float* B_data = B.Data();

        
        int64_t A_size = 1, B_size = 1;
        for (auto d : shape_A) A_size *= d;
        for (auto d : shape_B) B_size *= d;

        std::copy(A_data, A_data + A_size, C_data); // copy A's data into the beginning of C
        std::copy(B_data, B_data + B_size, C_data + A_size); // Copy B's data into C starting at offset A_size

        return Ort::Status{nullptr};
    }

    static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
        auto shape_A = ctx.GetInputShape(0);
        auto shape_B = ctx.GetInputShape(1);
        if (shape_A.size() != shape_B.size())
            return Ort::Status("Concat: Rank mismatch", OrtErrorCode::ORT_INVALID_ARGUMENT);
        return Ort::Status{nullptr};
    }
};


struct CustomReshape {
    CustomReshape(const OrtApi*, const OrtKernelInfo*) {}

    Ort::Status Compute(
        const Ort::Custom::Tensor<float>& A,
        const Ort::Custom::Tensor<int64_t>& shape_tensor,
        Ort::Custom::Tensor<float>& B
    ) {
        auto shape_vec = shape_tensor.Data();
        int64_t num_dims = shape_tensor.Shape()[0];
        std::vector<int64_t> out_shape(shape_vec, shape_vec + num_dims);

        int64_t numel = 1;
        for (auto d : out_shape) numel *= d;
        int64_t in_numel = 1;
        for (auto d : A.Shape()) in_numel *= d;
        CUSTOM_ENFORCE(numel == in_numel, "Reshape: Number of elements must match");

        float* B_data = B.Allocate(out_shape);
        const float* A_data = A.Data();
        std::copy(A_data, A_data + in_numel, B_data);

        return Ort::Status{nullptr};
    }

    static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
        auto shape_tensor_shape = ctx.GetInputShape(1);
        if (shape_tensor_shape.size() != 1)
            return Ort::Status("Reshape: shape tensor must be 1D", OrtErrorCode::ORT_INVALID_ARGUMENT);
        // Output shape is dynamic, so we skip setting it here.
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

    static const std::unique_ptr<OrtLiteCustomOp> c_CustomConv2D{
        Ort::Custom::CreateLiteCustomOp<CustomConv2D>(
            "CustomConv2D",
            "CPUExecutionProvider"
        )
    };

    static const std::unique_ptr<OrtLiteCustomOp> c_CustomReLU{
        Ort::Custom::CreateLiteCustomOp<CustomReLU>(
            "CustomReLU",
            "CPUExecutionProvider"
        )
    };

    static const std::unique_ptr<OrtLiteCustomOp> c_CustomConcat{
        Ort::Custom::CreateLiteCustomOp<CustomConcat>(
            "CustomConcat",
            "CPUExecutionProvider"
        )
    };

    static const std::unique_ptr<OrtLiteCustomOp> c_CustomReshape{
        Ort::Custom::CreateLiteCustomOp<CustomReshape>(
            "CustomReshape",
            "CPUExecutionProvider"
        )
    };

    // Register the custom operator with the domain
    domain.Add(c_CustomMatMul.get());
    domain.Add(c_CustomGELU.get());
    domain.Add(c_CustomConv2D.get());
    domain.Add(c_CustomReLU.get());
    domain.Add(c_CustomConcat.get());
    domain.Add(c_CustomReshape.get());
    

}

} // namespace CustomCPUOps