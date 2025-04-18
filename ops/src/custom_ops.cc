#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include <iostream> // For debug output

namespace CustomOps {

// Custom MatMul Operator
void CustomMatMul(const Ort::Custom::Tensor<float>& A,
                  const Ort::Custom::Tensor<float>& B,
                  Ort::Custom::Tensor<float>& C) {
  try {
    // Get shapes and verify them
    const auto& shape_A = A.Shape();
    const auto& shape_B = B.Shape();
    
    std::cerr << "MatMul shapes: A=" << shape_A.size() << "D, B=" << shape_B.size() << "D" << std::endl;
    
    // Validate matrix dimensions
    if (shape_A.size() != 2 || shape_B.size() != 2) {
      std::cerr << "MatMul error: Expected 2D matrices" << std::endl;
      throw std::runtime_error("Invalid shapes for MatMul. Expected 2D matrices.");
    }
    
    if (shape_A[1] != shape_B[0]) {
      std::cerr << "MatMul error: Incompatible dimensions: " << shape_A[1] << " vs " << shape_B[0] << std::endl;
      throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    }
    
    int64_t rows_A = shape_A[0];
    int64_t cols_A = shape_A[1];
    int64_t cols_B = shape_B[1];
    
    std::cerr << "MatMul dimensions: " << rows_A << "x" << cols_A << " * " 
              << cols_A << "x" << cols_B << std::endl;
    
    // Get data pointers safely
    auto A_data = A.Data();
    auto B_data = B.Data();
    
    // Allocate output with verified dimensions
    std::cerr << "Allocating output of shape [" << rows_A << ", " << cols_B << "]" << std::endl;
    auto C_data = C.Allocate({rows_A, cols_B});


    if (C_data == nullptr) {
      std::cerr << "MatMul error: Failed to allocate output tensor" << std::endl;
      throw std::runtime_error("Failed to allocate output tensor");
    }
    
    // Initialize to zeros
    for (int64_t i = 0; i < rows_A * cols_B; i++) {
      C_data[i] = 0.0f;
    }
    
    // Perform matrix multiplication with safe indexing
    for (int64_t i = 0; i < rows_A; ++i) {
      for (int64_t k = 0; k < cols_A; ++k) {
        float a_val = A_data[i * cols_A + k];
        for (int64_t j = 0; j < cols_B; ++j) {
          size_t c_idx = i * cols_B + j;
          size_t b_idx = k * cols_B + j;
          if (c_idx < rows_A * cols_B && b_idx < cols_A * cols_B) {
            C_data[c_idx] += a_val * B_data[b_idx];
          }
        }
      }
    }
    std::cerr << "MatMul completed successfully" << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Exception in CustomMatMul: " << e.what() << std::endl;
    throw;
  }
}

// Custom ReLU Operator
void CustomReLU(const Ort::Custom::Tensor<float>& input,
                Ort::Custom::Tensor<float>& output) {
  try {
    const auto& shape = input.Shape();
    std::cerr << "ReLU input shape: " << shape.size() << "D" << std::endl;
    
    auto input_data = input.Data();
    int64_t num_elements = input.NumberOfElement();
    
    std::cerr << "Allocating ReLU output with " << num_elements << " elements" << std::endl;
    auto output_data = output.Allocate(shape);
    
    for (int64_t i = 0; i < num_elements; ++i) {
      output_data[i] = std::max(0.0f, input_data[i]);
    }
    std::cerr << "ReLU completed successfully" << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Exception in CustomReLU: " << e.what() << std::endl;
    throw;
  }
}

// Register All Custom Operators - No changes needed here
void RegisterOps(Ort::CustomOpDomain& domain) {
  // Register MatMul
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_CustomMatMul{
      Ort::Custom::CreateLiteCustomOp("CustomMatMul", "CPUExecutionProvider", CustomMatMul)};
  domain.Add(c_CustomMatMul.get());
  std::cerr << "Registered CustomMatMul" << std::endl;

  // Register ReLU
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_CustomReLU{
      Ort::Custom::CreateLiteCustomOp("CustomReLU", "CPUExecutionProvider", CustomReLU)};
  domain.Add(c_CustomReLU.get());
  std::cerr << "Registered CustomReLU" << std::endl;
}

}  // namespace CustomOps