# ONNX Runtime Custom Operator Demo

This project demonstrates how to extend [ONNX Runtime](https://onnxruntime.ai/) with custom C++ operators for deep learning inference, and how to use them from Python. The goal is to enable advanced model optimizations and support for custom layers (e.g., optimized MatMul, GELU) that are not available in standard ONNX opsets, or to accelerate inference for specific workloads.

## Core Ideas

- **Custom Operator Registration:**  
  Implement new operators in C++ (e.g., `CustomMatMul`, `CustomGELU`) and register them with ONNX Runtime via a shared library.
- **ONNX Model Construction:**  
  Build ONNX models that use these custom ops, either from scratch (with ONNX Python API) or by modifying exported models (e.g., GPT-2) to fuse subgraphs into a single custom node.
- **Python Inference:**  
  Load and run ONNX models with custom ops in Python using `onnxruntime`, registering the custom op shared library at runtime.
- **Performance and Flexibility:**  
  Custom ops can be optimized for specific hardware or algorithms, enabling faster inference or support for novel layers.

---

## Project Structure

- ops  
  C++ source code for custom operators and the shared library (`libcustom_op.so`).
- src  
  Python scripts for model export, graph surgery, and inference.
- models  
  Directory for storing ONNX models (ignored by .gitignore).

---

## Example: Registering and Using Custom C++ Operators

### 1. Implement Custom Operators in C++

See custom_ops.cc:

```cpp
struct CustomMatMul {
    Ort::Status Compute(
        const Ort::Custom::Tensor<float>& A,
        const Ort::Custom::Tensor<float>& B,
        Ort::Custom::Tensor<float>& C
    ) {
        // Matrix multiplication logic...
    }
};
```

Register your ops in a domain:

```cpp
void RegisterOps(Ort::CustomOpDomain& domain) {
    domain.Add(...); // Add CustomMatMul, CustomGELU, etc.
}
```

### 2. Build the Shared Library

```sh
cd ops
mkdir build && cd build
cmake ..
make
# Produces libcustom_op.so
```

### 3. Create an ONNX Model Using Custom Ops

See custom_ops_model.py:

```python
import onnx
from onnx import helper

custom_node = helper.make_node(
    'CustomMatMul',
    inputs=['A', 'B'],
    outputs=['C'],
    domain='ai.onnx.custom'
)
gelu_node = helper.make_node(
    'CustomGELU',
    inputs=['C'],
    outputs=['D'],
    domain='ai.onnx.custom'
)
# Build and save the model...
```

### 4. Run Inference with ONNX Runtime

```python
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.register_custom_ops_library("/path/to/libcustom_op.so")
session = ort.InferenceSession("models/custom_matmul_relu.onnx", sess_options)
outputs = session.run(None, {"A": input_A, "B": input_B})
```

### 5. Fusing Custom Ops into Existing Models

You can use trace_graph.py to replace subgraphs (e.g., a sequence of nodes implementing GELU) with a single custom node for efficiency.

---

## Example: Custom GPT-2 Inference

- Export GPT-2 to ONNX: export_gpt2.py
- Replace GELU with custom op: trace_graph.py
- Run inference: infer_gpt.py

---

## References

- [ONNX Runtime Custom Ops Documentation](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [ONNX Python API](https://github.com/onnx/onnx)
- [ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)

---

## Notes

- Make sure to match the custom op domain and operator names in both C++ and Python/ONNX.
- The shared library must be built against the same ONNX Runtime version as your Python environment.
- See the code in ops and src for more details and usage examples.

---

**Contact:**  
For questions or contributions, please open an issue or pull request.