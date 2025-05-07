# ONNX Runtime Custom Operator Demo

This project demonstrates how to extend [ONNX Runtime](https://onnxruntime.ai/) with custom C++ operators for deep learning inference, and how to use them from Python. The goal is to enable advanced model optimizations and support for custom layers (e.g., optimized MatMul, GELU, Conv2D) that are not available in standard ONNX opsets, or to accelerate inference for specific workloads (CPU and CUDA).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Custom Operator Development](#custom-operator-development)
  - [CPU Custom Ops](#cpu-custom-ops)
  - [CUDA Custom Ops](#cuda-custom-ops)
- [Building the Custom Operator Libraries](#building-the-custom-operator-libraries)
- [Creating and Exporting ONNX Models](#creating-and-exporting-onnx-models)
  - [Exporting from PyTorch](#exporting-from-pytorch)
  - [Building Models with ONNX Python API](#building-models-with-onnx-python-api)
  - [Fusing Custom Ops into Existing Models](#fusing-custom-ops-into-existing-models)
- [Running Inference with ONNX Runtime](#running-inference-with-onnx-runtime)
  - [CPU Inference](#cpu-inference)
  - [CUDA Inference](#cuda-inference)
- [Advanced: Custom GPT-2 Inference Pipeline](#advanced-custom-gpt-2-inference-pipeline)
- [References](#references)
- [Notes](#notes)
- [Contact](#contact)

---

## Project Structure

```
ONNX_DEV/
├── ops/                # C++ source code for custom operators (CPU & CUDA)
│   ├── cpu/
│   │   ├── src/
│   │   ├── include/
│   │   ├── custom_op_library.cc/h
│   │   └── CMakeLists.txt
│   └── cuda/
│       ├── src/
│       ├── kernels/
│       ├── include/
│       ├── custom_op_library.cc/h
│       └── CMakeLists.txt
├── src/                # Python scripts for model export, graph surgery, and inference
│   ├── export_gpt2.py
│   ├── trace_graph.py
│   ├── infer_gpt_cpu.py
│   ├── infer_gpt_cuda.py
│   ├── custom_ops_model.py
│   ├── custom_model.py
│   ├── torch_custom_sym.py
│   └── play.ipynb
├── models/             # Directory for storing ONNX models (ignored by .gitignore)
│   └── onnx_models/
├── .gitignore
└── README.md           # This file
```

---

## Custom Operator Development

### CPU Custom Ops

- Implemented in [`ops/cpu/src/custom_ops.cc`](ops/cpu/src/custom_ops.cc).
- Includes: `CustomMatMul`, `CustomGELU`, `CustomConv2D`, `CustomReLU`, `CustomConcat`, `CustomReshape`.
- Registered under the domain `"ai.onnx.custom"`.

### CUDA Custom Ops

- Implemented in [`ops/cuda/src/custom_ops.cc`](ops/cuda/src/custom_ops.cc) and CUDA kernels in [`ops/cuda/kernels/`](ops/cuda/kernels/).
- Example: `CustomCUDAGELU` (GELU activation accelerated on GPU).
- Registered under the domain `"ai.onnx.custom"`.

---

## Building the Custom Operator Libraries

### CPU

```sh
cd ops/cpu
mkdir -p build && cd build
cmake ..
make
# Produces libcustom_op.so in build/
```

### CUDA

```sh
cd ops/cuda
mkdir -p build && cd build
cmake ..
make
# Produces libcustom_cuda_ops.so in build/
```

- Make sure to set `ONNXRUNTIME_CUDA_DIR` and `ORT_DIR` in the respective `CMakeLists.txt` to point to your ONNX Runtime installation.

---

## Creating and Exporting ONNX Models

### Exporting from PyTorch

- Use [`src/torch_custom_sym.py`](src/torch_custom_sym.py) to define and export models with custom symbolic ops.
- Example custom ops: `CustomConv2D`, `CustomReLU`, `CustomReshape`, `CustomConcat`.
- Register symbolic functions for ONNX export and export with `torch.onnx.export`.

### Building Models with ONNX Python API

- Use [`src/custom_ops_model.py`](src/custom_ops_model.py) to build ONNX models directly with custom ops using the ONNX Python API.
- Example: Compose a model with `CustomMatMul`, `CustomGELU`, `CustomConv2D`, etc.

### Fusing Custom Ops into Existing Models

- Use [`src/trace_graph.py`](src/trace_graph.py) to:
  - Trace the computation graph of an exported model (e.g., GPT-2).
  - Find and replace subgraphs (e.g., standard GELU implementation) with a single custom op node (e.g., `CustomCUDAGELU`).
  - Clean up the graph and save the new model.

---

## Running Inference with ONNX Runtime

### CPU Inference

- Use [`src/infer_gpt_cpu.py`](src/infer_gpt_cpu.py) to run inference on models with custom CPU ops.
- Register the CPU custom ops library at runtime:
  ```python
  sess_opts = ort.SessionOptions()
  sess_opts.register_custom_ops_library("/path/to/libcustom_op.so")
  session = ort.InferenceSession("model.onnx", sess_opts)
  ```

### CUDA Inference

- Use [`src/infer_gpt_cuda.py`](src/infer_gpt_cuda.py) to run inference on models with custom CUDA ops.
- Register the CUDA custom ops library and specify CUDA as the execution provider:
  ```python
  sess_opts = ort.SessionOptions()
  sess_opts.register_custom_ops_library("/path/to/libcustom_cuda_ops.so")
  session = ort.InferenceSession("model.onnx", sess_opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
  ```

---

## Advanced: Custom GPT-2 Inference Pipeline

1. **Export GPT-2 to ONNX:**  
   Use [`src/export_gpt2.py`](src/export_gpt2.py) to export a HuggingFace GPT-2 model to ONNX.

2. **Fuse GELU with Custom Op:**  
   Use [`src/trace_graph.py`](src/trace_graph.py) to replace the standard GELU subgraph with a single `CustomCUDAGELU` node.

3. **Run Inference:**  
   - For CPU: [`src/infer_gpt_cpu.py`](src/infer_gpt_cpu.py)
   - For CUDA: [`src/infer_gpt_cuda.py`](src/infer_gpt_cuda.py)

4. **Interactive Notebook:**  
   - See [`src/play.ipynb`](src/play.ipynb) for an interactive Python notebook demonstrating inference and prompt-based generation.

---

## References

- [ONNX Runtime Custom Ops Documentation](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [ONNX Python API](https://github.com/onnx/onnx)
- [ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
- [Transformers (HuggingFace)](https://github.com/huggingface/transformers)

---

## Notes

- **Domain and Operator Names:**  
  Ensure the custom op domain (`ai.onnx.custom`) and operator names match between your ONNX model and C++ registration.
- **ONNX Runtime Version:**  
  The custom op shared library must be built against the same ONNX Runtime version as your Python environment.
- **Model Compatibility:**  
  When fusing or replacing subgraphs, validate the model with `onnx.checker.check_model`.
- **Extensibility:**  
  You can add more custom ops by extending the C++ code and registering them in the same domain.

---

## Contact

For questions, issues, or contributions, please open an issue or pull request on this repository.