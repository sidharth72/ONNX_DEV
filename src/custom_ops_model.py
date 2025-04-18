import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np

# Define the model
node1 = helper.make_node(
    'CustomMatMul',         # Your custom op name
    inputs=['A', 'B'],      # Input tensor names
    outputs=['C'],          # Output tensor names
    domain='custom'         # Must match the domain in your C++ code
)

node2 = helper.make_node(
    'CustomReLU',           # Your custom op name
    inputs=['C'],           # Input tensor names
    outputs=['Y'],          # Output tensor names
    domain='custom'         # Must match the domain in your C++ code
)

# Create the graph
graph = helper.make_graph(
    [node1, node2],
    'custom_op_model',
    [
        helper.make_tensor_value_info('A', TensorProto.FLOAT, [3, 4]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, [4, 5]),
    ],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 5])],
)

# Create the model
model = helper.make_model(
    graph,
    producer_name='custom_op_example',
    opset_imports=[helper.make_opsetid('custom', 1)]
)

# Save the model
onnx.save(model, 'models/custom_op_model.onnx')

# Perform inference
# 1. Create session options
so = ort.SessionOptions()

# 2. Register your custom ops library
# Adjust the path to point to your compiled .so file
so.register_custom_ops_library("/home/mcw/work/ONNX_DEV/ops/build/libcustom_ops.so")

# 3. Create the inference session with the options
ort_session = ort.InferenceSession('models/custom_op_model.onnx', so)

# Create dummy input data
input_data = {
    'A': np.random.rand(3, 4).astype(np.float32),
    'B': np.random.rand(4, 5).astype(np.float32)
}

# Run the model
outputs = ort_session.run(None, input_data)

# Print the output
print("Output shape:", outputs[0].shape)
print("Output data:", outputs[0])