import onnx
import onnx.helper as helper
import onnxruntime as ort
import numpy as np

# 1. Build ONNX model with CustomMatMul node followed by Relu
A = helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, [2, 3])
B = helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [3, 4])
D = helper.make_tensor_value_info('D', onnx.TensorProto.FLOAT, [2, 4])  # Output after Relu

custom_node = helper.make_node(
    'CustomMatMul',
    inputs=['A', 'B'],
    outputs=['C'],
    domain='test.customop'
)

relu_node = helper.make_node(
    'Relu',
    inputs=['C'],
    outputs=['D']
)

graph = helper.make_graph(
    [custom_node, relu_node],
    'CustomMatMulReluGraph',
    [A, B],
    [D]
)

model = helper.make_model(graph, producer_name='custom_ops_example')
onnx.save(model, 'models/custom_matmul_relu.onnx')

# 2. Register custom op library and run inference
so_path = '/home/mcw/work/ONNX_DEV/ops/build/libcustom_op.so'
sess_options = ort.SessionOptions()
sess_options.register_custom_ops_library(so_path)

session = ort.InferenceSession('models/custom_matmul_relu.onnx', sess_options)

# 3. Prepare input and run inference
input_A = np.random.rand(2, 3).astype(np.float32)
input_B = np.random.rand(3, 4).astype(np.float32)

outputs = session.run(
    None,
    {'A': input_A, 'B': input_B}
)

print("Output D (Relu(CustomMatMul(A, B))):\n", outputs[0])
print("Shape of output D:", outputs[0].shape)