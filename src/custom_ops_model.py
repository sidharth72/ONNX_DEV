import onnx
import onnx.helper as helper
import onnxruntime as ort
import numpy as np

# MatMul inputs/outputs
A = helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 3])
B = helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [3, 4])
D = helper.make_tensor_value_info("D", onnx.TensorProto.FLOAT, [2, 4])

# Conv2D inputs/outputs
X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 5, 5])
W = helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [2, 3, 3, 3])
Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2, 3, 3])

# Concat inputs/outputs
ConcatA = helper.make_tensor_value_info("ConcatA", onnx.TensorProto.FLOAT, [2, 2])
ConcatB = helper.make_tensor_value_info("ConcatB", onnx.TensorProto.FLOAT, [2, 2])
ConcatOut = helper.make_tensor_value_info("ConcatOut", onnx.TensorProto.FLOAT, [2, 4])

# Reshape inputs/outputs
ReshapeIn = helper.make_tensor_value_info("ReshapeIn", onnx.TensorProto.FLOAT, [2, 4])
ShapeTensor = helper.make_tensor_value_info("ShapeTensor", onnx.TensorProto.INT64, [4])
ReshapeOut = helper.make_tensor_value_info(
    "ReshapeOut", onnx.TensorProto.FLOAT, [2, 2, 2, 1]
)

custom_node = helper.make_node(
    "CustomMatMul", inputs=["A", "B"], outputs=["C"], domain="ai.onnx.custom"
)
gelu_node = helper.make_node(
    "CustomGELU", inputs=["C"], outputs=["D"], domain="ai.onnx.custom"
)
conv_node = helper.make_node(
    "CustomConv2D", inputs=["X", "W"], outputs=["Y"], domain="ai.onnx.custom"
)
concat_node = helper.make_node(
    "CustomConcat",
    inputs=["ConcatA", "ConcatB"],
    outputs=["ConcatOut"],
    domain="ai.onnx.custom",
)
reshape_node = helper.make_node(
    "CustomReshape",
    inputs=["ReshapeIn", "ShapeTensor"],
    outputs=["ReshapeOut"],
    domain="ai.onnx.custom",
)

graph = helper.make_graph(
    [custom_node, gelu_node, conv_node, concat_node, reshape_node],
    "CustomMatMulGELUConv2DConcatReshapeGraph",
    [A, B, X, W, ConcatA, ConcatB, ReshapeIn, ShapeTensor],
    [D, Y, ConcatOut, ReshapeOut],
)

model = helper.make_model(graph, producer_name="custom_ops_example")
onnx.save(model, "../models/onnx_models/custom_matmul_gelu_conv_concat_reshape.onnx")

# 2. Register custom op library and run inference
so_path = "/home/mcw/sidharth/ONNX_DEV/ops/cpu/build/libcustom_op.so"
sess_options = ort.SessionOptions()
sess_options.register_custom_ops_library(so_path)

session = ort.InferenceSession(
    "../models/onnx_models/custom_matmul_gelu_conv_concat_reshape.onnx", sess_options
)

# 3. Prepare input and run inference for all branches

input_A = np.random.rand(2, 3).astype(np.float32)
input_B = np.random.rand(3, 4).astype(np.float32)
input_X = np.random.rand(1, 3, 5, 5).astype(np.float32)
input_W = np.random.rand(2, 3, 3, 3).astype(np.float32)
input_ConcatA = np.random.rand(2, 2).astype(np.float32)
input_ConcatB = np.random.rand(2, 2).astype(np.float32)
input_ReshapeIn = np.random.rand(2, 4).astype(np.float32)
input_ShapeTensor = np.array([2, 2, 2, 1], dtype=np.int64)

outputs = session.run(
    None,
    {
        "A": input_A,
        "B": input_B,
        "X": input_X,
        "W": input_W,
        "ConcatA": input_ConcatA,
        "ConcatB": input_ConcatB,
        "ReshapeIn": input_ReshapeIn,
        "ShapeTensor": input_ShapeTensor,
    },
)

print("Output D (GELU(CustomMatMul(A, B))):\n", outputs[0])
print("Shape of output D:", outputs[0].shape)
print("Output Y (CustomConv2D(X, W)):\n", outputs[1])
print("Shape of output Y:", outputs[1].shape)
print("Output ConcatOut (CustomConcat):\n", outputs[2])
print("Shape of output ConcatOut:", outputs[2].shape)
print("Output ReshapeOut (CustomReshape):\n", outputs[3])
print("Shape of output ReshapeOut:", outputs[3].shape)
