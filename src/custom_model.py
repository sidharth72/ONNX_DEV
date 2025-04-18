import sys
import os

# Add the ONNX source path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'onnx')))

import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto, AttributeProto, GraphProto
from onnxruntime_extensions import get_library_path
import custom_ops

def dense_net(name, input_shape, hidden_dim, output_dim):
    batch_size, input_dim = input_shape


    # Create input (ValueInfo)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [batch_size, input_dim])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [batch_size, output_dim])

    # Weights for the first dense layer
    W1_value = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    W1 = helper.make_tensor("W1", TensorProto.FLOAT, W1_value.shape, W1_value.tobytes(), raw = True)


    # Create bias for first dense layer
    B1_value = np.zeros(hidden_dim, dtype=np.float32)
    B1 = helper.make_tensor('B1', TensorProto.FLOAT, B1_value.shape, B1_value.tobytes(), raw=True)
    
    # Create weights for second dense layer
    W2_value = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
    W2 = helper.make_tensor('W2', TensorProto.FLOAT, W2_value.shape, W2_value.tobytes(), raw=True)
    
    # Create bias for second dense layer
    B2_value = np.zeros(output_dim, dtype=np.float32)
    B2 = helper.make_tensor('B2', TensorProto.FLOAT, B2_value.shape, B2_value.tobytes(), raw=True)
    
    # Define the graph nodes, (operations)

    node1 = helper.make_node(
        'MatMulCustom',           # Op type
        ['X', 'W1'],        # Inputs
        ['hidden_layer'],   # Outputs
        domain = "ai.onnx.contrib",  # Custom domain
        name='dense1_matmul'  # Name of the node
    )

    print(node1) # Print the custom domain

    node2 = helper.make_node(
        'Add',
        ['hidden_layer', 'B1'],
        ['hidden_with_bias'],
        name='dense1_add_bias'
    )

    # ReLU activation
    node3 = helper.make_node(
        'Relu',
        ['hidden_with_bias'],
        ['hidden_activated'],
        name='relu_activation'
    )

    node4 = helper.make_node(
        'MatMul',
        ['hidden_activated', 'W2'],
        ['output_layer'],
        name='dense2_matmul'
    )


    node5 = helper.make_node(
        'Add',
        ['output_layer', 'B2'],
        ['Y'],
        name='dense2_add_bias'
    )

    graph = helper.make_graph(
        [node1, node2, node3, node4, node5],  # nodes
        name = name,  # name
        inputs = [X],  # inputs
        outputs = [Y],  # outputs
        initializer = [W1, B1, W2, B2]  # initializers
    )


    model = helper.make_model(
        graph,
        producer_name='ONNX-DenseNet',
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ai.onnx.contrib", 1)], # Specify the opset version
        ir_version=6,  # Set the IR version
    )


    onnx.checker.check_model(model)

    return model
    
def run_model(model, input_data):
    
    so = ort.SessionOptions()
    print(get_library_path())
    so.register_custom_ops_library(get_library_path())
    
    session = ort.InferenceSession(model.SerializeToString(), so)

    input_name = session.get_inputs()[0].name

    result = session.run(None, {input_name: input_data})

    return result[0]


    
model = dense_net("DenseNet", (1, 10), 20, 2)
input_data = np.random.randn(1, 10).astype(np.float32)
output = run_model(model, input_data)

print("Model output shape:", output.shape)
print("Model output:", output)

# Create the models directory if it doesn't exist
# os.makedirs('models', exist_ok=True)
# onnx.save(model, 'models/dense_net_custom_op.onnx')
