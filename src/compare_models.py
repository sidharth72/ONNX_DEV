

import sys
import os

# Add the ONNX source path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'onnx')))

import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from onnx import helper, numpy_helper
import os
import tempfile

def visualize_onnx_model(model, title='ONNX Model Visualization'):
    """
    Visualize an ONNX model as a graph using NetworkX and Matplotlib.
    
    Args:
        model: An ONNX model object or path to an ONNX model file
        title: Title for the visualization plot
    """
    if isinstance(model, str):
        # Load the model if a path is provided
        model = onnx.load(model)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for model inputs
    for input_info in model.graph.input:
        G.add_node(input_info.name, type='input')
    
    # Add nodes for model initializers
    initializer_names = {init.name for init in model.graph.initializer}
    for init in model.graph.initializer:
        # Skip adding initializers that are also graph inputs
        if init.name not in [input.name for input in model.graph.input]:
            G.add_node(init.name, type='initializer')
    
    # Add nodes for model outputs
    for output_info in model.graph.output:
        G.add_node(output_info.name, type='output')
    
    # Add nodes for operations and edges for connections
    for node in model.graph.node:
        op_type = node.op_type
        node_name = node.name if node.name else f"{op_type}_{len(G)}"
        G.add_node(node_name, type='operation', op_type=op_type)
        
        # Connect inputs to this node
        for input_name in node.input:
            if input_name:  # Skip empty input names
                # If input is not in graph, it might be an intermediate result
                if input_name not in G:
                    G.add_node(input_name, type='intermediate')
                G.add_edge(input_name, node_name)
        
        # Connect this node to its outputs
        for output_name in node.output:
            if output_name:  # Skip empty output names
                # If output is not in graph, add it as an intermediate result
                if output_name not in G:
                    G.add_node(output_name, type='intermediate')
                G.add_edge(node_name, output_name)
    
    # Prepare node colors
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        if node_type == 'input':
            node_colors.append('lightblue')
            node_sizes.append(1000)
        elif node_type == 'output':
            node_colors.append('lightgreen')
            node_sizes.append(1000)
        elif node_type == 'initializer':
            node_colors.append('lightgrey')
            node_sizes.append(700)
        elif node_type == 'operation':
            node_colors.append('salmon')
            node_sizes.append(1200)
        else:
            node_colors.append('white')
            node_sizes.append(500)
    
    # Visualize the graph
    plt.figure(figsize=(20, 12))
    plt.title(title, fontsize=16)
    
    # Use a layout that works well for directed graphs
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowsize=20, width=1.5, alpha=0.5)
    
    # Draw labels
    labels = {}
    for node in G.nodes():
        node_type = G.nodes[node].get('type', '')
        if node_type == 'operation':
            op_type = G.nodes[node].get('op_type', '')
            labels[node] = f"{op_type}"
        elif node_type in ['input', 'output', 'initializer']:
            # Truncate long names
            short_name = node if len(node) < 20 else node[:17] + "..."
            labels[node] = f"{short_name}\n({node_type})"
        else:
            # Skip intermediate tensor labels to reduce clutter
            labels[node] = ""
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Input'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Output'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=15, label='Operation'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey', markersize=15, label='Initializer')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.axis('off')
    
    # Return statistics about the model
    stats = {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'operations': sum(1 for _, attr in G.nodes(data=True) if attr.get('type') == 'operation'),
        'op_types': set(attr.get('op_type', '') for _, attr in G.nodes(data=True) if attr.get('type') == 'operation')
    }
    
    return G, stats

def get_optimized_onnx_model(model_path, optimized_model_path=None, optimization_level=99):
    """
    Get an optimized version of an ONNX model using ONNX Runtime.
    
    Args:
        model_path: Path to the original ONNX model
        optimized_model_path: Path to save the optimized model (optional)
        optimization_level: Optimization level (0-99)
        
    Returns:
        Path to the optimized model
    """
    if optimized_model_path is None:
        # Create a temporary file if no path is provided
        temp_dir = tempfile.mkdtemp()
        optimized_model_path = os.path.join(temp_dir, 'optimized_model.onnx')
    
    # Create session options with desired optimization level
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = optimization_level
    sess_options.optimized_model_filepath = optimized_model_path
    
    # Creating session will optimize and save the model
    _ = ort.InferenceSession(model_path, sess_options)
    
    return optimized_model_path

def compare_onnx_models(original_model_path, optimized_model_path=None):
    """
    Compare the original and optimized ONNX models and visualize them.
    
    Args:
        original_model_path: Path to the original ONNX model
        optimized_model_path: Path to the optimized model (if None, will be generated)
    """
    # Load the original model
    original_model = onnx.load(original_model_path)
    
    # Get the optimized model
    if optimized_model_path is None or not os.path.exists(optimized_model_path):
        optimized_model_path = get_optimized_onnx_model(original_model_path)
    
    # Load the optimized model
    optimized_model = onnx.load(optimized_model_path)
    
    # Visualize the original model
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    original_graph, original_stats = visualize_onnx_model(original_model, "Original ONNX Model")
    
    # Visualize the optimized model
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 2)
    optimized_graph, optimized_stats = visualize_onnx_model(optimized_model, "Optimized ONNX Model")
    
    # Print statistics
    print("Original Model Statistics:")
    print(f"Total nodes: {original_stats['nodes']}")
    print(f"Total edges: {original_stats['edges']}")
    print(f"Operation nodes: {original_stats['operations']}")
    print(f"Operation types: {', '.join(sorted(original_stats['op_types']))}")
    print("\nOptimized Model Statistics:")
    print(f"Total nodes: {optimized_stats['nodes']}")
    print(f"Total edges: {optimized_stats['edges']}")
    print(f"Operation nodes: {optimized_stats['operations']}")
    print(f"Operation types: {', '.join(sorted(optimized_stats['op_types']))}")
    print(f"\nReduction in operations: {original_stats['operations'] - optimized_stats['operations']} ({(1 - optimized_stats['operations']/original_stats['operations'])*100:.2f}%)")
    
    # Identify operations that were eliminated or added
    original_ops = {attr.get('op_type', '') for _, attr in original_graph.nodes(data=True) if attr.get('type') == 'operation'}
    optimized_ops = {attr.get('op_type', '') for _, attr in optimized_graph.nodes(data=True) if attr.get('type') == 'operation'}
    
    eliminated_ops = original_ops - optimized_ops
    added_ops = optimized_ops - original_ops
    
    if eliminated_ops:
        print(f"\nOperation types eliminated: {', '.join(sorted(eliminated_ops))}")
    if added_ops:
        print(f"Operation types added: {', '.join(sorted(added_ops))}")
    
    return original_model, optimized_model, original_stats, optimized_stats

# Example usage
def create_example_model():
    """Create a simple example ONNX model with constant folding opportunities"""
    # Create inputs
    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    
    # Create constants
    const_A = numpy_helper.from_array(np.random.randn(16, 3, 3, 3).astype(np.float32), name='const_A')
    const_B = numpy_helper.from_array(np.random.randn(16).astype(np.float32), name='const_B')
    const_C = numpy_helper.from_array(np.random.randn(16, 16, 3, 3).astype(np.float32), name='const_C')
    const_D = numpy_helper.from_array(np.random.randn(16).astype(np.float32), name='const_D')
    
    # Create nodes
    node1 = helper.make_node('Conv', ['X', 'const_A'], ['conv1_out'], name='conv1', 
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    node2 = helper.make_node('Add', ['conv1_out', 'const_B'], ['add1_out'], name='add1')
    node3 = helper.make_node('Relu', ['add1_out'], ['relu1_out'], name='relu1')
    node4 = helper.make_node('Conv', ['relu1_out', 'const_C'], ['conv2_out'], name='conv2', 
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    node5 = helper.make_node('Add', ['conv2_out', 'const_D'], ['add2_out'], name='add2')
    node6 = helper.make_node('Relu', ['add2_out'], ['Y'], name='relu2')
    
    # Create graph and model
    Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 16, 224, 224])
    graph = helper.make_graph(
        [node1, node2, node3, node4, node5, node6],
        'example_model',
        [X],
        [Y],
        [const_A, const_B, const_C, const_D]
    )
    
    model = helper.make_model(graph, producer_name='onnx-example')
    model.opset_import[0].version = 13
    
    # Save the model
    model_path = 'models/example_model.onnx'
    onnx.save(model, model_path)
    
    return model_path

def main():
    """Main function to demonstrate ONNX model optimization and visualization"""
    # Either use an existing model or create an example one
    model_path = create_example_model()
    print(f"Model created at: {model_path}")
    
    # Compare the original and optimized models
    compare_onnx_models(model_path)
    
    plt.show()

if __name__ == "__main__":
    main()