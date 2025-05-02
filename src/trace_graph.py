# Trace the computation graph of a model, find the series of operations that is corresponding
# to the custom GELU operator that we are going to fuse, replace those, clean the graph
# with onnx-graphsurgeon, and save the new model.

import onnx
import re
import onnx_graphsurgeon as gs

model = onnx.load("/home/mcw/sidharth/ONNX_DEV/models/onnx_models/gpt2_124M.onnx")
graph = model.graph

reshape_re = re.compile(r"transformer/h\.\d+/mlp/c_fc/Reshape_1_output_0")
mul_re = re.compile(r"transformer/h\.\d+/mlp/act/Mul_3_output_0")

new_nodes = []
mid_name = None
for node in graph.node:
    # 1) intercept the Reshape, rewrite its output to a temp
    if node.op_type == "Reshape" and reshape_re.search(node.output[0]):
        orig = node.output[0]
        mid_name = orig + "_for_custom_gelu"
        node.output[0] = mid_name
        new_nodes.append(node)

    # 2) when you see the Mul you care about, append it and then hook in CustomGELU
    elif node.op_type == "Mul" and mul_re.search(node.output[0]) and mid_name:
        orig_mul = node.output[0]

        gelu = onnx.helper.make_node(
            "CustomCUDAGELU",
            inputs=[mid_name],
            outputs=[orig_mul],  # now overriding the Mul’s original output
            domain="ai.onnx.custom",
            name=node.name + "_CustomGELU",
        )
        new_nodes.append(gelu)

        # clear temp so you only insert once
        mid_name = None

    else:
        new_nodes.append(node)

# 3) replace and save
graph.ClearField("node")
graph.node.extend(new_nodes)

# Remove unused nodes
gs_graph = gs.import_onnx(model)
gs_graph.cleanup()
model = gs.export_onnx(gs_graph)

# model checker to validate the model
onnx.checker.check_model(model)

# Finally, save it
onnx.save(
    model, "/home/mcw/sidharth/ONNX_DEV/models/onnx_models/gpt2_124M_custom_cuda.onnx"
)
