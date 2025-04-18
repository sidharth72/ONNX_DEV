import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef

@onnx_op(op_type="MatMulCustom",  # Custom operator name
         inputs = [PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
            outputs = [PyCustomOpDef.dt_float],
            version = 1,
            domain = "ai.onnx.contrib")  # Custom domain
def matmul_custom(a, b):
    return np.matmul(a, b)
