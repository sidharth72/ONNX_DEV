import sys
import os

# Add the ONNX source path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'onnx')))

import onnx
import onnxruntime
import numpy as np
from onnx import helper, TensorProto, AttributeProto, GraphProto


def conv_net()