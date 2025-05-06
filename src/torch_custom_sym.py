import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic


class CustomConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        # Naive 2D convolution (no bias, stride=1, padding=0, dilation=1)
        N, C_in, H_in, W_in = x.shape
        C_out, _, kH, kW = w.shape
        H_out = H_in - kH + 1
        W_out = W_in - kW + 1
        y = torch.zeros((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        for n in range(N):
            for co in range(C_out):
                for ci in range(C_in):
                    for i in range(H_out):
                        for j in range(W_out):
                            y[n, co, i, j] += torch.sum(
                                x[n, ci, i : i + kH, j : j + kW] * w[co, ci]
                            )
        return y

    @staticmethod
    def symbolic(g, x, w):
        return g.op("customdomain::CustomConv2D", x, w)


class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.clamp(x, min=0)

    @staticmethod
    def symbolic(g, x):
        return g.op("customdomain::CustomReLU", x)


class CustomReshape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shape):
        return x.reshape(tuple(shape.tolist()))

    @staticmethod
    def symbolic(g, x, shape):
        return g.op("customdomain::CustomReshape", x, shape)


class CustomConcat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return torch.cat((a, b), dim=-1)

    @staticmethod
    def symbolic(g, a, b):
        return g.op("customdomain::CustomConcat", a, b)


# 2. Register symbolic functions (for opset 17)
register_custom_op_symbolic("customdomain::CustomConv2D", CustomConv2d.symbolic, 17)
register_custom_op_symbolic("customdomain::CustomReLU", CustomReLU.symbolic, 17)
register_custom_op_symbolic("customdomain::CustomReshape", CustomReshape.symbolic, 17)
register_custom_op_symbolic("customdomain::CustomConcat", CustomConcat.symbolic, 17)


# 3. Example model using the custom ops
class MyCustomModel(nn.Module):
    def forward(self, x, w, a, b, shape):
        y = CustomConv2d.apply(x, w)
        y_relu = CustomReLU.apply(y)
        y_reshaped = CustomReshape.apply(y_relu, shape)
        y_concat = CustomConcat.apply(a, b)
        return y_reshaped, y_concat


model = MyCustomModel()
x = torch.randn(1, 3, 5, 5)
w = torch.randn(2, 3, 3, 3)
a = torch.randn(2, 2)
b = torch.randn(2, 2)
shape = torch.tensor([1, 2, 3, 3], dtype=torch.long)


torch.onnx.export(
    model,
    (x, w, a, b, shape),
    "../models/onnx_models/custom_ops_torch_export.onnx",
    input_names=["X", "W", "A", "B", "Shape"],
    output_names=["YReshaped", "YConcat"],
    opset_version=13,
    custom_opsets={"customdomain": 17},
)

print("Exported custom_ops_torch_export.onnx with custom ops!")
