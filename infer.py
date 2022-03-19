import os.path

import torch.nn as nn
import torch
import random
import numpy as np
import sys
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import math
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
class Linear_Mango(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear_Mango, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def backward(self, input: Tensor) -> Tensor:
        return F.linear( input - self.bias,torch.inverse(self.weight.t()).t())
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MangoEncoder(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(MangoEncoder, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            Linear_Mango(input_channel, 512),
            nn.ReLU(),
            Linear_Mango(512, 512),
            nn.ReLU(),
            Linear_Mango(512, output_channel),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class MangoDecoder(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(MangoDecoder, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            Linear_Mango(input_channel, 6),
            #nn.ReLU(),
            Linear_Mango(6, 6),
            #nn.ReLU(),
            Linear_Mango(6, output_channel),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    def backward(self,x):
        for i in range(len(self.linear_relu_stack)-1,-1,-1):
            if isinstance(self.linear_relu_stack[i],Linear_Mango):
                x=self.linear_relu_stack[i].backward(x)
            else:
                x = self.linear_relu_stack[i].forward(x)
        return x
if __name__ == '__main__':
    setup_seed(2021)
    model_encode = MangoEncoder(6, 6)
    model_decode = MangoDecoder(6, 6)

    if os.path.exists("./model_encode.pth"):
        print("==>load model model_encode")
        model_encode = torch.load("./model_encode.pth")
    if os.path.exists("./model_decode.pth"):
        print("==>load model model_decode")
        model_decode = torch.load("./model_decode.pth")
    loss_fn = torch.nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_encode = model_encode.to(device)
    model_decode = model_decode.to(device)
    n = 0  # random.randint(0,10)
    x = torch.tensor(4,dtype=torch.float).to(device)
    y = x ** 2
    p = torch.tensor([ 1, 2], device=device)
    xx = x.unsqueeze(-1).pow(p).to(device)
    yy = y.unsqueeze(-1).pow(p).to(device)
    y_mid1 = model_encode(xx)
    y_mid2 = model_decode(yy)
    yynew=model_decode.backward(y_mid1)
    y_mid3=model_decode(yynew)
    print(loss_fn(y_mid1,y_mid2))
    print(loss_fn(yynew, yy))
    print(yynew)

