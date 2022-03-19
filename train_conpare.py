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


class MangoNet(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(MangoNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_channel, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, output_channel),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == '__main__':
    setup_seed(2021)
    model_encode = MangoNet(2, 2)
    if os.path.exists("./MangoNet.pth"):
        print("==>load model MangoNet")
        model_encode = torch.load("./MangoNet.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_encode = model_encode.to(device)


    loss_fn = torch.nn.MSELoss()
    learning_rate = 5e-5
    optimizer1 = torch.optim.RMSprop(model_encode.parameters(), lr=learning_rate)
    loss_min=300
    for t in range(200000):
        n =0#random.randint(0,10)
        x = torch.linspace(n, n + 10, 100).to(device)
        y = x ** 2
        p = torch.tensor([ 1, 2], device=device)
        xx = x.unsqueeze(-1).pow(p).to(device)
        yy = y.unsqueeze(-1).pow(p).to(device)
        y_pred = model_encode(xx)
        loss = loss_fn(y_pred, yy)
        if t % 1000 == 0:
            print(t, loss.item())

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
    torch.save(model_encode, "./MangoNet.pth")
