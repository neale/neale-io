import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import numpy as np


class SinLayer(nn.Module):
    def __init__(self):
        super(SinLayer, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class CosLayer(nn.Module):
    def __init__(self):
        super(CosLayer, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 c_dim,
                 layer_width,
                 scale_z,
                 name='Generator'):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.scale_z = scale_z
        self.name = name

        self.linear_z = nn.Linear(self.z_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)
        self.linear_h = nn.Linear(self.layer_width, self.layer_width)
        self.linear_out = nn.Linear(self.layer_width, self.c_dim)
        acts = [nn.ELU(), nn.Softplus()]
        self.order = torch.arange(len(acts))
        self.acts = nn.ModuleList(acts)

    def forward(self, x, y, z, r, z_scaled):
        z_pt = self.linear_z(z_scaled)
        x_pt = self.linear_x(x)
        y_pt = self.linear_y(y)
        r_pt = self.linear_r(r)
        U = z_pt + x_pt + y_pt + r_pt
        H = torch.tanh(U)
        H = F.elu(self.linear_h(H))
        H = F.softplus(self.linear_h(H))
        H = torch.tanh(self.linear_h(H))
        x = .5 * torch.sin(self.linear_out(H)) + .5
        return x


class GeneratorRandomAct(nn.Module):
    def __init__(self,
                 z_dim,
                 c_dim,
                 layer_width,
                 scale_z,
                 name='GeneratorRandomAct'):
        super(GeneratorRandomAct, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.layer_width = layer_width
        self.scale_z = scale_z
        self.name = name
        self.linear_z = nn.Linear(self.z_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)
        self.linear_h = nn.Linear(self.layer_width, self.layer_width)
        self.linear_out = nn.Linear(self.layer_width, self.c_dim)

        acts = [nn.ELU(), nn.Hardtanh(), nn.LeakyReLU(), nn.LogSigmoid(),
            nn.SELU(), nn.GELU(), nn.CELU(), nn.Sigmoid(), nn.Mish(),
            nn.Softplus(), nn.Softshrink(), nn.Tanh(), torch.nn.ReLU(),
            SinLayer(), CosLayer()]
        self.order = torch.randint(0, 15, size=(9,))
        a = [acts[i] for i in self.order]
        self.acts = nn.ModuleList(a)

    def forward(self, x, y, z, r, z_scaled):
        z_pt = self.acts[0](self.linear_z(z_scaled))
        x_pt = self.acts[1](self.linear_x(x))
        y_pt = self.acts[2](self.linear_y(y))
        r_pt = self.acts[3](self.linear_r(r))
        U = z_pt.add_(x_pt).add_(y_pt).add_(r_pt)
        H = self.acts[4](U)
        H = self.acts[5](self.linear_h(H))
        H = self.acts[6](self.linear_h(H))
        H = self.acts[7](self.linear_h(H))
        #x = .5 * self.acts[8](self.linear_out(H)) + .5
        x = self.acts[8](self.linear_out(H))
        return x
