import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.sigmoid = nn.Sigmoid()

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

