import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 c_dim,
                 x_dim,
                 y_dim,
                 layer_width,
                 scale_z,
                 batch_size,
                 name='Generator'):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.layer_width = layer_width
        self.scale_z = scale_z
        self.batch_size = batch_size
        self.name = name

        dim = self.x_dim * self.y_dim * self.batch_size
        self.linear_z = nn.Linear(self.z_dim, self.layer_width)
        self.linear_x = nn.Linear(1, self.layer_width, bias=False)
        self.linear_y = nn.Linear(1, self.layer_width, bias=False)
        self.linear_r = nn.Linear(1, self.layer_width, bias=False)
        self.linear_h = nn.Linear(self.layer_width, self.layer_width)
        self.linear_out = nn.Linear(self.layer_width, self.c_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, z, r):
        n_points = self.x_dim * self.y_dim
        ones = torch.ones(n_points, 1, dtype=torch.float)
        z_scaled = z.view(self.batch_size, 1, self.z_dim) * ones * self.scale_z
        z_pt = self.linear_z(z_scaled.view(self.batch_size*n_points, self.z_dim))
        x_pt = self.linear_x(x.view(self.batch_size*n_points, -1))
        y_pt = self.linear_y(y.view(self.batch_size*n_points, -1))
        r_pt = self.linear_r(r.view(self.batch_size*n_points, -1))
        U = z_pt + x_pt + y_pt + r_pt
        H = torch.tanh(U)
        H = F.elu(self.linear_h(H))
        H = F.softplus(self.linear_h(H))
        H = torch.tanh(self.linear_h(H))
        x = .5 * torch.sin(self.linear_out(H)) + .5
        img = x.reshape(self.batch_size, self.y_dim, self.x_dim, self.c_dim)
        return img

