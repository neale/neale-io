import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2

import tifffile

# Because imageio uses the root logger instead of warnings package...
import logging

from cppn_generator_graph_shared_repr import GeneratorRandGraph, plot_graph
from cppn_generator import GeneratorRandAct
from cppn_generator import Generator

logging.getLogger().setLevel(logging.ERROR)


class CPPN(object):
    """initializes a CPPN"""
    def __init__(self,
                 z_dim=4,
                 n_samples=6,
                 x_dim=512,
                 y_dim=512,
                 x_dim_gallary=256,
                 y_dim_gallary=256,
                 c_dim=3,
                 z_scale=10,
                 layer_width=4,
                 batch_size=16,
                 interpolation=10,
                 reinit_freq=10,
                 exp_name='.',
                 nodes=10,
                 name_style='params',
                 walk=False,
                 sample=True,
                 seed_gen=123456789,
                 seed=None,
                 graph=None,
                 init_at_startup=False):
        self.z_dim = z_dim
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim 
        self.x_dim_gallary = x_dim_gallary
        self.y_dim_gallary = y_dim_gallary
        self.z_scale = z_scale
        self.layer_width = layer_width
        self.batch_size = batch_size
        self.interpolation = interpolation
        self.reinit_freq = reinit_freq
        self.exp_name = exp_name
        self.name_style = name_style
        self.walk = walk
        self.sample = sample

        self.nodes = nodes
        self.graph = graph
        
        self.seed = seed
        self.seed_gen = seed_gen
        self.init_random_seed(seed=seed)
        self._init_paths()
        
        if init_at_startup:
            self.init_generator()
        else:
            self.generator = None

    def init_random_seed(self, seed=None):
        """ 
        initializes random seed for torch. Random seed needs
            to be a stored value so that we can save the right metadata. 
            This is not to be confused with the uid that is not a seed, 
            rather how we associate user sessions with data
        """
        if seed == None:
            print ('initing with seed gen: {}'.format(self.seed_gen))
            self.seed = np.random.randint(self.seed_gen)
            # np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        else:
            # np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def _init_paths(self):
        os.makedirs(os.path.join('temp/', self.exp_name), exist_ok=True)

    def _init_weights(self, model):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data)
        return model

    def init_generator(self, random=None, seed=None, graph=None):
        if random == 'act':
            generator = GeneratorRandomAct(self.z_dim, self.c_dim, self.layer_width, self.z_scale)
        elif random == 'graph':
            generator = GeneratorRandGraph(self.z_dim,
                                           self.c_dim,
                                           self.layer_width,
                                           self.z_scale,
                                           self.nodes,
                                           graph)
        else:
            generator = Generator(self.z_dim, self.c_dim, self.layer_width, self.z_scale)
        self.generator = self._init_weights(generator)
        self.generator.eval()

    def init_inputs(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.ones(batch_size, 1, self.z_dim).uniform_(-1., 1.)

    def _coordinates(self, x_dim, y_dim, batch_size):
        x_dim, y_dim, scale = x_dim, y_dim, self.z_scale
        n_pts = x_dim * y_dim
        x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
        y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
        print (x_dim, y_dim)
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_vec = np.tile(x_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)
        y_vec = np.tile(y_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)
        r_vec = np.tile(r_mat.flatten(), batch_size).reshape(batch_size*n_pts, -1)

        x_vec = torch.from_numpy(x_vec).float()
        y_vec = torch.from_numpy(y_vec).float()
        r_vec = torch.from_numpy(r_vec).float()


        return x_vec, y_vec, r_vec, n_pts

    def _write_image(self, path, x, suffix='jpg', metadata=None):
        if suffix in ['jpg', 'png']:
            path = path + f'.{suffix}'
            cv2.imwrite(path, x)
        elif suffix == 'tif':
            assert metadata is not None, "metadata must be included to save tif"
            path = path + '.tif'
            tifffile.imsave(path, x, metadata=metadata)
        else:
            raise NotImplementedError


    def sample_frame(self, z, x_dim, y_dim, batch_size):
        with torch.no_grad():
            print ('generating ({},{}) with seed {}-{}'.format(
                x_dim, y_dim, self.seed, self.seed_gen))
            x_vec, y_vec, r_vec, n_pts = self._coordinates(x_dim, y_dim, batch_size)
            one_vec = torch.ones(n_pts, 1, dtype=torch.float)
            z_scale = z.view(batch_size, 1, self.z_dim) * one_vec * self.z_scale
            z_scale = z_scale.view(batch_size*n_pts, self.z_dim)
            print ('inputs: ', x_vec.shape)
            frame = self.generator(x_vec, y_vec, r_vec, z_scale)
            print ('outputs: ', frame.shape)

            frame = frame.reshape(batch_size, y_dim, x_dim, self.c_dim)
        return frame


    def latent_walk(self, z1, z2):
        delta = (z2 - z1) / (self.n_samples + 1)
        total_frames = self.n_samples + 2
        states = []
        for i in range(total_frames):
            z = z1 + delta * float(i)
            if self.c_dim == 1:
                states.append(self.sample_frame(z)[0]*255)
            else:
                states.append(self.sample_frame(z)[0]*255)
        states = torch.stack(states).detach().numpy()
        return states

    
def run_cppn(cppn, autosave=False, z=None):
    if cppn.name_style == 'simple':
        suff = 'image'
    if cppn.name_style == 'params':
        suff = 'z-{}_scale-{}_cdim-{}_net-{}'.format(
            cppn.z_dim, cppn.z_scale, cppn.c_dim, cppn.layer_width)
    zs = []
    for _ in range(cppn.n_samples):
        zs.append(torch.zeros(1, cppn.z_dim).uniform_(-1, 1))
    zs, _ = torch.stack(zs).sort()
    batch_samples = [cppn.sample_frame(
                        z_i,
                        cppn.x_dim_gallary,
                        cppn.y_dim_gallary,
                        batch_size=1)[0].numpy() * 255. for z_i in zs]

    n = np.random.randint(99999999)
    if 'GeneratorRand' in cppn.generator.name:
        randgen = 1
    else:
        randgen = 0

    if 'GeneratorRand' in cppn.generator.name:
        graph = cppn.generator.get_graph_str()
    else:
        graph = ''
    print ('data')
    print('z', cppn.z_dim)
    print('net', cppn.layer_width)
    print('scale', cppn.z_scale)
    print('sample', str(zs[0].numpy().reshape(-1).tolist()))
    print ('seed', cppn.seed)

    for i, (img, z_j) in enumerate(zip([*batch_samples], [*zs])):
        metadata = dict(seed=str(cppn.seed),
                seed_gen=str(cppn.seed_gen),
                z_sample=str(z_j.numpy().reshape(-1).tolist()),
                z=str(cppn.z_dim), 
                c_dim=str(cppn.c_dim),
                scale=str(cppn.z_scale),
                net=str(cppn.layer_width),
                graph=graph,
                randgen=str(randgen))

        save_fn = 'temp/{}/{}{}_{}'.format(cppn.exp_name, n, suff, i)
        #print ('saving TIFF/PNG image pair at: {}'.format(save_fn))
        if i == 0:
            save_fn_lrg = save_fn[:-1]+'lrg{}'.format(i)
            cppn._write_image(path=save_fn_lrg, x=img, suffix='jpg')
        cppn._write_image(path=save_fn, x=img, suffix='jpg')
        cppn._write_image(path=save_fn, x=img.astype('u1'), suffix='tif',
            metadata=metadata)
    plot_graph(cppn.generator.graph,
               path=f'temp/{cppn.exp_name}/graph_{n}.png')


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--z_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--n_samples', default=1, type=int, help='images to generate')
    parser.add_argument('--x_dim', default=2048, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2048, type=int, help='out image height')
    parser.add_argument('--z_scale', default=10, type=float, help='mutiplier on z')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--layer_width', default=32, type=int, help='net width')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--interpolation', default=10, type=int)
    parser.add_argument('--reinit_freq', default=10, type=int, help='reinit generator every so often')
    parser.add_argument('--nodes', default=10, type=int, help='number of nodes in graph')
    parser.add_argument('--exp_name', default='./', type=str, help='output fn')
    parser.add_argument('--name_style', default='params', type=str, help='output fn')
    parser.add_argument('--walk', action='store_true', help='interpolate')
    parser.add_argument('--sample', action='store_true', help='sample n images')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()
    """
    for z in [1, 2, 4, 8, 16, 32]:
        for scale in [.5, 1, 2, 4, 10, 32]:
            for w in [2, 4, 8, 16, 32, 64]:
                for nodes in [4, 10, 20, 40]:
                    name = args.exp_name+f'_z{z}_scale{scale}_width{w}'
                    for _ in range(50):
                        cppn = CPPN(z_dim=z,
                                    n_samples=args.n_samples,
                                    x_dim=args.x_dim,
                                    y_dim=args.y_dim,
                                    c_dim=args.c_dim,
                                    z_scale=scale,
                                    layer_width=w,
                                    batch_size=args.batch_size,
                                    interpolation=args.interpolation,
                                    exp_name=name,
                                    nodes=nodes,
                                    name_style=args.name_style)
                        cppn.init_generator(random='graph')
                        run_cppn(cppn)
    """
    cppn = CPPN(z_dim=args.z_dim,
                n_samples=args.n_samples,
                x_dim=args.x_dim,
                y_dim=args.y_dim,
                c_dim=args.c_dim,
                z_scale=args.z_scale,
                layer_width=args.layer_width,
                batch_size=args.batch_size,
                interpolation=args.interpolation,
                exp_name=args.exp_name,
                nodes=args.nodes,
                name_style=args.name_style)
    cppn.init_generator(random='graph')
    run_cppn(cppn)
