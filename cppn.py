import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

import tifffile
from imageio import imwrite, imsave

# Because imageio uses the root logger instead of warnings package...
import logging

from cppn_generator import Generator

logging.getLogger().setLevel(logging.ERROR)


class CPPN(object):
    """initializes a CPPN"""
    def __init__(self,
                 z_dim=8,
                 n_samples=1,
                 x_dim=512,
                 y_dim=512,
                 c_dim=1,
                 z_scale=10,
                 layer_width=32,
                 batch_size=1,
                 interpolation=10,
                 reinit_freq=10,
                 exp_name='.',
                 name_style='params',
                 walk=False,
                 sample=True,
                 init_at_startup=False):
        self.z_dim = z_dim
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.c_dim = c_dim 
        self.z_scale = z_scale
        self.layer_width = layer_width
        self.batch_size = batch_size
        self.interpolation = interpolation
        self.reinit_freq = reinit_freq
        self.exp_name = exp_name
        self.name_style = name_style
        self.walk = walk
        self.sample = sample
        
        self.seed_gen = 1234567890
        self._init_random_seed()
        self._init_paths()
        
        if init_at_startup:
            self.init_generator()
        else:
            self.generator = None
        print (self.generator)

    def _init_random_seed(self):
        self.seed = np.random.randint(self.seed_gen)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


    def _init_paths(self):
        if not os.path.exists('./trials/'):
            os.makedirs('./trials/')

        if not os.path.exists('trials/'+ self.exp_name):
            os.makedirs('trials/'+ self.exp_name)
        else:
            path = self.exp_name
            while os.path.exists('trials/'+ path):
                response = input('Exp Directory Exists, rename (y/n/overwrite):\t')
                if response == 'y':
                    path = input('New Exp Directory Name:\t')
                elif response == 'overwrite':
                    break
            os.makedirs('trials/'+path, exist_ok=True)
            self.exp_name = path

    def _init_weights(self, model):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data)
        return model

    def init_generator(self):
        generator = Generator(self.z_dim,
            self.c_dim, self.x_dim, self.y_dim, self.layer_width, self.z_scale, self.batch_size)
        self.generator = self._init_weights(generator) 

    def init_inputs(self):
        zs = []
        for _ in range(self.n_samples):
            zs.append(torch.zeros(1, self.z_dim).uniform_(-1.0, 1.0))
        return zs

    def _coordinates(self):
        x_dim, y_dim, scale = self.x_dim, self.y_dim, self.z_scale
        n_points = x_dim * y_dim
        x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
        y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_mat = np.tile(x_mat.flatten(), args.batch_size).reshape(self.batch_size, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), args.batch_size).reshape(self.batch_size, n_points, 1)
        r_mat = np.tile(r_mat.flatten(), args.batch_size).reshape(self.batch_size, n_points, 1)
        x_mat = torch.from_numpy(x_mat).float()
        y_mat = torch.from_numpy(y_mat).float()
        r_mat = torch.from_numpy(r_mat).float()
        return x_mat, y_mat, r_mat

    def _write_image(self, path, x, suffix='PNG', metadata=None):
        if suffix == 'PNG':
            path = path + '.png'
            imwrite(path, x)
        elif suffix == 'TIF':
            assert metadata is not None, "metadata must be included for tiff file saving"
            path = path + '.tif'
            tifffile.imsave(path, x, metadata=metadata)
        else:
            raise NotImplementedError

    def sample_frame(self, z):
        x_vec, y_vec, r_vec = self._coordinates()
        frame = self.generator((x_vec, y_vec, z, r_vec))
        return frame


    def latent_walk(self, z1, z2):
        delta = (z2 - z1) / (self.n_samples + 1)
        total_frames = self.n_samples + 2
        states = []
        for i in range(total_frames):
            z = z1 + delta * float(i)
            if args.c_dim == 1:
                states.append(self.sample_frame(z)[0]*255)
            else:
                states.append(self.sample_frame(z)[0]*255)
        states = torch.stack(states).detach().numpy()
        return states
    
def run_cppn(cppn):
    if cppn.name_style == 'simple':
        suff = 'image'
    if cppn.name_style == 'params':
        suff = 'z-{}_scale-{}_cdim-{}_net-{}'.format(
            cppn.z_dim, cppn.z_scale, cppn.c_dim, cppn.layer_width)

    zs = cppn.init_inputs()

    if cppn.walk:
        k = 0
        for i in range(cppn.n_samples):
            if i+1 not in range(cppn.n_samples):
                samples = cppn.latent_walk(zs[i], zs[0])
                break
            samples = cppn.latent_walk(zs[i], zs[i+1])

            for sample in samples:
                save_fn = 'trials/{}/{}_{}'.format(cppn.exp_name, suff, k)
                print ('saving PNG image at: {}'.format(save_fn))
                cppn._write_image(path=save_fn, x=sample, suffix='PNG')
                k += 1
            print ('walked {}/{}'.format(i+1, cppn.n_samples))

    elif cppn.sample:
        zs, _ = torch.stack(zs).sort()
        for i, z in enumerate(zs):
            sample = cppn.sample_frame(z).cpu().detach().numpy()
            sample = sample[0]
            sample = sample * 255

            metadata = dict(seed=str(cppn.seed),
                    z_sample=str(list(z.numpy()[0])),
                    z=str(cppn.z_dim), 
                    c_dim=str(cppn.c_dim),
                    scale=str(cppn.z_scale),
                    net=str(cppn.layer_width))

            save_fn = 'trials/{}/{}_{}'.format(cppn.exp_name, suff, i)
            print ('saving TIFF/PNG image pair at: {}'.format(save_fn))
            cppn._write_image(path=save_fn, x=sample.astype('u1'), suffix='TIF',
                metadata=metadata)
            cppn._write_image(path=save_fn, x=sample, suffix='PNG')
    
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
    parser.add_argument('--exp_name', default='.', type=str, help='output fn')
    parser.add_argument('--name_style', default='params', type=str, help='output fn')
    parser.add_argument('--walk', action='store_true', help='interpolate')
    parser.add_argument('--sample', action='store_true', help='sample n images')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = load_args()
    cppn = CPPN(args.z_dim,
                args.n_samples,
                args.x_dim,
                args.y_dim,
                args.c_dim,
                args.z_scale,
                args.layer_width,
                args.batch_size,
                args.interpolation,
                args.reinit_freq,
                args.exp_name,
                args.name_style,
                args.walk,
                args.sample,
                init_at_startup=True)
    run_cppn(cppn)

