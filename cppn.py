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

from cppn_generator import Generator, RandomGenerator
torch.backends.cudnn.benchmark = True

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
                 name_style='params',
                 walk=False,
                 sample=True,
                 uid='1234',
                 seed_gen=123456789,
                 seed=None,
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
        self.uid = uid
        
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
        if not os.path.exists(self.exp_name+'/'+str(self.uid)):
            os.makedirs(self.exp_name+'/'+str(self.uid))

    def _init_weights(self, model):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data)
        return model

    def init_generator(self, random_generator=False, seed=None):
        if random_generator:
            generator = RandomGenerator(self.z_dim, self.c_dim, self.layer_width, self.z_scale)
        else:
            generator = Generator(self.z_dim, self.c_dim, self.layer_width, self.z_scale)
        self.generator = torch.jit.script(self._init_weights(generator))
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
            frame = self.generator(x_vec, y_vec, z, r_vec, z_scale)
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

    
def run_cppn(cppn, uid, autosave=False, z=None):
    if cppn.name_style == 'simple':
        suff = 'image'
    if cppn.name_style == 'params':
        suff = 'z-{}_scale-{}_cdim-{}_net-{}'.format(
            cppn.z_dim, cppn.z_scale, cppn.c_dim, cppn.layer_width)
    if z is None:
        z_batch = cppn.init_inputs(batch_size=6)
    else:
        z_batch = z.repeat(6, 1)
        z_batch += torch.ones_like(z_batch).uniform_(-.25, .25)
        idx_rand = torch.randint(0, 5, size=(1,))
        z_batch[idx_rand] = torch.tensor([1.,]).normal_(0, .4)

    # get file number
    n_files = len(glob.glob(f"{cppn.exp_name}/{uid}/*.jpg"))
    if cppn.walk:
        print(f"Initializing Walk for {uid}")
        k = n_files
        for i in range(cppn.n_samples):
            if i+1 not in range(cppn.n_samples):
                samples = cppn.latent_walk(zs[i], zs[0])
                break
            samples = cppn.latent_walk(zs[i], zs[i+1])
            for sample in samples:
                if autosave:
                    save_fn = '{}/{}/{}_{}'.format(cppn.exp_name, uid, suff, k)
                    print (f'saving JPG image at: {save_fn}')
                    cppn._write_image(path=save_fn, x=sample, suffix='jpg')
                k += 1
            print ('walked {}/{}'.format(i+1, cppn.n_samples))

    elif cppn.sample:
        print(f"Initializing Sample for UID: {uid}")
        z = z_batch[0].unsqueeze(0)
        sample = cppn.sample_frame(z, cppn.x_dim, cppn.y_dim, batch_size=1)
        sample = sample[0].numpy() * 255.

        batch_samples = [cppn.sample_frame(
                            z_i,
                            cppn.x_dim_gallary,
                            cppn.y_dim_gallary,
                            batch_size=1)[0].numpy() * 255. for z_i in z_batch]

        n = np.random.randint(99999999)
        if cppn.generator.name == 'RandomGenerator':
            randgen = 1
        else:
            randgen = 0
        for i, (img, z_j) in enumerate(zip([sample, *batch_samples], [z, *z_batch])):
            metadata = dict(seed=str(cppn.seed),
                    seed_gen=str(cppn.seed_gen),
                    z_sample=str(z_j.numpy().reshape(-1).tolist()),
                    z=str(cppn.z_dim), 
                    c_dim=str(cppn.c_dim),
                    scale=str(cppn.z_scale),
                    net=str(cppn.layer_width),
                    randgen=str(randgen))
            if autosave:
                save_fn = f'{cppn.exp_name}/{uid}/temp/{n}{suff}_{i}'
                #print ('saving TIFF/PNG image pair at: {}'.format(save_fn))
                if i == 0:
                    save_fn_lrg = save_fn[:-1]+'lrg{}'.format(i)
                    cppn._write_image(path=save_fn_lrg, x=img, suffix='jpg')
                cppn._write_image(path=save_fn, x=img, suffix='jpg')
                cppn._write_image(path=save_fn, x=img.astype('u1'), suffix='tif',
                    metadata=metadata)

        return (sample, batch_samples), suff
    
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

