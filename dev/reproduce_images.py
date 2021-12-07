import os
import cv2
import argparse
import numpy as np
import torch
import tifffile
import glob
from cppn import CPPN
from cppn_generator_graph_shared_repr import GeneratorRandGraph, plot_graph
from ast import literal_eval

import logging
logging.getLogger().setLevel(logging.ERROR)


def load_args():

    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('--x_dim', default=2048, type=int, help='out image width')
    parser.add_argument('--y_dim', default=2048, type=int, help='out image height')
    parser.add_argument('--c_dim', default=1, type=int, help='channels')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--interpolation', default=10, type=int)
    parser.add_argument('--name_style', default='params', type=str, help='output fn')
    parser.add_argument('--exp', default='.', type=str, help='output fn')
    parser.add_argument('--name', default='.', type=str, help='output fn')
    parser.add_argument('--file', action='store_true', help='choose file path to reproduce')
    parser.add_argument('--rarch', action='store_true', help='load arch graph')
    parser.add_argument('--dir', action='store_true', help='input directory of images to reproduce')
    parser.add_argument('--z_dim', default=8, type=int, help='latent space width')
    parser.add_argument('--z_scale', default=10., type=float, help='mutiplier on z')
    parser.add_argument('--net', default=32, type=int, help='net width')

    args = parser.parse_args()
    return args


def main():
    args = load_args()
    if args.file:
        fn = args.name
        files = [fn]
    if args.dir:
        dirname = args.name
        files = glob.glob(dirname+'/*.tif')
    if not os.path.exists('trials/'+args.exp):
        os.makedirs('trials/'+args.exp)
    print ('Generating {} files'.format(len(files)))

    for idx, path in enumerate(files):
        print ('loaded file from {}'.format(path))
        with tifffile.TiffFile(path) as tif:
           img = tif.asarray()
           metadata = tif.shaped_metadata[0]

        seed = int(metadata['seed'])
        z_dim = int(metadata['z'])
        z_scale = float(metadata['scale'])
        net = int(metadata['net'])
        c_dim = int(metadata['c_dim'])
        graph = metadata['graph']
        z = literal_eval(metadata['z_sample'])
        z = torch.tensor(z)
        print ('z', z_dim)
        print ('scale', z_scale)
        print ('width', net)
        print ('sample', z)
        cppn = CPPN(z_dim=z_dim,
                    x_dim=args.x_dim,
                    y_dim=args.y_dim,
                    c_dim=c_dim,
                    z_scale=z_scale,
                    seed=seed,
                    graph=graph,
                    layer_width=net)
        print('seed', cppn.seed)

        np.random.seed(int(metadata['seed']))
        torch.manual_seed(int(metadata['seed']))
        if graph:
            random = 'graph'
        else:
            random = None
        cppn.init_generator(random=random, graph=graph)
        plot_graph(cppn.generator.graph, path=f'temp/{args.exp}/reprojected_graph.png')
        img = cppn.sample_frame(z, args.x_dim, args.y_dim, 1).cpu().detach().numpy()
        if args.c_dim == 1:
            img = img[0]
        elif args.c_dim == 3:
            if args.x_dim == args.y_dim:
                img = img[0]
            else:
                img = img[0]
        img = img * 255
        if args.name_style == 'params':
            suff = f'z-{z_dim}_scale-{z_scale}_cdim-{c_dim}_net-{net}'
        save_fn = f'temp/{args.exp}/reprojection_{suff}'
        print ('saving PNG image at: ', save_fn)
        cv2.imwrite(save_fn+'.png', img)


if __name__ == '__main__':
    main()
