from __future__ import print_function

import h5py
import numpy as np
import cv2
import imageio
import glob
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--output_prefix', type=str, default='output')
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--num_frames', type=int, default=10)
parser.add_argument('--h', type=int, default=64)
parser.add_argument('--w', type=int, default=64)
parser.add_argument('--c', type=int, default=3)
args = parser.parse_args()

if not args.train_dir:
    raise ValueError("Please specify train_dir")

if not os.path.exists('./outputs'):
    os.mkdir('./outputs')


def visualize(name):
    for counter, file in enumerate(sorted(glob.glob(os.path.join(args.train_dir, '{}_*.hy'.format(name))), key=os.path.getmtime)[-args.n:]):
        print (file)
        f = h5py.File(file, 'r')
        # I = np.zeros((args.h, args.num_frames * args.w, args.c))
        generated_frames = f[f.keys()[0]]
        _, _, h, w, c = generated_frames.shape
        h_low = (h - args.h) / 2
        h_high = (h + args.h) / 2
        w_low = (w - args.w) / 2
        w_high = (w + args.w) / 2
        # Take only first set of frames from batch
        II = []
        if args.c == 1:
            for j in range(args.num_frames):
                I = np.reshape(generated_frames[0, j, h_low:h_high, w_low:w_high, 0], (args.h, args.w))
                if (I < 1.0).all() and (I > -1.0).all():
                    # print ('Image in [-1, 1]')
                    I = ((I + 1.0) / 2 * 255).astype(np.int32)

                II.append(I)

        else:
            for j in range(args.num_frames):
                I = np.reshape(generated_frames[0, j, h_low:h_high, w_low:w_high, 0:args.c], (args.h, args.w, args.c))
                II.append(I)

        II = np.stack(II)
        II = np.reshape(II, (args.num_frames * args.h, args.w))
        output_img_path = './outputs/{}_{}_{}.png'.format(args.output_prefix, name, str(counter))
        print ('Writing image:', output_img_path)
        print (II.shape)
        cv2.imwrite(output_img_path, II)
        # imageio.mimwrite(output_img_path, II)


visualize('generated_current')
visualize('generated_future')
