import h5py
import numpy as np
import imageio
import glob
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--output_prefix', type=str, default='output')
parser.add_argument('--n', type=int, default=8)
parser.add_argument('--num_frames', type=int, default=10)
parser.add_argument('--h', type=int, default=64)
parser.add_argument('--w', type=int, default=64)
parser.add_argument('--c', type=int, default=3)
args = parser.parse_args()

if not args.train_dir:
    raise ValueError("Please specify train_dir")


def visualize(name):
    II = []
    for file in sorted(glob.glob(os.path.join(args.train_dir, '{}_*.hy'.format(name))), key=os.path.getmtime)[:args.n]:
        print (file)
        f = h5py.File(file, 'r')
        I = np.zeros((args.h, args.num_frames * args.w, args.c))
        generated_frames = f[f.keys()[0]]
        _, _, h, w, c = generated_frames.shape
        h_low = (h - args.h) / 2
        h_high = (h + args.h) / 2
        w_low = (w - args.w) / 2
        w_high = (w + args.w) / 2
        # Take only first set of frames from batch
        I = np.reshape(generated_frames[0, 0:args.num_frames, h_low:h_high, w_low:w_high,
                                        0:args.c], (args.num_frames * args.h, args.w, args.c))
        II.append(I)

    # II = np.stack(II)
    imageio.mimwrite('{}_{}.png'.format(args.output_prefix, name), II)


visualize('generated_current')
visualize('generated_future')
