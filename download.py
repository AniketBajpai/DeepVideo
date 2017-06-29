from __future__ import print_function

import os
import sys
import tarfile
import subprocess
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description='Download dataset for DeepVideo.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+',
                    choices=['moving_mnist', 'yfcc100m/beach', 'yfcc100m/golf'])


def download_mnist(download_path):
    data_dir = os.path.join(download_path, 'moving_mnist')

    if os.path.exists(data_dir):
        if os.path.isfile(os.path.join(data_dir, 'moving_mnist.npy')) and \
                os.path.isfile(os.path.join(data_dir, 'ids.txt')):
            print('Data was downloaded.')
            return
        else:
            print('Creating directory for data')
            os.mkdir(data_dir)

    data_url = 'http://www.cs.toronto.edu/~emansim/datasets/bouncing_mnist_test.npy'
    filename = 'moving_mnist.npy'
    target_path = os.path.join(data_dir, filename)
    cmd = ['curl', data_url, '-o', target_path]
    print('Downloading data')
    subprocess.call(cmd)
    print('Done')

    num_samples = 10000
    print('Generating ids')
    id_list = range(num_samples)
    id_str_list = [str(x) + '\n' for x in id_list]
    id_path = os.path.join(data_dir, 'ids.txt')
    with open(id_path, 'w') as f:
        f.writelines(id_str_list)
    print('Done')


def download_yfcc(download_path, dataset_type):
    data_dir = os.path.join(download_path, 'yfcc100m')

    if os.path.exists(data_dir):
        if os.path.isfile(os.path.join(data_dir, 'frames-stable-many')) and \
                os.path.isfile(os.path.join(data_dir, '{}.txt'.format(dataset_type))):
            print('Data was downloaded.')
            return
    else:
        print('Creating directory for data')
        os.mkdir(data_dir)

    data_url = 'http://data.csail.mit.edu/videogan/{}.tar.bz2'.format(dataset_type)
    filename = '{}.tar.bz2'.format(dataset_type)
    target_path = os.path.join(data_dir, filename)
    cmd = ['curl', data_url, '-o', target_path]
    print('Downloading data')
    subprocess.call(cmd)
    print('Extracting data')
    cmd = ['tar', 'xvjf', target_path]
    print('Done')

    ids_url = 'http://data.csail.mit.edu/videogan/{}.txt'.format(dataset_type)
    filename = '{}.txt'.format(dataset_type)
    target_path = os.path.join(data_dir, filename)
    cmd = ['curl', ids_url, '-o', target_path]
    print('Downloading ids')
    subprocess.call(cmd)
    print('Done')


if __name__ == '__main__':
    args = parser.parse_args()
    path = '../datasets'
    if not os.path.exists(path):
        os.mkdir(path)

    if 'moving_mnist' in args.datasets:
        download_mnist('../datasets')
    if 'yfcc100m/beach' in args.datasets:
        download_yfcc('../datasets', 'beach')
    if 'yfcc100m/golf' in args.datasets:
        download_yfcc('../datasets', 'golf')
