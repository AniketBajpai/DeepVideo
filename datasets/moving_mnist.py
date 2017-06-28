from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np

from utils import log, from_json_file

# __PATH__ = os.path.abspath(os.path.dirname(__file__))
__PATH__ = '../datasets/moving_mnist'

rs = np.random.RandomState(123)


class DataLoader(object):
    ''' Operations on moving MNIST dataset '''

    def __init__(self, ids, configs, name='default', max_examples=None, is_train=True):
        self.configs = configs
        self.data_file = os.path.join(__PATH__, self.configs.data_file)

        self.ids = ids
        self.name = name
        self.is_train = is_train

        log.info('DATALOADER: ' + name)
        log.info('Loading data ...')
        self.load_data()
        log.info('Loading done')

        if max_examples is not None:
            self.ids = self.ids[:max_examples]

    def load_data(self):
        try:
            self.raw_data = np.load(self.data_file).astype(np.float32)
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')

        # ids_int = [int(x) for x in self.ids]
        # self.data = self.raw_data[:, ids_int, :, :]
        self.data = self.raw_data
        self.num_data_frames, self.num_samples, self.image_height, self.image_width = self.data.shape
        # assert(self.num_samples == len(self.ids))

        assert(self.configs.crop_height <= self.image_height)
        assert(self.configs.crop_width <= self.image_width)
        if(self.image_height != self.configs.crop_height or self.image_width != self.configs.crop_width):
            height_low = (self.image_height - self.configs.crop_height) / 2
            height_high = (self.image_height + self.configs.crop_height) / 2
            width_low = (self.image_width - self.configs.crop_width) / 2
            width_high = (self.image_width + self.configs.crop_width) / 2
            self.data = self.data[:, :, height_low:height_high, width_low:width_high]

        self.num_frames = self.configs.data_info.num_frames
        assert(self.num_data_frames >= 2 * self.num_frames)
        self.current_data = self.data[0:self.num_frames, :, :, :]
        self.future_data = self.data[self.num_frames: 2 * self.num_frames, :, :, :]
        self.current_data = self.current_data.reshape(self.num_samples, self.num_frames, self.configs.crop_height, self.configs.crop_width, 1)
        self.future_data = self.future_data.reshape(self.num_samples, self.num_frames, self.configs.crop_height, self.configs.crop_width, 1)
        print ('Current data shape:', self.current_data.shape)
        print ('Future data shape:', self.future_data.shape)

    def get_data(self, id):
        id_int = int(id)
        current_frames = self.current_data[id_int]
        future_frames = self.future_data[id_int]
        return id, current_frames, future_frames

    @property
    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return ('Dataset ({}, {} examples)'.format(self.name, len(self.ids)))


def load_configs():
    config_filename = os.path.join(os.path.abspath('.'), 'configs', 'moving_mnist.json')
    print ('Config file:', config_filename)
    configs = from_json_file(config_filename)
    return configs


def load_ids():
    ids_filename = 'ids.txt'
    ids_filename_abs = os.path.join(__PATH__, ids_filename)
    print ('Ids file:', ids_filename_abs)

    try:
        with open(ids_filename_abs, 'r') as fp:
            ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Id file not found. Please make sure theat id file exists.')

    log.info('Completed reading ids')
    rs.shuffle(ids)
    return ids


def create_default_splits(configs, train_fraction=0.8):
    ids = load_ids()
    num_samples = len(ids)
    num_trains = int(train_fraction * num_samples)

    dataset_train = DataLoader(ids[:num_trains], configs, name='train', is_train=True)
    dataset_test = DataLoader(ids[num_trains:], configs, name='test', is_train=False)
    return dataset_train, dataset_test
