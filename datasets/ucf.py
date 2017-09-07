from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np

from utils import log, from_json_file

# __PATH__ = os.path.abspath(os.path.dirname(__file__))
__PATH__ = '../datasets/ucf'

rs = np.random.RandomState(123)


class DataLoader(object):
    ''' Operations on UCF101 dataset '''

    def __init__(self, ids, configs, name='default', max_examples=None, is_train=True):
        self.configs = configs

        self.ids = ids
        self.name = name
        self.is_train = is_train

        # reading data list
        self.data_root = "../.."
        self.data_list_path = os.path.join(self.data_root, 'list.txt')
        print('reading data list...')
        with open(self.data_list_path, 'r') as f:
            self.video_index = [x.strip() for x in f.readlines()]
            np.random.shuffle(self.video_index)
        self.size = len(self.video_index)
        self.cursor = 0

        log.info('DATALOADER: ' + name)
        log.info('Loading data ...')
        self.load_data()
        log.info('Loading done')

        if max_examples is not None:
            self.ids = self.ids[:max_examples]

    def load_data(self):
        if self.cursor + self.configs.batch_size > self.configs.size:
            self.cursor = 0
            np.random.shuffle(self.video_index)
        out = np.zeros((self.configs.batch_size, self.configs.frame_size,
                        self.configs.crop_size, self.configs.crop_size, 3))
        for idx in xrange(self.configs.batch_size):
            video_path = self.video_index[self.cursor]
            self.cursor += 1
            inputimage = cv2.imread(video_path)
            count = inputimage.shape[0] / self.configs.image_size
            for j in xrange(self.configs.frame_size):
                if j < count:
                    cut = j * self.configs.image_size
                else:
                    cut = (count - 1) * self.configs.image_size
                crop = inputimage[cut: cut + self.configs.image_size, :]
                out[idx, j, :, :, :] = cv2.resize(
                    crop, (self.configs.crop_size, self.configs.crop_size))

        out = out / 255.0 * 2 - 1

        return out


def load_configs(config_filename=None):
    if config_filename is None:
        config_filename = os.path.join(os.path.abspath('.'), 'configs', 'ucf101.json')
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
