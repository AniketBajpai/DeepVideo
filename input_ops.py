from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import numpy as np
import tensorflow as tf

from utils import log


def check_data_id(dataset, data_ids):
    ''' Check if data ids are valid '''
    pass


def create_input_ops(dataset, batch_size, is_train, num_threads=16, data_ids=None, scope='inputs', shuffle=True):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''
    input_ops = {}

    if data_ids is None:
        data_ids = dataset.ids
        log.info('input_ops [%s]: Using %d IDs from dataset', scope, len(data_ids))
    else:
        log.info('input_ops [%s]: Using specified %d IDs', scope, len(data_ids))

    # single operations
    with tf.device('/cpu:0'), tf.name_scope(scope):
        input_ops['id'] = tf.train.string_input_producer(
            tf.convert_to_tensor(data_ids),
            capacity=128
        ).dequeue(name='input_ids_dequeue')

        def load_fn(id):
            id, current_frames, future_frames = dataset.get_data(id)
            return (id, current_frames, future_frames)

        input_ops['id'], input_ops['current_frames'], input_ops['future_frames'] = tf.py_func(
            load_fn, inp=[input_ops['id']],
            Tout=[tf.string, tf.float32, tf.float32],
            name='func_hp'
        )

        sample_id, sample_current_frames, sample_future_frames = dataset.get_data(data_ids[0])
        input_ops['id'].set_shape([])
        input_ops['current_frames'].set_shape(list(sample_current_frames.shape))
        input_ops['future_frames'].set_shape(list(sample_future_frames.shape))

    # batchify
    capacity = 2 * batch_size * num_threads
    min_capacity = min(int(capacity * 0.75), 1024)

    if shuffle:
        batch_ops = tf.train.shuffle_batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_capacity,
        )
    else:
        batch_ops = tf.train.batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
        )

    return input_ops, batch_ops
