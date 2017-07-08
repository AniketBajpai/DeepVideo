from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from operator import mul

from ops import conv3d, deconv2d, deconv3d, linear, add_gaussian_noise
from utils import print_message


class Encoder(object):
    '''
    Encoder network to map videos(fixed size) to latent space
    Input: Video having fixed number of frames(16)
    Output: Video encoding in latent space
    '''

    def __init__(self, name, configs_encoder):
        self.name = name
        self.configs = configs_encoder
        self.batch_size = configs_encoder.batch_size
        self.latent_dimension = configs_encoder.latent_dimension
        self.net = {}

    def __call__(self, inputs, is_train=True, is_debug=False):
        self.is_train = is_train
        self.is_debug = is_debug

        inputs = tf.convert_to_tensor(inputs)   # Check if necessary

        # Assert that input is in [-1, 1]
        encoder_max_assert_op = tf.Assert(tf.less_equal(tf.reduce_max(inputs), 1.), [
                                          inputs], summarize=0, name='assert/encoder_max')
        encoder_min_assert_op = tf.Assert(tf.greater_equal(tf.reduce_max(inputs), -1.),
                                          [inputs], summarize=0, name='assert/encoder_min')
        tf.add_to_collection('Assert', encoder_max_assert_op)
        tf.add_to_collection('Assert', encoder_min_assert_op)

        assert(inputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.input)
        with tf.variable_scope(self.name) as scope:
            print_message(scope.name)
            with tf.variable_scope('conv1') as vscope:
                outputs, self.net['w1'], self.net['b1'] = conv3d(
                    inputs, [self.batch_size] + self.configs.conv_info.l1, is_train=self.is_train,
                    k=self.configs.conv_info.k1, s=self.configs.conv_info.s1, with_w=True)
                if is_debug:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l1)
                self.net['conv1_outputs'] = outputs
            with tf.variable_scope('conv2') as vscope:
                outputs, self.net['w2'], self.net['b2'] = conv3d(
                    outputs, [self.batch_size] + self.configs.conv_info.l2, is_train=self.is_train,
                    k=self.configs.conv_info.k2, s=self.configs.conv_info.s2, with_w=True)
                if is_debug:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l2)
                self.net['conv2_outputs'] = outputs
            with tf.variable_scope('conv3') as vscope:
                outputs, self.net['w3'], self.net['b3'] = conv3d(
                    outputs, [self.batch_size] + self.configs.conv_info.l3, is_train=self.is_train,
                    k=self.configs.conv_info.k3, s=self.configs.conv_info.s3, with_w=True)
                if is_debug:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l3)
                self.net['conv3_outputs'] = outputs
            with tf.variable_scope('fc') as vscope:
                fc_dim = reduce(mul, self.configs.conv_info.l3, 1)
                outputs = tf.reshape(outputs, [self.batch_size] + [fc_dim], name='reshape')
                outputs = linear(outputs, self.latent_dimension)
                outputs = tf.nn.relu(outputs)
                if is_debug:
                    print(vscope.name, outputs)
                self.net['fc_outputs'] = outputs

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return outputs

    def build_summary(self):
        # Distribution of encoder activations
        tf.summary.histogram('encoder/conv1_outputs', self.net['conv1_outputs'])
        tf.summary.histogram('encoder/conv2_outputs', self.net['conv2_outputs'])
        tf.summary.histogram('encoder/conv3_outputs', self.net['conv3_outputs'])

        # Encoder weights, biases
        tf.summary.scalar('encoder/w1', tf.norm(self.net['w1']))
        tf.summary.scalar('encoder/w2', tf.norm(self.net['w2']))
        tf.summary.scalar('encoder/w3', tf.norm(self.net['w3']))

        tf.summary.scalar('encoder/b1', tf.norm(self.net['b1']))
        tf.summary.scalar('encoder/b2', tf.norm(self.net['b2']))
        tf.summary.scalar('encoder/b3', tf.norm(self.net['b3']))


class Generator(object):
    '''
    Generator network to generate videos(fixed size) from latent variable
    Input: Latent variable in embedding space
    Output: Video having same no. of frames as encoder(16)
    '''

    def __init__(self, name, configs_generator):
        self.name = name
        self.configs = configs_generator
        self.batch_size = configs_generator.batch_size
        self.latent_dimension = configs_generator.latent_dimension
        self.net = {}

    def __call__(self, inputs, is_train=True, is_debug=False):
        self.is_train = is_train
        self.is_debug = is_debug

        inputs = tf.convert_to_tensor(inputs)   # Check if necessary
        assert(inputs.get_shape().as_list() == [self.batch_size, self.latent_dimension])
        with tf.variable_scope(self.name) as scope:
            print_message(scope.name)
            # Foreground generator
            with tf.variable_scope('fc_f') as vscope:
                fc_dim = reduce(mul, self.configs.deconv_f_info.l1, 1)
                outputs_f = linear(inputs, fc_dim)
                outputs_f = tf.nn.relu(outputs_f)
                if is_debug:
                    print(vscope.name, outputs_f)
                assert(outputs_f.get_shape().as_list() == [self.batch_size, fc_dim])
                outputs_f = tf.layers.dropout(outputs_f, rate=self.configs.dropout, training=self.is_train, name='dropout')
                outputs_f = tf.reshape(outputs_f, [self.batch_size] + self.configs.deconv_f_info.l1, name='reshape')
                self.net['f_fc_outputs'] = outputs_f
            with tf.variable_scope('deconv2_f') as vscope:
                k2 = self.configs.deconv_f_info.k2
                s2 = self.configs.deconv_f_info.s2
                k2_d = self.configs.deconv_f_info.k2_d
                s2_d = self.configs.deconv_f_info.s2_d
                outputs_f, self.net['w2_f'], self.net['b2_f'] = deconv3d(
                    outputs_f, [self.batch_size] + self.configs.deconv_f_info.l2, is_train=self.is_train,
                    k=(k2_d, k2, k2), s=(s2_d, s2, s2), padding='VALID', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_f)
                outputs_f = tf.layers.dropout(outputs_f, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs_f.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l2)
                self.net['f_deconv2_outputs'] = outputs_f
            with tf.variable_scope('deconv3_f') as vscope:
                k3 = self.configs.deconv_f_info.k3
                s3 = self.configs.deconv_f_info.s3
                k3_d = self.configs.deconv_f_info.k3_d
                s3_d = self.configs.deconv_f_info.s3_d
                outputs_f, self.net['w3_f'], self.net['b3_f'] = deconv3d(
                    outputs_f, [self.batch_size] + self.configs.deconv_f_info.l3, is_train=self.is_train,
                    k=(k3_d, k3, k3), s=(s3_d, s3, s3), padding='VALID', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_f)
                outputs_f = tf.layers.dropout(outputs_f, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs_f.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l3)
                self.net['f_deconv3_outputs'] = outputs_f
            with tf.variable_scope('deconv4_f') as vscope:
                k4 = self.configs.deconv_f_info.k4
                s4 = self.configs.deconv_f_info.s4
                k4_d = self.configs.deconv_f_info.k4_d
                s4_d = self.configs.deconv_f_info.s4_d
                outputs_f, self.net['w4_f'], self.net['b4_f'] = deconv3d(
                    outputs_f, [self.batch_size] + self.configs.deconv_f_info.l4, is_train=self.is_train,
                    k=(k4_d, k4, k4), s=(s4_d, s4, s4), padding='VALID', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_f)
                outputs_f = tf.layers.dropout(outputs_f, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs_f.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l4)
                self.net['f_deconv4_outputs'] = outputs_f
            with tf.variable_scope('deconv5_fi') as vscope:
                k5 = self.configs.deconv_f_info.k5
                s5 = self.configs.deconv_f_info.s5
                k5_d = self.configs.deconv_f_info.k5_d
                s5_d = self.configs.deconv_f_info.s5_d
                outputs_fi, self.net['w5_fi'], self.net['b5_fi'] = deconv3d(
                    outputs_f, [self.batch_size] + self.configs.deconv_f_info.l5_i, is_train=self.is_train,
                    k=(k5_d, k5, k5), s=(s5_d, s5, s5), padding='SAME', activation_fn='tanh', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_fi)
                assert(outputs_fi.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l5_i)
                self.net['f_deconv5i_outputs'] = outputs_fi
            with tf.variable_scope('deconv5_fm') as vscope:
                k5 = self.configs.deconv_f_info.k5
                s5 = self.configs.deconv_f_info.s5
                k5_d = self.configs.deconv_f_info.k5_d
                s5_d = self.configs.deconv_f_info.s5_d
                outputs_fm, self.net['w5_fm'], self.net['b5_fm'] = deconv3d(
                    outputs_f, [self.batch_size] + self.configs.deconv_f_info.l5_m, is_train=self.is_train,
                    k=(k5_d, k5, k5), s=(s5_d, s5, s5), padding='SAME', activation_fn='sigmoid', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_fm)
                assert(outputs_fm.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l5_m)
                self.net['f_deconv5m_outputs'] = outputs_fm

            # Background generator
            with tf.variable_scope('fc_b') as vscope:
                fc_dim = reduce(mul, self.configs.deconv_b_info.l1, 1)
                outputs_b = linear(inputs, fc_dim)
                if is_debug:
                    print(vscope.name, outputs_b)
                assert(outputs_b.get_shape().as_list() == [self.batch_size, fc_dim])
                outputs_b = tf.layers.dropout(outputs_b, rate=self.configs.dropout, training=self.is_train, name='dropout')
                outputs_b = tf.reshape(outputs_b, [self.batch_size] + self.configs.deconv_b_info.l1, name='reshape')
                self.net['b_fc_outputs'] = outputs_b
            with tf.variable_scope('deconv2_b') as vscope:
                outputs_b, self.net['w2_b'], self.net['b2_b'] = deconv2d(
                    outputs_b, [self.batch_size] + self.configs.deconv_b_info.l2, is_train=self.is_train,
                    k=self.configs.deconv_b_info.k2, s=self.configs.deconv_f_info.s2, padding='VAID', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_b)
                outputs_b = tf.layers.dropout(outputs_b, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs_b.get_shape().as_list() == [self.batch_size] + self.configs.deconv_b_info.l2)
                self.net['b_deconv2_outputs'] = outputs_b
            with tf.variable_scope('deconv3_b') as vscope:
                outputs_b, self.net['w3_b'], self.net['b3_b'] = deconv2d(
                    outputs_b, [self.batch_size] + self.configs.deconv_b_info.l3, is_train=self.is_train,
                    k=self.configs.deconv_b_info.k3, s=self.configs.deconv_f_info.s3, padding='VAID', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_b)
                outputs_b = tf.layers.dropout(outputs_b, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs_b.get_shape().as_list() == [self.batch_size] + self.configs.deconv_b_info.l3)
                self.net['b_deconv3_outputs'] = outputs_b
            with tf.variable_scope('deconv4_b') as vscope:
                outputs_b, self.net['w4_b'], self.net['b4_b'] = deconv2d(
                    outputs_b, [self.batch_size] + self.configs.deconv_b_info.l4, is_train=self.is_train,
                    k=self.configs.deconv_b_info.k4, s=self.configs.deconv_f_info.s4, padding='VAID', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_b)
                outputs_b = tf.layers.dropout(outputs_b, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs_b.get_shape().as_list() == [self.batch_size] + self.configs.deconv_b_info.l4)
                self.net['b_deconv4_outputs'] = outputs_b
            with tf.variable_scope('deconv5_b') as vscope:
                outputs_b, self.net['w5_b'], self.net['b5_b'] = deconv2d(
                    outputs_b, [self.batch_size] + self.configs.deconv_b_info.l5, is_train=self.is_train,
                    k=self.configs.deconv_b_info.k5, s=self.configs.deconv_f_info.s5, padding='SAME',  activation_fn='tanh', with_w=True)
                if is_debug:
                    print(vscope.name, outputs_b)
                assert(outputs_b.get_shape().as_list() == [self.batch_size] + self.configs.deconv_b_info.l5)
                self.net['b_deconv5_outputs'] = outputs_b

            # Construct output video from forground, background, mask
            outputs_b = tf.reshape(outputs_b, [self.batch_size, 1] + self.configs.deconv_b_info.l5)
            outputs_b_vol = tf.tile(outputs_b, [1, self.configs.num_frames, 1, 1, 1])
            outputs = outputs_fm * outputs_fi + (1 - outputs_fm) * outputs_b_vol

            # Assert that frames are in [-1, 1]
            generator_max_assert_op = tf.Assert(tf.less_equal(tf.reduce_max(outputs), 1.),
                                                [outputs], summarize=0, name='assert/generator_max')
            generator_min_assert_op = tf.Assert(tf.greater_equal(tf.reduce_max(outputs), -1.),
                                                [outputs], summarize=0, name='assert/generator_min')
            tf.add_to_collection('Assert', generator_max_assert_op)
            tf.add_to_collection('Assert', generator_min_assert_op)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return outputs

    def build_summary(self, name):
        # Distribution of generator activations
        tf.summary.histogram('generator/{}/f_deconv2_outputs'.format(name), self.net['f_deconv2_outputs'])
        tf.summary.histogram('generator/{}/f_deconv3_outputs'.format(name), self.net['f_deconv3_outputs'])
        tf.summary.histogram('generator/{}/f_deconv4_outputs'.format(name), self.net['f_deconv4_outputs'])
        tf.summary.histogram('generator/{}/f_deconv5i_outputs'.format(name), self.net['f_deconv5i_outputs'])
        tf.summary.histogram('generator/{}/f_deconv5m_outputs'.format(name), self.net['f_deconv5m_outputs'])
        tf.summary.histogram('generator/{}/b_deconv2_outputs'.format(name), self.net['b_deconv2_outputs'])
        tf.summary.histogram('generator/{}/b_deconv3_outputs'.format(name), self.net['b_deconv3_outputs'])
        tf.summary.histogram('generator/{}/b_deconv4_outputs'.format(name), self.net['b_deconv4_outputs'])
        tf.summary.histogram('generator/{}/b_deconv5_outputs'.format(name), self.net['b_deconv5_outputs'])

        # Generator weights, biases
        tf.summary.scalar('generator/{}/w2_f'.format(name), tf.norm(self.net['w2_f']))
        tf.summary.scalar('generator/{}/w3_f'.format(name), tf.norm(self.net['w3_f']))
        tf.summary.scalar('generator/{}/w4_f'.format(name), tf.norm(self.net['w4_f']))
        tf.summary.scalar('generator/{}/w5_fi'.format(name), tf.norm(self.net['w5_fi']))
        tf.summary.scalar('generator/{}/w5_fm'.format(name), tf.norm(self.net['w5_fm']))
        tf.summary.scalar('generator/{}/w2_b'.format(name), tf.norm(self.net['w2_b']))
        tf.summary.scalar('generator/{}/w3_b'.format(name), tf.norm(self.net['w3_b']))
        tf.summary.scalar('generator/{}/w4_b'.format(name), tf.norm(self.net['w4_b']))
        tf.summary.scalar('generator/{}/w5_b'.format(name), tf.norm(self.net['w5_b']))

        tf.summary.scalar('generator/{}/b2_f'.format(name), tf.norm(self.net['b2_f']))
        tf.summary.scalar('generator/{}/b3_f'.format(name), tf.norm(self.net['b3_f']))
        tf.summary.scalar('generator/{}/b4_f'.format(name), tf.norm(self.net['b4_f']))
        tf.summary.scalar('generator/{}/b5_fi'.format(name), tf.norm(self.net['b5_fi']))
        tf.summary.scalar('generator/{}/b5_fm'.format(name), tf.norm(self.net['b5_fm']))
        tf.summary.scalar('generator/{}/b2_b'.format(name), tf.norm(self.net['b2_b']))
        tf.summary.scalar('generator/{}/b3_b'.format(name), tf.norm(self.net['b3_b']))
        tf.summary.scalar('generator/{}/b4_b'.format(name), tf.norm(self.net['b4_b']))
        tf.summary.scalar('generator/{}/b5_b'.format(name), tf.norm(self.net['b5_b']))


class Discriminator(object):
    '''
    Discriminator network to classify videos as real/generated
    Input: Video having fixed number of frames(16)
    Output: (probability, logit) of FAKE
    '''

    def __init__(self, name, configs_discriminator):
        self.name = name
        self.configs = configs_discriminator
        self.batch_size = configs_discriminator.batch_size
        self.net = {}

    def __call__(self, inputs, reuse, is_train=True, is_debug=False):
        self.is_train = is_train
        self.is_debug = is_debug

        inputs = tf.convert_to_tensor(inputs)   # Check if necessary

        # Assert that input is in [-1, 1] - removed due to instance noise
        # discriminator_max_assert_op = tf.Assert(tf.less_equal(tf.reduce_max(inputs), 1.), [
        #                                         inputs], summarize=0, name='assert/discriminator_max')
        # discriminator_min_assert_op = tf.Assert(tf.greater_equal(tf.reduce_max(
        #     inputs), -1.), [inputs], summarize=0, name='assert/discriminator_min')
        # tf.add_to_collection('Assert', discriminator_max_assert_op)
        # tf.add_to_collection('Assert', discriminator_min_assert_op)

        assert(inputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.input)
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            print_message(scope.name)
            with tf.variable_scope('conv1') as vscope:
                outputs, self.net['w1'], self.net['b1'] = conv3d(
                    inputs, [self.batch_size] + self.configs.conv_info.l1, is_train=self.is_train,
                    k=self.configs.conv_info.k1, s=self.configs.conv_info.s1, with_w=True)
                if is_debug:
                    print(vscope.name, outputs)
                outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l1)
                self.net['conv1_outputs'] = outputs
            with tf.variable_scope('conv2') as vscope:
                outputs, self.net['w2'], self.net['b2'] = conv3d(
                    outputs, [self.batch_size] + self.configs.conv_info.l2, is_train=self.is_train,
                    k=self.configs.conv_info.k2, s=self.configs.conv_info.s2, with_w=True)
                if is_debug:
                    print(vscope.name, outputs)
                outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l2)
                self.net['conv2_outputs'] = outputs
            with tf.variable_scope('conv3') as vscope:
                outputs, self.net['w3'], self.net['b3'] = conv3d(
                    outputs, [self.batch_size] + self.configs.conv_info.l3, is_train=self.is_train,
                    k=self.configs.conv_info.k3, s=self.configs.conv_info.s3, with_w=True)
                if is_debug:
                    print(vscope.name, outputs)
                outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='dropout')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l3)
                self.net['conv3_outputs'] = outputs
            with tf.variable_scope('fc') as vscope:
                fc_dim = reduce(mul, self.configs.conv_info.l3, 1)
                outputs = tf.reshape(outputs, [self.batch_size] + [fc_dim], name='reshape')
                outputs = linear(outputs, 1)
                if is_debug:
                    print(vscope.name, outputs)
                self.net['fc_outputs'] = outputs

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return tf.nn.sigmoid(outputs), outputs

    def build_summary(self):
        # Distribution of encoder activations
        tf.summary.histogram('discriminator/conv1_outputs', self.net['conv1_outputs'])
        tf.summary.histogram('discriminator/conv2_outputs', self.net['conv2_outputs'])
        tf.summary.histogram('discriminator/conv3_outputs', self.net['conv3_outputs'])

        # Encoder weights, biases
        tf.summary.scalar('discriminator/w1', tf.norm(self.net['w1']))
        tf.summary.scalar('discriminator/w2', tf.norm(self.net['w2']))
        tf.summary.scalar('discriminator/w3', tf.norm(self.net['w3']))

        tf.summary.scalar('discriminator/b1', tf.norm(self.net['b1']))
        tf.summary.scalar('discriminator/b2', tf.norm(self.net['b2']))
        tf.summary.scalar('discriminator/b3', tf.norm(self.net['b3']))


class Model:
    ''' Overall model '''

    def __init__(self, configs, is_train=True, is_debug=False):
        self.configs = configs
        self.is_debug = is_debug

        # Model info
        self.configs_encoder = configs.configs_encoder
        self.configs_generator = configs.configs_generator
        self.configs_discriminator = configs.configs_discriminator

        self.configs_encoder.batch_size = configs.batch_size
        self.configs_encoder.num_frames = configs.data_info.num_frames
        self.configs_encoder.latent_dimension = configs.latent_dimension
        self.configs_generator.batch_size = configs.batch_size
        self.configs_generator.num_frames = configs.data_info.num_frames
        self.configs_generator.latent_dimension = configs.latent_dimension
        self.configs_discriminator.batch_size = configs.batch_size

        # Data info
        self.num_frames = configs.data_info.num_frames
        self.image_height = configs.data_info.image_height
        self.image_width = configs.data_info.image_width
        self.num_channels = configs.data_info.num_channels
        # self.num_classes = configs.data_info.num_classes

        self.latent_dimension = configs.latent_dimension
        self.lr_ae = configs.learner_hyperparameters.lr_ae
        self.lr_d = configs.learner_hyperparameters.lr_d
        self.beta1 = configs.learner_hyperparameters.beta1

        self.batch_size = configs.batch_size
        self.dataset_name = configs.dataset

        # Build model, loss, and summary
        self.build_model(is_train)
        self.build_loss()
        self.build_summary()

    def get_feed_dict(self, batch_chunk, step=None, is_train=True):
        ''' Organize data into a feed dictionary '''
        fd = {
            self.current_frames: batch_chunk['current_frames'],
            # self.future_frames: batch_chunk['future_frames'],
            # self.label: batch_chunk['label'],
        }

        # latent variable - (-1, 1 chosen as encoder has relu activation)
        fd[self.z] = np.random.uniform(-1, 1, [self.batch_size, self.latent_dimension]).astype(np.float32)

        # TODO: add weight annealing

        fd[self.is_train] = is_train

        return fd

    def build_model(self, is_train=True):
        ''' Build model '''

        # Placeholders for data
        self.current_frames = tf.placeholder(
            name='current_frames', dtype=tf.float32,
            shape=[self.batch_size, self.num_frames, self.image_height, self.image_width, self.num_channels]
        )
        # self.future_frames = tf.placeholder(
        #     name='future_frames', dtype=tf.float32,
        #     shape=[self.batch_size, self.num_frames, self.image_height, self.image_width, self.num_channels]
        # )
        # self.label = tf.placeholder(
        #     name='label', dtype=tf.float32, shape=[self.batch_size, self.num_classes]
        # )

        self.is_train = tf.placeholder_with_default(bool(is_train), [], name='is_train')

        # Encoder
        # self.E = Encoder('Encoder', self.configs_encoder)
        # self.z = self.E(self.current_frames, is_debug=self.is_debug)
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.latent_dimension], name='z')

        # Generators
        self.Gr = Generator('Generator_R', self.configs_generator)
        # self.Gf = Generator('Generator_F', self.configs_generator)

        self.generated_current_frames = self.Gr(self.z, is_debug=self.is_debug)
        # self.generated_future_frames = self.Gf(self.z, is_debug=self.is_debug)

        # # Add instance noise
        # instance_noise_spread = self.configs.instance_noise_spread
        # self.generated_current_frames_noisy = add_gaussian_noise(self.generated_current_frames, instance_noise_spread)
        # self.current_frames_noisy = add_gaussian_noise(self.generated_current_frames, instance_noise_spread)

        # Discriminators
        self.D = Discriminator('Discriminator', self.configs_discriminator)

        self.D_real_current, self.D_real_current_logits = self.D(
            self.current_frames, reuse=False, is_debug=self.is_debug)
        self.D_fake_current, self.D_fake_current_logits = self.D(
            self.generated_current_frames, reuse=True, is_debug=self.is_debug)
        # self.D_real_future, self.D_real_future_logits = self.D(self.future_frames, is_debug=self.is_debug)
        # self.D_fake_future, self.D_fake_future_logits = self.D(self.generated_future_frames, is_debug=self.is_debug)

        self.all_predictions = self.D_fake_current

        print_message('Successfully loaded the model')

    def build_loss(self):
        ''' Build model loss and accuracy '''
        self.loss = {}

        # Reconstruction loss

        # L2 loss
        # self.loss['input_reconstruction_loss_mse'] = tf.reduce_mean(
        #     tf.nn.l2_loss(self.generated_current_frames - self.current_frames))
        # self.loss['future_reconstruction_loss_mse'] = tf.reduce_mean(
        #     tf.nn.l2_loss(self.generated_future_frames - self.future_frames))

        # self.loss['input_reconstruction_loss'] = self.loss['input_reconstruction_loss_mse']
        # self.loss['future_reconstruction_loss'] = self.loss['future_reconstruction_loss_mse']

        # Adversarial loss

        # Label smoothing
        label_noise_spread = self.configs.label_noise_spread
        mean_real = label_noise_spread / 2.0
        mean_fake = 1.0 - label_noise_spread / 2.0
        stddev_real = stddev_fake = label_noise_spread / 2.0
        label_real_current = tf.random_normal([self.batch_size, 1], mean=mean_real, stddev=stddev_real, dtype=tf.float32)
        # label_real_future = tf.zeros([self.batch_size, 1])
        label_fake_current = tf.random_normal([self.batch_size, 1], mean=mean_fake, stddev=stddev_fake, dtype=tf.float32)
        # label_fake_future = tf.ones([self.batch_size, 1])

        # Generator
        self.loss['generator_current'] = tf.reduce_mean(tf.log(self.D_fake_current))
        # self.loss['generator_future'] = tf.reduce_mean(tf.log(self.D_fake_future))
        self.loss['autoencoder'] = self.loss['generator_current']   # + self.configs.reconstruction_weight * self.loss['input_reconstruction_loss']
        # + self.loss['future_reconstruction_loss']
        # + self.loss['generator_future']

        # Discriminator adversarial loss
        self.loss['discriminator_real_current'] = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label_real_current, logits=self.D_real_current_logits))
        self.loss['discriminator_fake_current'] = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label_fake_current, logits=self.D_fake_current_logits))
        # Add disciminator terms for future_frames
        self.loss['discriminator'] = self.loss['discriminator_real_current'] + self.loss['discriminator_fake_current']
        # + self.loss['discriminator_real_future']
        # + self.loss['discriminator_fake_future']

        # # Classification accuracy - for supervised
        # self.accuracy

    def build_summary(self):
        ''' Build summary for model '''

        # Distribution of latent variable
        tf.summary.histogram('latent', self.z)

        # Build encoder summary
        # self.E.build_summary()

        # Build generator(s) summary
        self.Gr.build_summary('current')
        # self.Gf.build_summary('future')

        # Build discriminator summary
        self.D.build_summary()

        # Loss summary
        # tf.summary.scalar('loss/input_reconstruction_loss', self.loss['input_reconstruction_loss'])
        # tf.summary.scalar('loss/future_reconstruction_loss', self.loss['future_reconstruction_loss'])
        tf.summary.scalar('loss/generator_current', self.loss['generator_current'])
        tf.summary.scalar('loss/autoencoder', self.loss['autoencoder'])

        tf.summary.scalar('loss/discriminator_real_current', self.loss['discriminator_real_current'])
        tf.summary.scalar('loss/discriminator_fake_current', self.loss['discriminator_fake_current'])
        tf.summary.scalar('loss/discriminator', self.loss['discriminator'])

        # Input data summary
        input_current_summary = tf.reshape(self.current_frames,
                                           (-1, self.num_frames * self.image_height, self.image_width, self.num_channels))
        tf.summary.image('input/current', input_current_summary)
        # input_future_summary = tf.reshape(self.future_frames,
        #                                   (-1, self.num_frames * self.image_height, self.image_width, self.num_channels))
        # tf.summary.image('input/future', input_future_summary)

        # TODO: refactor inside generator summary
        # Generated data summary
        generated_current_summary = tf.reshape(self.generated_current_frames,
                                               (-1, self.num_frames * self.image_height, self.image_width, self.num_channels))
        tf.summary.image('generated/current', generated_current_summary)
        # generated_future_summary = tf.reshape(self.generated_future_frames,
        #                                       (-1, self.num_frames * self.image_height, self.image_width, self.num_channels))
        # tf.summary.image('generated/future', generated_future_summary)

        # Label summary
        tf.summary.scalar('label/real_current', tf.reduce_mean(self.D_real_current))
        tf.summary.scalar('label/fake_current', tf.reduce_mean(self.D_fake_current))
        tf.summary.image('label/pred_real_current', tf.reshape(self.D_real_current, [1, self.batch_size, 1, 1]))
        tf.summary.image('label/pred_fake_current', tf.reshape(self.D_fake_current, [1, self.batch_size, 1, 1]))
