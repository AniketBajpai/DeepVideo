from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from operator import mul

from ops import conv3d, deconv2d, deconv3d, linear
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
        self.reuse = False
        self.net = {}

    def __call__(self, inputs, is_train=True, is_debug=False):
        self.is_train = is_train
        self.is_debug = is_debug

        outputs = tf.convert_to_tensor(inputs)   # Check if necessary
        tf.Assert(tf.less_equal(tf.reduce_max(outputs), 1.), [outputs], summarize=0, name='encoder_max_assert')
        tf.Assert(tf.greater_equal(tf.reduce_max(outputs), -1.), [outputs], summarize=0, name='encoder_min_assert')

        assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.input)
        with tf.variable_scope(self.name, reuse=self.reuse) as scope:
            print_message(scope.name)
            with tf.variable_scope('conv1') as vscope:
                outputs = conv3d(outputs, [self.batch_size] + self.configs.conv_info.l1)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l1)
                self.net['conv1_outputs'] = outputs
            with tf.variable_scope('conv2') as vscope:
                outputs = conv3d(outputs, [self.batch_size] + self.configs.conv_info.l2)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l2)
                self.net['conv2_outputs'] = outputs
            with tf.variable_scope('conv3') as vscope:
                outputs = conv3d(outputs, [self.batch_size] + self.configs.conv_info.l3)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                assert(outputs.get_shape().as_list() == [self.batch_size] + self.configs.conv_info.l3)
                self.net['conv3_outputs'] = outputs
            with tf.variable_scope('fc') as vscope:
                fc_dim = reduce(mul, self.configs.conv_info.l3, 1)
                outputs = tf.reshape(outputs, [self.batch_size] + [fc_dim], name='reshape')
                outputs = linear(outputs, self.latent_dimension)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs)
                self.net['fc_outputs'] = outputs

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return outputs


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
        self.reuse = False
        self.net = {}

    def __call__(self, inputs, is_train=True, is_debug=False):
        self.is_train = is_train
        self.is_debug = is_debug

        inputs = tf.convert_to_tensor(inputs)   # Check if necessary
        assert(inputs.get_shape().as_list() == [self.batch_size, self.latent_dimension])
        with tf.variable_scope(self.name, reuse=self.reuse) as scope:
            print_message(scope.name)
            # Foreground generator
            with tf.variable_scope('fc_f') as vscope:
                fc_dim = reduce(mul, self.configs.deconv_f_info.l1, 1)
                outputs_f = linear(inputs, fc_dim)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_f)
                assert(outputs_f.get_shape().as_list() == [self.batch_size, fc_dim])
                # outputs_f = tf.layers.dropout(outputs_f, rate=self.configs.dropout, training=self.is_train, name='outputs_f')
                outputs_f = tf.reshape(outputs_f, [self.batch_size] + self.configs.deconv_f_info.l1, name='reshape')
                self.net['f_fc_outputs'] = outputs_f
            with tf.variable_scope('deconv2_f') as vscope:
                outputs_f = deconv3d(outputs_f, [self.batch_size] + self.configs.deconv_f_info.l2)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_f)
                # outputs_f = tf.layers.dropout(outputs_f, rate=self.configs.dropout, training=self.is_train, name='outputs_f')
                assert(outputs_f.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l2)
                self.net['f_deconv2_outputs'] = outputs_f
            with tf.variable_scope('deconv3_f') as vscope:
                outputs_f = deconv3d(outputs_f, [self.batch_size] + self.configs.deconv_f_info.l3)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_f)
                # outputs_f = tf.layers.dropout(outputs_f, rate=self.configs.dropout, training=self.is_train, name='outputs_f')
                assert(outputs_f.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l3)
                self.net['f_deconv3_outputs'] = outputs_f
            with tf.variable_scope('deconv4_fi') as vscope:
                outputs_fi = deconv3d(outputs_f, [self.batch_size] + self.configs.deconv_f_info.l4_i)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_fi)
                assert(outputs_fi.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l4_i)
                outputs_fi = tf.nn.tanh(outputs_fi)
                self.net['f_deconv4i_outputs'] = outputs_fi
            with tf.variable_scope('deconv4_fm') as vscope:
                outputs_fm = deconv3d(outputs_f, [self.batch_size] + self.configs.deconv_f_info.l4_m)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_fm)
                assert(outputs_fm.get_shape().as_list() == [self.batch_size] + self.configs.deconv_f_info.l4_m)
                outputs_fm = tf.nn.sigmoid(outputs_fm)
                self.net['f_deconv4m_outputs'] = outputs_fm

            # Background generator
            with tf.variable_scope('fc_b') as vscope:
                fc_dim = reduce(mul, self.configs.deconv_b_info.l1, 1)
                outputs_b = linear(inputs, fc_dim)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_b)
                assert(outputs_b.get_shape().as_list() == [self.batch_size, fc_dim])
                # outputs_b = tf.layers.dropout(outputs_b, rate=self.configs.dropout, training=self.is_train, name='outputs_b')
                outputs_b = tf.reshape(outputs_b, [self.batch_size] + self.configs.deconv_b_info.l1, name='reshape')
            with tf.variable_scope('deconv2_b') as vscope:
                outputs_b = deconv2d(outputs_b, [self.batch_size] +
                                     self.configs.deconv_b_info.l2, is_train=self.is_train)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_b)
                # outputs_b = tf.layers.dropout(outputs_b, rate=self.configs.dropout, training=self.is_train, name='outputs_b')
                assert(outputs_b.get_shape().as_list() == [self.batch_size] + self.configs.deconv_b_info.l2)
                self.net['b_deconv2_outputs'] = outputs_b
            with tf.variable_scope('deconv3_b') as vscope:
                outputs_b = deconv2d(outputs_b, [self.batch_size] +
                                     self.configs.deconv_b_info.l3, is_train=self.is_train)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_b)
                # outputs_b = tf.layers.dropout(outputs_b, rate=self.configs.dropout, training=self.is_train, name='outputs_b')
                assert(outputs_b.get_shape().as_list() == [self.batch_size] + self.configs.deconv_b_info.l3)
                self.net['b_deconv3_outputs'] = outputs_b
            with tf.variable_scope('deconv4_b') as vscope:
                outputs_b = deconv2d(outputs_b, [self.batch_size] +
                                     self.configs.deconv_b_info.l4, is_train=self.is_train)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs_b)
                assert(outputs_b.get_shape().as_list() == [self.batch_size] + self.configs.deconv_b_info.l4)
                outputs_b = tf.nn.tanh(outputs_b)
                self.net['b_deconv4_outputs'] = outputs_b

            # Construct output video from forground, background, mask
            outputs_b = tf.reshape(outputs_b, [self.batch_size, 1] + self.configs.deconv_b_info.l4)
            outputs_b_vol = tf.tile(outputs_b, [1, self.configs.num_frames, 1, 1, 1])
            outputs = outputs_fm * outputs_fi + (1 - outputs_fm) * outputs_b_vol
            tf.Assert(tf.less_equal(tf.reduce_max(outputs), 1.), [outputs], summarize=0, name='generator_max_assert')
            tf.Assert(tf.greater_equal(tf.reduce_max(outputs), -1.), [outputs], summarize=0, name='generator_min_assert')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return outputs


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
        self.reuse = False
        self.net = {}

    def __call__(self, inputs, is_train=True, is_debug=False):
        self.is_train = is_train
        self.is_debug = is_debug

        outputs = tf.convert_to_tensor(inputs)   # Check if necessary
        # assert input shape
        with tf.variable_scope(self.name, reuse=self.reuse) as scope:
            print_message(scope.name)
            with tf.variable_scope('conv1') as vscope:
                outputs = conv3d(outputs, [self.batch_size] + self.configs.conv_info.l1)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                self.net['conv1_outputs'] = outputs
            with tf.variable_scope('conv2') as vscope:
                outputs = conv3d(outputs, [self.batch_size] + self.configs.conv_info.l2)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                self.net['conv2_outputs'] = outputs
            with tf.variable_scope('conv3') as vscope:
                outputs = conv3d(outputs, [self.batch_size] + self.configs.conv_info.l3)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs)
                # outputs = tf.layers.dropout(outputs, rate=self.configs.dropout, training=self.is_train, name='outputs')
                self.net['conv3_outputs'] = outputs
            with tf.variable_scope('fc') as vscope:
                fc_dim = reduce(mul, self.configs.conv_info.l3, 1)
                outputs = tf.reshape(outputs, [self.batch_size] + [fc_dim], name='reshape')
                outputs = linear(outputs, 1)
                if is_debug and not self.reuse:
                    print(vscope.name, outputs)
                self.net['fc_outputs'] = outputs

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return tf.nn.sigmoid(outputs), outputs


class Model:
    ''' Overall model '''

    def __init__(self, configs, is_train=True, is_debug=False):
        self.configs = configs
        self.is_debug = is_debug
        self.reuse = False

        # Model info
        self.configs_encoder = configs.configs_encoder
        self.configs_generator = configs.configs_generator
        # self.configs_discriminator = configs.configs_discriminator

        self.configs_encoder.batch_size = configs.batch_size
        self.configs_encoder.num_frames = configs.data_info.num_frames
        self.configs_encoder.latent_dimension = configs.latent_dimension
        self.configs_generator.batch_size = configs.batch_size
        self.configs_generator.num_frames = configs.data_info.num_frames
        self.configs_generator.latent_dimension = configs.latent_dimension
        # self.configs_discriminator.batch_size = configs.batch_size

        # Data info
        self.num_frames = configs.data_info.num_frames
        self.image_height = configs.data_info.image_height
        self.image_width = configs.data_info.image_width
        self.num_channels = configs.data_info.num_channels
        # self.num_classes = configs.data_info.num_classes

        self.latent_dimension = configs.latent_dimension
        self.lr_ae = configs.learner_hyperparameters.lr_ae
        # self.lr_d = configs.learner_hyperparameters.lr_d
        self.beta1 = configs.learner_hyperparameters.beta1

        self.batch_size = configs.batch_size
        self.dataset_name = configs.dataset
        self.ckpt_dir = configs.ckpt_dir
        self.log_dir = configs.log_dir

        # Build model and loss
        self.build_model(is_train)
        self.build_loss()
        self.build_summary()

    def get_feed_dict(self, batch_chunk, step=None, is_train=True):
        ''' Organize data into a feed dictionary '''
        fd = {
            self.current_frames: batch_chunk['current_frames'],
            self.future_frames: batch_chunk['future_frames'],
            # self.label: batch_chunk['label'],
        }

        # TODO: add weight annealing

        # if is_train is not None:
        #     fd[self.is_train] = is_train

        return fd

    def build_model(self, is_train=True):
        ''' Build model '''

        # Placeholders for data
        self.current_frames = tf.placeholder(
            name='current_frames', dtype=tf.float32,
            shape=[self.batch_size, self.num_frames, self.image_height, self.image_width, self.num_channels]
        )
        self.future_frames = tf.placeholder(
            name='future_frames', dtype=tf.float32,
            shape=[self.batch_size, self.num_frames, self.image_height, self.image_width, self.num_channels]
        )
        # self.label = tf.placeholder(
        #     name='label', dtype=tf.float32, shape=[self.batch_size, self.num_classes]
        # )

        self.is_train = tf.placeholder_with_default(bool(is_train), [], name='is_train')

        # Encoder
        self.E = Encoder('Encoder', self.configs_encoder)
        self.z = self.E(self.current_frames, is_debug=self.is_debug)

        # Generators
        self.Gr = Generator('Generator_R', self.configs_generator)
        self.Gf = Generator('Generator_F', self.configs_generator)

        self.generated_current_frames = self.Gr(self.z, is_debug=self.is_debug)
        self.generated_future_frames = self.Gf(self.z, is_debug=self.is_debug)

        # # Discriminators
        # self.D = Discriminator('Discriminator', self.configs_discriminator)
        #
        # self.D_real_current, self.D_real_current_logits = self.D(self.current_frames, is_debug=self.is_debug)
        # self.D_fake_current, self.D_fake_current_logits = self.D(self.generated_current_frames, is_debug=self.is_debug)
        # self.D_real_future, self.D_real_future_logits = self.D(self.future_frames, is_debug=self.is_debug)
        # self.D_fake_future, self.D_fake_future_logits = self.D(self.generated_future_frames, is_debug=self.is_debug)

        print_message('Successfully loaded the model')

    def build_loss(self):
        ''' Build model loss and accuracy '''
        self.loss = {}

        # Reconstruction loss

        # L2 loss
        self.loss['input_reconstruction_loss_mse'] = tf.reduce_mean(
            tf.nn.l2_loss(self.generated_current_frames - self.current_frames))
        self.loss['future_reconstruction_loss_mse'] = tf.reduce_mean(
            tf.nn.l2_loss(self.generated_future_frames - self.future_frames))

        self.loss['input_reconstruction_loss'] = self.loss['input_reconstruction_loss_mse']
        self.loss['future_reconstruction_loss'] = self.loss['future_reconstruction_loss_mse']

        # # Adversarial loss
        # label_real_current = tf.zeros([self.batch_size, 1])
        # label_real_future = tf.zeros([self.batch_size, 1])
        # label_fake_current = tf.ones([self.batch_size, 1])
        # label_fake_future = tf.ones([self.batch_size, 1])

        # Generator
        # self.loss['generator_current'] = tf.reduce_mean(tf.log(self.D_fake_current))
        # self.loss['generator_future'] = tf.reduce_mean(tf.log(self.D_fake_future))
        self.loss['autoencoder'] = self.loss['input_reconstruction_loss'] + self.loss['future_reconstruction_loss']
        #  + self.loss['generator_current'] + self.loss['generator_future']

        # # Discriminator adversarial loss
        # self.loss['discriminator_real_current'] = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_current, label_real_current))
        # self.loss['discriminator_fake_current'] = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_current, label_fake_current))
        # self.loss['discriminator_real_future'] = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_future, label_real_future))
        # self.loss['discriminator_fake_future'] = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_future, label_fake_future))
        # self.loss['discriminator'] = self.loss['discriminator_real_current'] + self.loss['discriminator_fake_current'] + \
        #     self.loss['discriminator_fake_current'] + self.loss['discriminator_fake_future']

        # # Classification accuracy - for supervised
        # self.accuracy

    def build_summary(self):
        ''' Build summary for model '''
        # Latent variable distribution
        tf.summary.histogram('latent', self.z)

        # Loss summary
        tf.summary.scalar('loss/input_reconstruction_loss', self.loss['input_reconstruction_loss'])
        tf.summary.scalar('loss/future_reconstruction_loss', self.loss['future_reconstruction_loss'])
        tf.summary.scalar('loss/autoencoder', self.loss['autoencoder'])

        # Generated data summary
        generated_current_summary = tf.reshape(self.generated_current_frames,
                                               (-1, self.num_frames * self.image_height, self.image_width, self.num_channels))
        tf.summary.image('generated/current', generated_current_summary)
        generated_future_summary = tf.reshape(self.generated_future_frames,
                                              (-1, self.num_frames * self.image_height, self.image_width, self.num_channels))
        tf.summary.image('generated/future', generated_future_summary)
