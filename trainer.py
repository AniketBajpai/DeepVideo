from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import better_exceptions
import os
from time import time, strftime
from six.moves import xrange
from tqdm import tqdm
import h5py
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import log, pp
from input_ops import create_input_ops
from model import Model


class Trainer(object):
    def __init__(self, configs, dataset_train, dataset_test):
        self.configs = configs
        hyperparameter_str = '{}_lr_{}_update_G{}_D{}'.format(
            configs.dataset,
            str(configs.learner_hyperparameters.lr_ae),
            str(configs.learner_hyperparameters.update_ratio),
            str(1)
        )
        self.train_dir = '{}/train_dir/{}-{}-{}'.format(
            configs.home_dir,
            configs.prefix,
            hyperparameter_str,
            strftime('%Y%m%d-%H%M%S')
        )

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        log.infov('Train Dir: %s', self.train_dir)

        # ------------------ input ops ------------------ #
        self.batch_size = configs.batch_size

        _, self.batch_train = create_input_ops(dataset_train, self.batch_size, is_train=True)
        _, self.batch_test = create_input_ops(dataset_test, self.batch_size, is_train=False)

        # ------------------ create model ------------------ #
        self.model = Model(configs, is_train=True, is_debug=False)

        # ------------------ optimizer ------------------ #
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.lr_ae = configs.learner_hyperparameters.lr_ae
        self.lr_d = configs.learner_hyperparameters.lr_d

        # weight decay
        # if configs.learner_hyperparameters.lr_weight_decay:
        #     self.lr_ae = tf.train.exponential_decay(
        #         self.lr_ae,
        #         global_step=self.global_step,
        #         decay_steps=10000,
        #         decay_rate=0.5,
        #         staircase=True,
        #         name='decaying_learning_rate'
        #     )

        self.check_op = tf.no_op()

        # ------------------ checkpoint and monitoring ------------------ #
        all_vars = tf.trainable_variables()

        log.warn('*************************** ')
        log.warn('********* ae_var ********** ')
        log.warn('*************************** ')
        e_var = [v for v in all_vars if v.name.startswith('Encoder')]
        log.warn('********* e_var ********** ')
        slim.model_analyzer.analyze_vars(e_var, print_info=True)

        g_r_var = [v for v in all_vars if v.name.startswith(('Generator_R'))]
        log.warn('********* g_r_var ********** ')
        slim.model_analyzer.analyze_vars(g_r_var, print_info=True)

        # g_f_var = [v for v in all_vars if v.name.startswith(('Generator_F'))]
        # log.warn('********* g_f_var ********** ')
        # slim.model_analyzer.analyze_vars(g_f_var, print_info=True)

        # ae_var = e_var + g_r_var + g_f_var
        ae_var = e_var + g_r_var

        d_var = [v for v in all_vars if v.name.startswith('Discriminator')]
        log.warn('*************************** ')
        log.warn('********* d_var ********** ')
        log.warn('*************************** ')
        slim.model_analyzer.analyze_vars(d_var, print_info=True)

        # rem_var = (set(all_vars) - set(e_var) - set(g_r_var) - set(g_f_var) - set(d_var))
        rem_var = (set(all_vars) - set(e_var) - set(g_r_var) - set(d_var))
        log.warn('********* rem ********** ')
        print([v.name for v in rem_var])
        assert not rem_var

        self.ae_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss['autoencoder'],
            global_step=self.global_step,
            learning_rate=configs.learner_hyperparameters.lr_ae,
            optimizer=tf.train.AdamOptimizer(learning_rate=configs.learner_hyperparameters.lr_ae,
                                             beta1=configs.learner_hyperparameters.beta1),
            clip_gradients=configs.learner_hyperparameters.clip_gradients,
            name='ae_optimize_loss',
            variables=ae_var
        )

        self.d_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss['discriminator'],
            global_step=self.global_step,
            learning_rate=configs.learner_hyperparameters.lr_d,
            optimizer=tf.train.AdamOptimizer(learning_rate=configs.learner_hyperparameters.lr_d,
                                             beta1=configs.learner_hyperparameters.beta1),
            clip_gradients=configs.learner_hyperparameters.clip_gradients,
            name='d_optimize_loss',
            variables=d_var
        )

        self.summary_op = tf.summary.merge_all()

        assert_ops = tf.get_collection('Assert', scope='assert')
        print ('Total asserts:', len(assert_ops))
        self.assert_op = tf.group(*assert_ops)

        self.saver = tf.train.Saver(max_to_keep=100)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.summaries_secs = configs.summaries_secs
        self.checkpoint_secs = configs.checkpoint_secs

        # Automatic saving of models and summaries disabled
        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=self.summaries_secs,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.log_dir = configs.log_dir
        self.ckpt_dir = configs.ckpt_dir
        # # Restore saved weights
        # if self.ckpt_dir is not None:
        #     log.info('Checkpoint path: %s', self.ckpt_dir)
        #     self.saver.restore(self.session, self.ckpt_dir)
        #     log.info('Loaded the pretrain parameters from the provided checkpoint path')

    def train(self):
        log.infov('Training started')
        pp.pprint(self.batch_train)

        max_steps = self.configs.max_steps

        output_save_step = self.configs.output_save_step
        test_step = self.configs.test_step
        log_step = self.configs.log_step

        for s in tqdm(xrange(max_steps)):
            step, ae_loss, g_loss, d_loss, summary, step_time, generated_current_frames, all_predicitons = self.run_single_step(
                self.batch_train, step=s, is_train=True)

            # periodic inference
            if s % test_step == 0:
                ae_loss_test, g_loss_test, d_loss_test, generated_current_frames_test, all_predicitons = self.run_test(
                    self.batch_test, is_train=False)
                log.infov('Test')
                self.log_step_message(step, ae_loss_test, g_loss_test, d_loss_test, step_time, is_train=False)

            if s % log_step == 0:
                self.log_step_message(step, ae_loss, g_loss_test, d_loss_test, step_time)
                self.summary_writer.add_summary(summary, global_step=step)

            if s % output_save_step == 0:
                log.infov('Saved checkpoint at %d', s)
                save_path = self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step)
                if self.configs.dump_result:
                    f = h5py.File(os.path.join(self.train_dir, 'generated_current_' + str(s) + '.hy'), 'w')
                    f['generated_current_frames'] = generated_current_frames
                    f.close()
                    # f = h5py.File(os.path.join(self.train_dir, 'generated_future_' + str(s) + '.hy'), 'w')
                    # f['generated_future_frames'] = generated_future_frames
                    # f.close()

    def run_single_step(self, batch, step=None, is_train=True):
        ''' Run a single step of training iteration on an batch '''
        _start_time = time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.model.loss['autoencoder'], self.model.loss['generator_current'], self.model.loss['discriminator'],
                 self.summary_op, self.assert_op, self.model.generated_current_frames, self.model.all_predictions, self.check_op]

        if step % (self.configs.learner_hyperparameters.update_ratio + 1) > 0:
            # Train the generator
            fetch.append(self.ae_optimizer)
        else:
            # Train the discriminator
            fetch.append(self.d_optimizer)

        step, ae_loss, g_loss, d_loss, summary, _, generated_current_frames, all_predictions, _, _ = self.session.run(
            fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step, is_train=True)
        )

        _end_time = time()

        return step, ae_loss, g_loss, d_loss, summary, (_end_time - _start_time), generated_current_frames, all_predictions

    def run_test(self, batch, is_train=False, repeat_times=8):
        ''' Run test iteration on batch '''

        batch_chunk = self.session.run(batch)
        fetch = [self.global_step, self.model.loss['autoencoder'], self.model.loss['generator_current'], self.model.loss['discriminator'],
                 self.model.generated_current_frames, self.model.all_predictions]

        [step, ae_loss, g_loss, d_loss, generated_current_frames, all_predictions] = self.session.run(
            fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk, is_train=False))

        return ae_loss, g_loss, d_loss, generated_current_frames, all_predictions

    def log_step_message(self, step, ae_loss, g_loss, d_loss, step_time, is_train=True):
        ''' Periodic log message '''
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((' [{split_mode:5s} step {step:4d}] ' +
                'AE loss: {ae_loss:.5f} ' +
                # 'Supervised loss: {s_loss:.5f} ' +
                'G loss: {g_loss:.5f} ' +
                'D loss: {d_loss:.5f} ' +
                # 'Accuracy: {accuracy:.5f} '
                # 'test loss: {test_loss:.5f} ' +
                '({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) '
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         ae_loss=ae_loss,
                         #  s_loss=s_loss,
                         g_loss=g_loss,
                         d_loss=d_loss,
                         #  accuracy=accuracy,
                         #  test_loss=accuracy_test,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='moving_mnist',
                        choices=['moving_mnist', 'yfcc100m/golf', 'yfcc100m/beach'])
    args = parser.parse_args()

    if args.dataset == 'moving_mnist':
        import datasets.moving_mnist as dataset
    elif args.dataset == 'yfcc100m/golf':
        pass
    elif args.dataset == 'yfcc100m/beach':
        pass
    else:
        raise ValueError(args.dataset)

    configs = dataset.load_configs()
    dataset_train, dataset_test = dataset.create_default_splits(configs)
    trainer = Trainer(configs, dataset_train, dataset_test)

    log.warning('dataset: %s, ae learning_rate: %f', configs.dataset, configs.learner_hyperparameters.lr_ae)
    trainer.train()


if __name__ == '__main__':
    main()
