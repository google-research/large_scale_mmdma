# coding=utf-8
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Main module that generates the data and trains the model.

Instructions:

python3 -m trainer.main -- --output_dir dir1/dir2 \
--data branch --n 300 --p 400 \
--k 4 --ns 100 \
--e 10001 --d 5 --nr 1000 --ne 1000 --keops True --dual False --pca 1000 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.

"""
import os

from absl import app
from absl import flags
from absl import logging
from mmdma import train
from mmdma.data import checkpointer
from mmdma.data import data_pipeline
from mmdma.metrics import SupervisedEvaluation
import numpy as np
import torch
import tensorflow as tf
from tensorflow.io import gfile

# Flags for input and output.
flags.DEFINE_string('output_dir',
                    'gs://xcloud_shared/lpapaxanthos/mmdma',
                    'Output directory.')
flags.DEFINE_string('input_dir',
                    'gs://xcloud-shared/lpapaxanthos/data/mmd-ma',
                    'Output directory.')
flags.DEFINE_string(
    'input_fv', '', 'Input first view, can be point cloud or kernel.')
flags.DEFINE_string(
    'input_sv', '', 'Input second view, can be point cloud or kernel.')
flags.DEFINE_string('rd_vec', '', 'Permutation of first view.')
flags.DEFINE_bool(
    'kernel', False, 'Whether the input is a point cloud or kernel.')
flags.DEFINE_enum('data', 'branch', ['branch', 'triangle', ''],
                  'Chooses simulation or user input.')
flags.DEFINE_integer('n', 3000, 'Sample size of generated data.')
flags.DEFINE_integer('p', 1000, 'Number of features in the generated data.')

# Random seeds.
flags.DEFINE_integer('k', 4, 'Seed.')
flags.DEFINE_integer('ns', 100, 'Number of seeds with which to run the model.')

# Flags for training and evaluation.
flags.DEFINE_integer('e', 10001, 'Number of epochs.')
flags.DEFINE_integer('ne', 1000, 'When to evaluate.')
flags.DEFINE_integer('nr', 1000, 'When to record the loss.')
flags.DEFINE_integer('pca', 1000, 'Applies PCA to embeddings every X epochs.')

# Flags to define the algorithm.
flags.DEFINE_boolean('keops', True, 'Uses keops or not.')
flags.DEFINE_enum('m', 'primal', ['dual', 'primal'], 'Dual or primal mode.')

# Flags for model hyperparameters.
flags.DEFINE_integer('d', 5, 'Dimension of output space.')
flags.DEFINE_string('init', 'uniform,0.,1.', 'Initialiser.')
flags.DEFINE_float('l1', 1e-4, 'Hyperparameter for penalty terms.')
flags.DEFINE_float('l2', 1e-4, 'Hyperparameter for distortion terms.')
flags.DEFINE_float('lr', 1e-5, 'Learning rate.')
flags.DEFINE_float('s', 1., 'Scale parameter.')


FLAGS = flags.FLAGS


def main(_):
  tf.config.experimental.set_visible_devices([], 'GPU')
  logging.info('cuda is available: %s', torch.cuda.is_available())
  logging.info('device name is: %s', torch.cuda.get_device_name(0))
  logging.info('size of dataset: %s x %s', FLAGS.n, FLAGS.p)

  # Gets input data.
  if FLAGS.data:
    first_view, second_view, rd_vec = data_pipeline.generate_data(
        FLAGS.n, FLAGS.p, simulation=FLAGS.data, implementation='pytorch')
  else:
    first_view = data_pipeline.load(FLAGS.input_dir, FLAGS.input_fv)
    second_view = data_pipeline.load(FLAGS.input_dir, FLAGS.input_sv)
    if FLAGS.rd_vec:
      rd_vec = data_pipeline.load(FLAGS.input_dir, FLAGS.rd_vec)
    else:
      rd_vec = np.arange(first_view.shape[0])

  # Creates output directory and filename.
  gfile.makedirs(FLAGS.output_dir)
  logging.info('output directory: %s', FLAGS.output_dir)
  if FLAGS.input_fv:
    filename = ':'.join(['m' + str(FLAGS.m),
                         'keops' + str(FLAGS.keops),
                         'ni' + str(FLAGS.e),
                         'k' + str(FLAGS.k),
                         'ns' + str(FLAGS.ns),
                         'lr' + str(FLAGS.lr),
                         's' + str(FLAGS.s),
                         'l1' + str(FLAGS.l1),
                         'l2' + str(FLAGS.l2),
                         'i' + str(FLAGS.init),
                         FLAGS.input_fv.split('.')[0]])
  else:
    filename = ':'.join(['m' + str(FLAGS.m),
                         'keops' + str(FLAGS.keops),
                         'n' + str(FLAGS.n),
                         'ni' + str(FLAGS.e),
                         'k' + str(FLAGS.k),
                         'ns' + str(FLAGS.ns),
                         'p' + str(FLAGS.p),
                         'lr' + str(FLAGS.lr),
                         's' + str(FLAGS.s),
                         'l1' + str(FLAGS.l1),
                         'l2' + str(FLAGS.l2),
                         'i' + str(FLAGS.init),
                         str(FLAGS.data)])

  # Prepares metrics.
  eval_fn = SupervisedEvaluation(ground_truth_alignment=rd_vec)

  # Sets the hyperparameters of the model.
  cfg_model = train.ModelGetterConfig(
      key=FLAGS.k,
      n_seed=FLAGS.ns,
      low_dim=FLAGS.d,
      n_iter=FLAGS.e,
      keops=FLAGS.keops,
      mode=FLAGS.m,
      n_eval=FLAGS.ne,
      n_record=FLAGS.nr,
      learning_rate=FLAGS.lr,
      sigmas=FLAGS.s,
      lambda1=FLAGS.l1,
      lambda2=FLAGS.l2,
      pca=FLAGS.pca,
      init=FLAGS.init
      )

  # Moves input to device.
  device = torch.device('cuda')
  first_view = torch.FloatTensor(first_view).to(device)
  second_view = torch.FloatTensor(second_view).to(device)
  if cfg_model.mode == 'dual' and not FLAGS.kernel:
    first_view = train.get_kernel(first_view)
    second_view = train.get_kernel(second_view)

  # Runs model.
  out = train.train_and_evaluate(
      cfg_model, first_view, second_view, eval_fn, workdir=FLAGS.output_dir,
      device=device)
  optim, model, eval_loss, eval_matching, pca_results, key = out

  # Saves results to files.
  loss = eval_loss['loss'][-1]
  mmd = eval_loss['mmd'][-1]
  foscttm = eval_matching['foscttm'][-1]
  top1 = eval_matching['top1'][-1]
  top5 = eval_matching['top5'][-1]
  results = [foscttm, top1, top5]
  logging.info('Save metrics in %s.', FLAGS.output_dir)
  with gfile.GFile(
      os.path.join(FLAGS.output_dir, filename + '.tsv'), 'w') as my_file:
    my_file.write('model\tkey\tn_sample\tn_feat\tlow_dim\tn_iter\tkeops\t '
                  'loss\tmmd\tfoscttm\ttop1\ttop5\n')
    checkpointer.save_data_eval(
        my_file, FLAGS, key, loss, mmd, results, cfg_model)
  logging.info('Save tracking in %s.', FLAGS.output_dir)
  checkpointer.save_tracking(FLAGS.output_dir,
                             filename,
                             eval_loss,
                             eval_matching,
                             key,
                             FLAGS.e,
                             cfg_model)
  logging.info('Save model in %s.', FLAGS.output_dir)
  checkpointer.save_model(
      FLAGS.output_dir, filename, optim, model, key, FLAGS.e, loss, rd_vec)
  if FLAGS.pca != 0:
    logging.info('Save PCA representation in %s', FLAGS.output_dir)
    checkpointer.save_pca(FLAGS.output_dir, filename, pca_results, rd_vec)
  logging.info('End.')

if __name__ == '__main__':
  app.run(main)
