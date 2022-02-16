# coding=utf-8
# Copyright 2022 Google LLC.
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

1. To run the algorithm on simulated data from data_pipeline.py:

python3 -m lsmmdma.main --output_dir outdir \
--data branch --n 300 --p 400 \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops True --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'

2. To run the algorithm on user input data, in the form n_sample x p_feature.
--data should be '' (default value) and --kernel should be False. The
argument --keops can be True or False, --mode can be 'dual' or 'primal'.

python3 -m lsmmdma.main --input_dir datadir --output_dir outdir \
--input_fv my_data_1 --input_sv my_data_2 --kernel False \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops True --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'

3. To run the algorithm on user kernel data, in the form n_sample x n_sample.
--data should be '' (default value) and --kernel should be True. The
argument --keops can be True or False, --mode can only be `dual`.

python3 -m lsmmdma.main --input_dir datadir --output_dir outdir \
--input_fv my_data_1 --input_sv my_data_2 --kernel True \
--k 4 --ns 5 \
--e 1001 --d 5 --nr 100 --ne 100 --keops True --m dual --pca 100 \
--lr 1e-5 --l1 1e-4 --l2 1e-4 --s 1.0 --init 'uniform,0,0.1'
"""
import os

from absl import app
from absl import flags
from absl import logging
import dataclasses
from lsmmdma import train
from lsmmdma.data import checkpointer
from lsmmdma.data import data_pipeline
from lsmmdma.metrics import SupervisedEvaluation
import numpy as np
import torch
import tensorflow as tf
from tensorflow.io import gfile

# Flags for input and output.
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('input_dir', None, 'Input directory.')
flags.DEFINE_string(
    'input_fv', None, 'Input first view, can be point cloud or kernel.')
flags.DEFINE_string(
    'input_sv', None, 'Input second view, can be point cloud or kernel.')
flags.DEFINE_string('rd_vec', None, 'Permutation of first view.')
flags.DEFINE_bool(
    'kernel', False, 'Whether the input is a point cloud or kernel.')
flags.DEFINE_enum('data', None, ['branch', 'triangle', None],
                  'Chooses simulation or user input.')
flags.DEFINE_integer('n', None, 'Sample size of generated data.')
flags.DEFINE_integer('p', None, 'Number of features in the generated data.')

# Random seeds.
flags.DEFINE_integer('seed', 0, 'Seed.')
flags.DEFINE_integer('ns', 1, 'Number of seeds with which to run the model.')

# Flags for training and evaluation.
flags.DEFINE_integer('e', 5001, 'Number of epochs.')
flags.DEFINE_integer('ne', 100, 'When to evaluate.')
flags.DEFINE_integer('nr', 100, 'When to record the loss.')
flags.DEFINE_integer('pca', 100, 'Applies PCA to embeddings every X epochs.')

# Flags to define the algorithm.
flags.DEFINE_boolean('keops', True, 'Uses keops or not.')
flags.DEFINE_enum('m', 'dual', ['dual', 'primal'], 'Dual or primal mode.')

# Flags for model hyperparameters.
flags.DEFINE_integer('d', 5, 'Dimension of output space.')
flags.DEFINE_string('init', 'uniform,0.,0.1', 'Initialiser.')
flags.DEFINE_float('l1', 1e-4, 'Hyperparameter for penalty terms.')
flags.DEFINE_float('l2', 1e-4, 'Hyperparameter for distortion terms.')
flags.DEFINE_float('lr', 1e-5, 'Learning rate.')
flags.DEFINE_float('s', 1.0, 'Scale parameter.')

# Flags to assess runtime
flags.DEFINE_bool('time', False, 'Whether or not to measure runtime.')


FLAGS = flags.FLAGS


def create_dummy_config(cfg: train.ModelGetterConfig):
  """Creates dummy config, needed when timing the training loop."""
  cfg = dataclasses.replace(cfg)
  cfg.n_iter = 1
  cfg.pca = 0
  cfg.n_record = 0
  cfg.n_eval = 0
  return cfg


def time_training_loop(func):
  def inner_fn(cfg_model: train.ModelGetterConfig, *args):
    """Enables to time the training loop."""
    cfg_model_time = create_dummy_config(cfg_model)
    _ = func(cfg_model_time, *args)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = func(cfg_model, *args)
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    runtime = start.elapsed_time(end)
    return runtime, out
  return inner_fn


def main(_):
  if FLAGS.kernel and FLAGS.mode == 'primal':
    raise ValueError(f"""The flag kernel set to {FLAGS.kernel} is not compatible
                     with the flag mode set to {FLAGS.mode}""")

  tf.config.experimental.set_visible_devices([], 'GPU')
  logging.info('cuda is available: %s', torch.cuda.is_available())

  # Gets input data.
  if FLAGS.data:
    first_view, second_view, rd_vec = data_pipeline.generate_data(
        path=FLAGS.output_dir,
        random_seed=FLAGS.seed,
        n_sample=FLAGS.n,
        p_feature=FLAGS.p,
        simulation=FLAGS.data)
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
  filename = ':'.join(['m' + str(FLAGS.m),
                       'keops' + str(FLAGS.keops),
                       'ni' + str(FLAGS.e),
                       'seed' + str(FLAGS.seed),
                       'ns' + str(FLAGS.ns),
                       'lr' + str(FLAGS.lr),
                       's' + str(FLAGS.s),
                       'l1' + str(FLAGS.l1),
                       'l2' + str(FLAGS.l2),
                       'i' + str(FLAGS.init)])
  if FLAGS.data:
    filename = ':'.join([filename,
                         'n' + str(FLAGS.n),
                         'p' + str(FLAGS.p),
                         str(FLAGS.data)])
  else:
    filename = ':'.join([filename,
                         FLAGS.input_fv.split('.')[0]])

  # Chooses device.
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = 'cpu'

  # Prepares metrics.
  eval_fn = SupervisedEvaluation(ground_truth_alignment=rd_vec,
                                 device=device)

  # Sets the hyperparameters of the model.
  cfg_model = train.ModelGetterConfig(
      seed=FLAGS.seed,
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
      init=FLAGS.init,
      )

  # Moves input to device.
  first_view = torch.FloatTensor(first_view).to(device)
  second_view = torch.FloatTensor(second_view).to(device)
  if cfg_model.mode == 'dual' and not FLAGS.kernel:
    first_view = train.get_kernel(first_view)
    second_view = train.get_kernel(second_view)

  # Runs model.
  train_fn = (time_training_loop(train.train_and_evaluate)
              if FLAGS.time else train.train_and_evaluate)
  if FLAGS.time:
    runtime, out = train_fn(
        cfg_model, first_view, second_view, eval_fn, '', device)
  else:
    out = train_fn(
        cfg_model, first_view, second_view, eval_fn, FLAGS.output_dir, device)
    runtime = '-'
  optim, model, eval_loss, eval_matching, emb_results, pca_results, seed = out

  # Saves results to files.
  loss = eval_loss['loss'][-1] if FLAGS.nr != 0 else -1
  mmd = eval_loss['mmd'][-1] if FLAGS.nr != 0 else -1
  foscttm = eval_matching['foscttm'][-1] if FLAGS.ne != 0 else -1
  top1 = eval_matching['top1'][-1] if FLAGS.ne != 0 else -1
  top5 = eval_matching['top5'][-1] if FLAGS.ne != 0 else -1
  results = [foscttm, top1, top5]

  logging.info('Save results in %s.', FLAGS.output_dir)
  with gfile.GFile(
      os.path.join(FLAGS.output_dir, filename + '.tsv'), 'w') as my_file:
    colnames = ['model', 'seed', 'n_sample', 'n_feat', 'low_dim', 'n_iter',
                'keops', 'loss', 'mmd', 'foscttm', 'top1', 'top5', 'time']
    my_file.write('\t'.join(colnames) + '\n')
    checkpointer.save_data_eval(
        my_file, FLAGS, seed, loss, mmd, results, runtime, cfg_model)
  if FLAGS.ne != 0:
    logging.info('Save tracking in %s.', FLAGS.output_dir)
    checkpointer.save_tracking(FLAGS.output_dir,
                               filename,
                               eval_loss,
                               eval_matching,
                               seed,
                               FLAGS.e,
                               cfg_model)
  logging.info('Save model in %s.', FLAGS.output_dir)
  checkpointer.save_model(
      FLAGS.output_dir, filename, optim, model, seed, FLAGS.e, loss )
  logging.info('Save embeddings in %s', FLAGS.output_dir)
  checkpointer.save_embeddings(
      FLAGS.output_dir, filename, emb_results)
  if FLAGS.pca != 0:
    logging.info('Save PCA representation in %s', FLAGS.output_dir)
    checkpointer.save_pca(FLAGS.output_dir, filename, pca_results)
  logging.info('End.')

if __name__ == '__main__':
  flags.mark_flags_as_mutual_exclusive(['data', 'input_dir'], required=True)
  flags.mark_flags_as_mutual_exclusive(['data', 'input_fv'], required=True)
  flags.mark_flags_as_mutual_exclusive(['data', 'input_sv'], required=True)
  flags.mark_flags_as_mutual_exclusive(['data', 'rd_vec'], required=True)
  flags.mark_flags_as_mutual_exclusive(['n', 'input_dir'], required=True)
  flags.mark_flags_as_mutual_exclusive(['p', 'input_dir'], required=True)

  app.run(main)
