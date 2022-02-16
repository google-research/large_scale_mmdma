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

"""Saves results."""
import dataclasses
import json
import os
from typing import Tuple, DefaultDict, List

import absl
from lsmmdma.train import ModelGetterConfig
import numpy as np
import torch
from tensorflow.io import gfile


def save_generated_data(
    path: str,
    first_view: torch.FloatTensor,
    second_view: torch.FloatTensor,
    rd_vec: np.ndarray):
  """Saves generated data."""
  file_prefix = 'generated_data'
  with gfile.GFile(os.path.join(path, file_prefix + '_fv.pt'), 'wb') as my_file:
    torch.save(first_view, my_file)
  with gfile.GFile(os.path.join(path, file_prefix + '_sv.pt'), 'wb') as my_file:
    torch.save(second_view, my_file)
  with gfile.GFile(
      os.path.join(path, file_prefix + '_rd_vec.npy'), 'wb') as my_file:
    np.save(rd_vec, my_file)


def save_data_eval(
    my_file: gfile.GFile,
    flags: absl.flags,
    seed: int,
    loss: float,
    mmd: float,
    res: float,
    runtime: float,
    cfg_model: ModelGetterConfig,
    ):
  """Saves main results."""
  cfg_dict = dataclasses.asdict(cfg_model)
  val = cfg_dict.get('keops', None)
  args = [flags.m, seed, flags.n, flags.p, flags.d,
          flags.e, val, loss, mmd, res[0], res[1], res[2], runtime]
  my_file.write('\t'.join(map(str, args)) + '\n')


def save_model(
    path: str,
    filename: str,
    optimizer: torch.optim,
    model: torch.nn.Module,
    seed: int,
    epoch: int,
    loss: float
    ):
  """Saves model parameters."""
  filename = f'{filename}_model.json'
  with gfile.GFile(os.path.join(path, filename), 'w') as my_file:
    torch.save({
        'seed': seed,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, my_file)


def save_pca(
    path: str,
    filename: str,
    pca_results: Tuple[np.ndarray],
    ):
  """Saves 2D representation."""
  filename_pca = f'{filename}_pca.npy'
  with gfile.GFile(os.path.join(path, filename_pca), 'w') as my_file:
    np.save(my_file, np.array(pca_results))


def save_embeddings(
    path: str,
    filename: str,
    embeddings_results: Tuple[np.ndarray],
    ):
  """Saves low dimensional representation."""
  filename_emb = f'{filename}_embeddings.npy'
  with gfile.GFile(os.path.join(path, filename_emb), 'w') as my_file:
    np.save(my_file, np.array(embeddings_results))


def save_tracking(
    path: str,
    filename: str,
    evaluation_loss: DefaultDict[str, List[float]],
    evaluation_matching: DefaultDict[str, List[float]],
    seed: int,
    epoch: int,
    cfg_model: ModelGetterConfig
    ):
  """Saves evaluation measures."""
  filename = f'{filename}_tracking.json'
  with gfile.GFile(os.path.join(path, filename), 'w') as my_file:
    data = {
        'seed': int(seed),
        'mode': cfg_model.mode,
        'keops': cfg_model.keops,
        'epoch': epoch,
        'evaluation_loss': evaluation_loss,
        'evaluation_matching': evaluation_matching
    }
    json.dump(data, my_file)
