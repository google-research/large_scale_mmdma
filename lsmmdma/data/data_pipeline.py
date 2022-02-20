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

"""Generates data with varying number of samples and features."""
import os
import random
from typing import Tuple, Union

from lsmmdma.data.checkpointer import save_generated_data
import numpy as np
import scanpy
import torch
from tensorflow.io import gfile


def generate_data(
    path: str,
    n_sample: int,
    p_feature: int,
    random_seed: int = 4,
    simulation: str = 'branch',
    ) -> Union[Tuple[np.ndarray], Tuple[torch.Tensor]]:
  """Generates two views following the `simulation` shape in their latent space.

  The simulation process follows the one given in the original MMD-MA paper (Liu
  et al. https://pubmed.ncbi.nlm.nih.gov/34632462/). A manifold (of shape
  'branch' or 'triangle') is generated in two dimensions. The resulting set of
  points is standardised. The points are then mapped to `p_feature` dimensions
  using a (2 x p_feature) mapping, sampled from a standard Gaussian distribution,
  resulting in a (n_sample * p_feature) matrix. Gaussian noise is then added to
  each element of the matrix.

  In these simulations, we assume that the numbers of samples and features are
  are the same for both domains.

  Arguments:
    path: path to output directory where to save the generated data.
    n_sample: number of samples.
    p_feature: number of features.
    random_seed: seed for random (random_seed) and numpy (random_seed + 1)
    simulation: str, latent structure for the input data. The latent structure
      can either be a 'branch' (two elongated Gaussian point clouds
      perpendicular to each other) or a 'triangle' (three elongated Gaussian
      point clouds forming a triangle).

  Returns:
    tuple, first view, second view and corresponding indices from the first
      view.
  """
  latent_dim = 2
  # TODO(lpapaxanthos): allow for different number of features.
  p_features1 = p_feature
  p_features2 = p_feature
  noise = 0.05

  if simulation == 'triangle':
    means = [(0, 1), (3.5, 4.33), (-3.5, 4.33)]
    covs = [[[8, 0], [0, 0.05]],
            [[1.7, -1.5], [-1.5, 1.7]],
            [[1.7, 1.5], [1.5, 1.7]]]
    n1 = int(n_sample / 4)
    shapes = [2 * n1, n1, n1 - 3 * n1]
    rot = None
  elif simulation == 'branch':
    means = [[0, 0], [0, 8]]
    covs = [[[20, 0], [0, 1]], [[1, 0], [0, 10]]]
    n1 = int(n_sample / 4 * 3)
    shapes = [n1, n_sample - n1]
    rot = np.array([[np.cos(np.pi/4.), -np.sin(np.pi/4.)],
                    [np.sin(np.pi/4.), np.cos(np.pi/4.)]])
  else:
    raise ValueError('Value for simulation is not supported')

  length = len(means)

  random.seed(random_seed)
  np.random.seed(random_seed + 1)
  axis = [0] * length
  for i in range(length):
    axis[i] = np.random.multivariate_normal(
        mean=np.array(means[i]), cov=np.array(covs[i]), size=shapes[i])

  latent_array = np.concatenate((list(axis[i] for i in range(length))))
  if rot is not None:
    latent_array = np.dot(latent_array, rot)
  latent_array = (latent_array - latent_array.mean(0)) / latent_array.std(0)

  noise_first_view = np.random.normal(
      size=(latent_dim, p_features1))
  noise_second_view = np.random.normal(
      size=(latent_dim, p_features2))
  first_view = (torch.FloatTensor(
      np.dot(latent_array, noise_first_view) + np.random.normal(
          size=(n_sample, p_features1)) * noise))
  second_view = (torch.FloatTensor(
      np.dot(latent_array, noise_second_view) + np.random.normal(
          size=(n_sample, p_features2)) * noise))

  rd_vec = np.random.choice(n_sample, n_sample, replace=False)
  first_view = first_view[rd_vec]

  if path:
    save_generated_data(path, first_view, second_view, rd_vec)
  return first_view, second_view, rd_vec


def load(input_dir: str, filename: str) -> np.ndarray:
  """Loads data."""
  path_file = os.path.join(input_dir, filename)
  file_ext = os.path.splitext(path_file)[-1]
  with gfile.GFile(path_file, 'rb') as my_file:
    if file_ext == '.h5ad':
      data = scanpy.read_h5ad(my_file)
      data = data.X.todense()
    elif file_ext == '.npy':
      data = np.load(my_file)
    elif file_ext == '.tsv':
      data = np.loadtxt(my_file, delimiter='\t')
    elif file_ext == '.csv':
      data = np.loadtxt(my_file, delimiter=',')
    else:
      try:
        data = np.loadtxt(my_file)
      except:
        raise ValueError('Input data is not in a supported format.')
  return data
