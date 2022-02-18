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

"""Functions used in MMDMA, PyTorch implementation."""
import numpy as np
import pykeops
pykeops.clean_pykeops()  # just in case old build files are still present
from pykeops.torch import LazyTensor
from sklearn import decomposition
import torch
from typing import Optional


# Functions in dual.
def pen_dual(
    embedding: torch.Tensor,
    param: torch.Tensor,
    device: torch.device
    ) -> torch.Tensor:
  """Computes the non-collapsing penalty for the dual formulation.

  In the dual formulation, when we project n points to low_dim dimensions,
  the model parameter 'param' is an n*low_dim matrix. The 'embedding'
  matrix is also an n*low_dim matrix equal to embedding=K*param,
  where K is the n*n kernel matrix of the n points in the input space.
  The 'embedding' matrix corresponds to the mapped set of points.
  The non-collapsing penalty is then equal to
  ||param.T * K * param - I ||, where I is the low_dim*low_dim
  identity matrix, which is also equal to ||param.T * embedding - I ||.

  Arguments:
    embedding: torch.Tensor, low dimensional representation of a view.
    param: torch.Tensor, model parameters.
    device: torch.device, whether cuda or cpu.

  Returns:
    penalty_value: torch.Tensor, scalar value.
  """
  low_dim = embedding.shape[1]
  identity_matrix_embedding_space = torch.eye(low_dim).to(device)
  penalty_value = (torch.matmul(param.t(), embedding)
                   - identity_matrix_embedding_space).norm(2)
  return penalty_value


def dis_dual(
    embedding: torch.Tensor,
    kernel: torch.Tensor
    ) -> torch.Tensor:
  """Computes the distortion penalty for the dual formulation.

  In the dual formulation, when we project n points to a shared space of
  low_dim dimensions, the 'embedding' matrix is an n*low_dim matrix equal to
  embedding=K*param, where K is the n*n kernel matrix of the n points in the
  input space and the n*low_dim 'param' matrix is the model parameter.  The
  'embedding' matrix corresponds to the mapped set of points.
  The distortion penalty is then equal to ||K * param * param.T * K - K ||,
  which is also equal to ||embedding * embedding.T - K||.

  Arguments:
    embedding: torch.Tensor, low dimensional representation of a view.
    kernel: torch.Tensor, view.

  Returns:
    penalty_value: torch.Tensor, scalar value.
  """
  distortion_value = (
      torch.matmul(embedding, embedding.t()) - kernel).norm(2)
  return distortion_value


# Functions for MMD.
def squared_mmd(
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    sigmas: torch.Tensor,
    keops: bool,
    device: torch.device
    ) -> float:
  """Computes squared MMD with Gaussian kernel.

  Arguments:
    first_view: torch.Tensor, embedding for first view (sample x low_dim).
    second_view: torch.Tensor, embedding for second view (sample x low_dim).
    sigmas: torch.Tensor, scale parameter for Gaussian kernel.
    keops: bool, whether or not to use keops.
    device: torch.device, device can be cuda or cpu.
  Returns:
    float, MMD value.
  """
  if keops:
    cost = squared_mmd_keops(first_view, second_view, sigmas, device)
  else:
    n_sample1 = first_view.shape[0]
    n_sample2 = second_view.shape[0]
    cost = gaussian_kernel(
        first_view, None, sigmas, keops).sum() / n_sample1**2
    cost += gaussian_kernel(
        second_view, None, sigmas, keops).sum() / n_sample2**2
    cost -= 2 * gaussian_kernel(
        first_view, second_view, sigmas, keops).sum() / (
            n_sample1 * n_sample2)
  return torch.clip(cost, min=0)


def gaussian_kernel(
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    sigmas: torch.Tensor,
    keops: bool
    ) -> float:
  """Computes Gaussian kernel.

  Arguments:
    first_view: torch.Tensor, embedding for first view (sample x low_dim).
    second_view: torch.Tensor, embedding for second view (sample x low_dim).
    sigmas: torch.Tensor, scale parameter of Gaussian kernel.
    keops: bool, whether or not to use keops.

  Returns:
    Gaussian kernel.
  """
  if keops:
    beta = 1.0 / (2.0 * (sigmas.unsqueeze(1)))
    m, d = first_view.shape
    n, d = second_view.shape

    first_view = first_view * torch.sqrt(beta)
    second_view = second_view * torch.sqrt(beta)
    # Turn our dense Tensors into KeOps symbolic variables with "virtual"
    # dimensions at positions 0 and 1 (for "i" and "j" indices):
    first_view_i = LazyTensor(first_view.view(m, 1, d))
    second_view_j = LazyTensor(second_view.view(1, n, d))

    # We can now perform large-scale computations, without memory overflows:
    distance_ij = ((first_view_i - second_view_j)**2).sum(dim=2)
    # Symbolic (m, n, 1) matrix of squared distances
    kernel_ij = (- distance_ij).exp()
    return kernel_ij
  else:
    beta = 1. / (2. * sigmas)
    distances = compute_sqpairwise_distances(first_view, second_view)
    kernel = torch.exp(- beta * distances)
    return kernel


def squared_mmd_keops(
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device
    ) -> float:
  """Computes the Gaussian kernel between the two given views.

  This function is inspired by https://github.com/jeanfeydy/geomloss.

  Arguments:
    first_view: torch.Tensor, first view.
    second_view: torch.Tensor, second view.
    sigmas: torch.Tensor, scale parameter for gaussian kernel.
    device: torch.device, device can be cuda or cpu.

  Returns:
    Squared MMD value.
  """
  n_sample1 = first_view.shape[0]
  n_sample2 = second_view.shape[0]
  ones1 = torch.ones(n_sample1).to(device) / n_sample1
  ones2 = torch.ones(n_sample2).to(device) / n_sample2

  # (B,N,N) tensor
  kernel_11 = gaussian_kernel(
      first_view, first_view, sigmas=sigmas, keops=True)
  # (B,M,M) tensor
  kernel_22 = gaussian_kernel(
      second_view, second_view, sigmas=sigmas, keops=True)
  # (B,N,M) tensor
  kernel_12 = gaussian_kernel(
      first_view, second_view, sigmas=sigmas, keops=True)

  # (B,N,N) @ (B,N) = (B,N)
  kernel_sum_1 = (kernel_11 @ ones1.unsqueeze(-1)).squeeze(-1)
  # (B,M,M) @ (B,M) = (B,M)
  kernel_sum_2 = (kernel_22 @ ones2.unsqueeze(-1)).squeeze(-1)
  # (B,N,M) @ (B,M) = (B,N)
  kernel_sum_12 = (kernel_12 @ ones2.unsqueeze(-1)).squeeze(-1)

  return (
      torch.dot(ones1.view(-1), kernel_sum_1.view(-1))
      + torch.dot(ones2.view(-1), kernel_sum_2.view(-1))
      - 2 * torch.dot(ones1.view(-1), kernel_sum_12.view(-1))
  )


# Functions in primal.
def pen_primal(param: torch.Tensor, device: torch.device) -> torch.Tensor:
  """Computes non-collapsing penalty for the primal formulation.

  In the primal formulation, the 'param' matrix is the n*low_dim model parameter
  and the non-collapsing penalty is equal to ||param.T * param - I||,
  where I is the low_dim*low_dim identity matrix.

  Arguments:
    param: torch.Tensor, model parameters.
    device: torch.device, whether cuda or cpu.
  Returns:
    penalty_value: torch.Tensor, scalar value.
  """
  low_dim = param.shape[1]
  identity_matrix_embedding_space = torch.eye(low_dim).to(device)
  penalty_value = (torch.matmul(param.t(), param)
                   - identity_matrix_embedding_space).norm(2)
  return penalty_value


def dis_primal(
    input_view: torch.Tensor,
    param: torch.Tensor
) -> torch.Tensor:
  """Computes distortion penalty for the primal formulation.

  In the primal formulation, the 'param' matrix is the n*low_dim model parameter
  and input_view corresponds to the input data, of shape n*p. The distortion
  penalty can be written as
  ||input_view*input_view.T - input_view*param*param.T*input_view.T|| which
  can be rewritten as sqrt(Tr((I - param*param.T)*input_view.T*input_view
  *(I - param*param.T)*input_view.T*input_view)) to
  avoid computing terms that are O(n**2) in memory or runtime.

  Arguments:
    input_view: torch.Tensor, one of the two views.
    param: torch.Tensor, model parameters.
  Returns:
    distortion_value: torch.Tensor, scalar value.
  """
  gram = torch.matmul(input_view.t(), input_view)
  tmp = torch.matmul(param, torch.matmul(param.t(), gram))
  prod = gram - tmp
  distortion_value = torch.sqrt(torch.trace(torch.matmul(prod, prod)))
  return distortion_value


# Others.
def compute_sqpairwise_distances(
    first_view: torch.Tensor,
    second_view: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
  """Computes squared euclidean pairwise distances.

  Args:
    first_view: numpy array, format is samples_x x features.
    second_view: numpy array, format is samples_y x features.
  Returns:
    jax numpy array, format is samples_y x samples_x.
  """
  if second_view is None:
    sample_norm_first_view = torch.sum(
        torch.square(first_view), axis=1).reshape(-1, 1)
    cross_inner_product = torch.matmul(first_view, first_view.t())
    diff = (sample_norm_first_view
            + sample_norm_first_view.reshape(1, -1) - 2 * cross_inner_product)
    diff -= torch.diag(torch.diag(diff))
  else:
    sample_norm_first_view = torch.sum(
        torch.square(first_view), axis=1).reshape(-1, 1)
    sample_norm_second_view = torch.sum(
        torch.square(second_view), axis=1).reshape(1, -1)
    cross_inner_product = torch.matmul(first_view, second_view.t())
    diff = (sample_norm_first_view
            + sample_norm_second_view - 2 * cross_inner_product)
  diff = torch.clip(diff, min=0)
  return diff


# Output function.
def pca(first_view, second_view, low_dim=2):
  """Project the samples into a lower dimension with PCA."""
  pca_fn = decomposition.PCA(n_components=low_dim)
  n = first_view.shape[0]
  views = np.concatenate((first_view, second_view), axis=0)
  pca_rep = pca_fn.fit_transform(views)
  pca_rep_fv = pca_rep[:n]
  pca_rep_sv = pca_rep[n:]
  return pca_rep_fv, pca_rep_sv


