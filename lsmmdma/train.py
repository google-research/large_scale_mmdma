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

"""Implementation of a large-scale version of MMD-MA with PyTorch.

MMD-MA has been developped by Liu et al. 2019, Jointly Embedding Multiple
Single-Cell Omics Measurements (https://pubmed.ncbi.nlm.nih.gov/34632462/).
"""

import random
from absl import logging

from collections import defaultdict as ddict
import dataclasses
from lsmmdma.initializers import initialize
from lsmmdma.metrics import Evaluation
import lsmmdma.mmdma_functions as mmdma_fn
from math import ceil
import numpy as np
from tenacity import retry, stop_after_attempt
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils import tensorboard
from typing import Tuple, Any, Dict, DefaultDict, List, Union


@dataclasses.dataclass(unsafe_hash=True)
class ModelGetterConfig:
  """Model attributes."""
  seed: int
  learning_rate: float = 1e-3
  low_dim: int = 5
  n_record: int = 10
  n_eval: int = 10
  sigmas: float = 1.
  lambda1: float = 1.
  lambda2: float = 1.
  n_iter: int = 1000
  mode: str = 'dual'
  keops: bool = False
  pca: bool = False
  n_seed: int = 1
  init: str = 'uniform'
  use_unbiased_mmd: bool = True


class Model(nn.Module):
  """Model for MMD-MA, defines learned parameters and computes the loss.

  In MMD-MA, we project two sets of points, from two different spaces endowed
  with a positive definite kernel, to a shared Euclidean space of dimension
  low_dim. The mappings from high to low dimensional space is
  obtained using functions belonging to the respective RKHS. To obtain the
  mappings, we minimise a loss function that is composed of three terms:
  - an MMD term between the low dimensional representations of the two views,
  which encourages them to have the same distribution.
  - two non-collapsing penalty terms (corresponding to the pen_dual or
  pen_primal functions), one for each view. These terms ensure that
  the low dimensional representations are mutually orthogonal, preventing
  collapsing.
  - two distortion penalties (corresponding to the dis_dual or
  dis_primal functions), one for each view. These terms encourage the
  low dimensional representation to obtain the same pairwise structure as
  the original views.

  MMD-MA can be formulated using either the primal (when we use the linear
  kernel in the input spaces) or the dual problem. Each has
  advantages or disadvantages depending on the input data. Let D1 and D2 be two
  sets of points, of respective size n_sample1*p_feature1 and
  n_sample2*p_feature2, where n_sample1 (resp. n_sample2) corresponds to
  the number of samples of the first set of points D1 (resp.
  the second set of points D2) and p_feature1 (resp. p_feature2)
  corresponds to the number of features of the first set of points D1
  (resp. the second one D2). In each view, when the
  number of features is larger than the number of samples
  p_featureX >> n_sampleX, then the dual formulation is beneficial in terms
  of runtime and memory, while if n_sampleX >> p_sampleX, the primal
  formulation is favorable.

  The original implementation of MMD-MA (see Liu et al.) uses the dual
  formulation. The resulting loss can be written as follows:
  min_{W1, W2} Loss(W1, W2)
        = MMD(K1W1, K2W1)
        + lambda1(||W1.TK1.TW1 - I|| + ||W2.TK2.TW2 - I||)
        + lambda2(||K1W1W1.TK1.T - K1|| + ||K2W2W2.TK2.T - K2||)
  where W1 and W2 are the set of weights that we wish to learn, K1 and K2 are
  kernels of the input views D1 and D2, and I are identity matrices. The dual
  form has the advantage of only involving the kernels K1 and K2 and not the
  original sets of points D1 ans D2. Therefore, non linear mappings can also
  be used.

  If the number of samples is much larger than the number of features in both
  views, then the primal formulation can be favorable:
  min_{W1, W2} Loss(W1, W2)
        = MMD(D1W1, D2W1)
        + lambda1(||W1.TW1 - I|| + ||W2.TW2 - I||)
        + lambda2(||D1W1W1.TD1.T - D1D1.T|| + ||D2W2W2.TD2.T - D2D2.T||)
  where W1 and W2 are the model parameters.

  Attributes:
    cfg_model: ModelGetterConfig, sets parameters for the MMDMA algorithm.
    n_sample1: int, number of samples in the first view.
    n_sample2: int, number of samples in the second view.
    p_feature1: int, number of features in the first view.
    p_feature2: int, number of features in the second view.
    keops: bool, whether to use keops when calculating MMD.
    device: torch.device, whether a cpu or gpu is used.
  """

  def __init__(
      self,
      cfg_model: ModelGetterConfig,
      n_sample1: int,
      n_sample2: int,
      p_feature1: int,
      p_feature2: int,
      keops: bool,
      device: torch.device,
      ):
    super(Model, self).__init__()
    self.n_sample1 = n_sample1
    self.n_sample2 = n_sample2
    self.p_feature1 = p_feature1
    self.p_feature2 = p_feature2
    self.cfg_model = cfg_model
    self.keops = keops
    self.device = device

    if self.cfg_model.mode == 'primal':
      param1_dim = self.p_feature1
      param2_dim = self.p_feature2
    elif self.cfg_model.mode == 'dual':
      param1_dim = self.n_sample1
      param2_dim = self.n_sample2
    else:
      raise NotImplementedError

    init_args = cfg_model.init.split(',')
    init_args[1:] = [float(i) for i in init_args[1:]]
    initializer = initialize(init_args[0])
    empty_tensor1 = torch.empty(param1_dim, cfg_model.low_dim)
    empty_tensor2 = torch.empty(param2_dim, cfg_model.low_dim)
    init1 = initializer(empty_tensor1, *init_args[1:])
    init2 = initializer(empty_tensor2, *init_args[1:])
    self.weights_first_view = Parameter(init1)
    self.weights_second_view = Parameter(init2)

  def forward(self,
              first_view: torch.Tensor,
              second_view: torch.Tensor
              ) -> Tuple[float, Tuple[float, ...]]:
    """Forward pass.

    Arguments:
      first_view: torch.Tensor, first view. The dimension of this tensor is
        (sample x feature) if mode == 'primal' and (sample x sample)
        if mode == 'dual'.
      second_view: torch.Tensor, second view. The dimension of this tensor is
        (sample x feature) if mode == 'primal' and (sample x sample)
        if mode == 'dual'.

    Returns:
      loss: float, MMDMA loss
      loss_components: 5-tuple, individual components of the MMDMA loss
    """
    loss, loss_components = loss_mmdma(
        [self.weights_first_view, self.weights_second_view],
        first_view, second_view, self.n_sample1, self.n_sample2,
        self.cfg_model, self.keops, self.device)
    return loss, loss_components


def loss_mmdma(
    params: torch.Tensor,
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    n_first_view: int,
    n_second_view: int,
    cfg_model: ModelGetterConfig,
    keops: bool,
    device: torch.device
    ) -> Tuple[float, Tuple[float, ...]]:
  """Loss of the MMDMA algorithm.

  The loss is composed of an MMD term to which two penalty terms are added. The
  first penalty type (`penalty_xv`) penalises collapsing embedding
  spaces and the second penalty type (`distortion_xv`) penalises
  distortion between the original space and the embedding space.

  Arguments:
    params: torch.Tensor, model parameters.
    first_view: torch.Tensor, first set of points. The dimension of this
      tensor is (sample x feature) if mode == 'primal' and (sample x sample)
      if mode == 'dual'.
    second_view: torch.Tensor, second set of points. The dimension of this
      tensor is (sample x feature) if mode == 'primal' and (sample x sample)
      if mode == 'dual'.
    n_first_view: int, number of samples in the first view.
    n_second_view: int, number of samples in the second view.
    cfg_model: ModelGetterConfig, sets parameters for the MMDMA algorithm.
    keops: bool, whether to use keops when calculating MMD.
    device: torch.device, whether a cpu or gpu ('cuda') is used.

  Returns:
    tuple containing the loss and the individual components of the loss
  """
  embedding_fv = torch.matmul(first_view, params[0])
  embedding_sv = torch.matmul(second_view, params[1])
  mmd = mmdma_fn.squared_mmd(
      embedding_fv, embedding_sv, cfg_model.sigmas, keops,
      cfg_model.use_unbiased_mmd, device)
  if cfg_model.mode == 'primal':
    penalty_fv = mmdma_fn.pen_primal(params[0], device)
    penalty_sv = mmdma_fn.pen_primal(params[1], device)
    distortion_fv = mmdma_fn.dis_primal(
        first_view, params[0], n_first_view)
    distortion_sv = mmdma_fn.dis_primal(
        second_view, params[1], n_second_view)
  elif cfg_model.mode == 'dual':
    penalty_fv = mmdma_fn.pen_dual(embedding_fv, params[0], device)
    penalty_sv = mmdma_fn.pen_dual(embedding_sv, params[1], device)
    distortion_fv = mmdma_fn.dis_dual(embedding_fv, first_view)
    distortion_sv = mmdma_fn.dis_dual(embedding_sv, second_view)
  else:
    raise ValueError('cfg_model.mode must be "dual" or "primal."')
  return (mmd + cfg_model.lambda1 * (penalty_fv + penalty_sv)
          + cfg_model.lambda2 * (distortion_fv + distortion_sv),
          (mmd.item(), penalty_fv.item(), penalty_sv.item(),
           distortion_fv.item(), distortion_sv.item()))


def get_kernel(data: torch.Tensor, kernel_type='linear') -> torch.Tensor:
  """Gives linear kernel needed in the dual mode.

  Arguments:
    data: torch.Tensor, one of the two views (sample x feature).
    kernel_type: str, 'linear' by default but other kernels can be implemented.

  Returns:
    kernel: torch.Tensor, linear kernel of one of the two views
      (sample x sample).
  """
  if kernel_type == 'linear':
    kernel = torch.matmul(data, data.T)
  else:
    raise NotImplementedError(f'{kernel_type} kernel is not supported yet.')
  return kernel


def save_loss_value(
    loss_tmp: float,
    loss_components_tmp: List[float],
    evaluation: Dict[str, float],
    ) -> Dict[str, float]:
  """Records loss related metrics.

  Arguments:
    loss_tmp: float, loss value.
    loss_components_tmp: tuple, components of the loss.
    evaluation: dictionary, records the output values.

  Returns:
    evaluation: dictionary that records the output values.
  """
  evaluation['loss'].append(loss_tmp.item())
  for i, val in enumerate(
      ['mmd', 'pen_fv', 'pen_sv', 'dis_fv', 'dis_sv']):
    evaluation[val].append(loss_components_tmp[i])
  return evaluation


def save_evaluation(
    evaluation_tmp: Dict[str, float],
    evaluation: Dict[str, float]
    ) -> Dict[str, float]:
  """Appends evaluation values to dictionary.

  Arguments:
    evaluation_tmp: dictionary, current evaluation values.
    evaluation: dictionary, records the output values.

  Returns:
    evaluation: dictionary that records the output values.
  """
  for key, val in evaluation_tmp.items():
    evaluation[key].append(val)
  return evaluation


def _evaluate(
    i: int,
    first_view: Union[torch.FloatTensor, torch.utils.data.DataLoader],
    second_view: Union[torch.FloatTensor, torch.utils.data.DataLoader],
    model: torch.nn.Module,
    eval_fn: Evaluation,
    loss: List[float],
    loss_components: List[List[float]],
    evaluation_loss: DefaultDict[str, List[float]],
    evaluation_matching: DefaultDict[str, List[float]],
    embeddings_results: Tuple[np.ndarray],
    pca_results: Tuple[np.ndarray],
    cfg_model: ModelGetterConfig,
    device: torch.device,
    summary_writer: tensorboard.SummaryWriter,
    workdir: str
    ) -> Tuple[DefaultDict[str, List[float]],
               DefaultDict[str, List[float]], Tuple[np.ndarray]]:
  """Records loss and evaluation metrics, runs PCA on embeddings.

  Arguments:
    i: number of epoch.
    first_view: first point cloud (sample x feature) if primal and first kernel
      (sample x sample) if dual.
    second_view: second point cloud (sample x feature) if primal and second
      kernel (sample x sample) if dual.
    model: mmdma model.
    eval_fn: metrics to evaluate the model.
    loss: loss.
    loss_components: loss components.
    evaluation_loss: records the loss and loss components
      during training.
    evaluation_matching: records the metrics during training.
    embeddings_results: records embeddings during training.
    pca_results: records results of PCA on embeddings during training.
    cfg_model: contains the parameters of the model and algorithm.
    device: either torch.device('cuda') or 'cpu'.
    summary_writer: summary writer for tensorboard.
    workdir: directory where the files are saved.

  Returns:
    evaluation_loss: records the loss and loss components
      during training.
    evaluation_matching: records the metrics during training.
    pca_results: records results of PCA on embeddings during training.
  """
  @retry(stop=stop_after_attempt(3))
  def init_summary():
    if (i == 0
        and (cfg_model.n_record != 0 or cfg_model.n_eval != 0) and workdir):
      summary_writer = tensorboard.SummaryWriter(workdir)
      return summary_writer
    else:
      return None

  if i == 0:
    summary_writer = init_summary()

  weights1, weights2 = list(model.parameters())
  embeddings_fv = torch.matmul(first_view, weights1)
  embeddings_sv = torch.matmul(second_view, weights2)

  # Records loss.
  if cfg_model.n_record != 0 and i % cfg_model.n_record == 0:
    evaluation_loss = save_loss_value(
        loss, loss_components, evaluation_loss)
    if workdir:
      for key, val in evaluation_loss.items():
        summary_writer.add_scalar(f'train/{key:s}', val[-1], i)
      logging.info('Flushing TensorBoard writer.')
      summary_writer.flush()

  # Saves evaluation metrics.
  if cfg_model.n_eval != 0 and i % cfg_model.n_eval == 0:
    eval_out = eval_fn.compute_all_evaluation(embeddings_fv, embeddings_sv)
    eval_out_dict = dataclasses.asdict(eval_out)
    logging.info('Train evaluation: %s.', eval_out)
    if workdir:
      for key, val in eval_out_dict.items():
        summary_writer.add_scalar(f'train/{key:s}', val, i)
        logging.info('Flushing TensorBoard writer.')
    evaluation_matching = save_evaluation(
        eval_out_dict, evaluation_matching)

  # Saves low dimensional representation
  if ((cfg_model.n_eval != 0 and i % cfg_model.n_eval == 0)
      or (cfg_model.n_record != 0 and i % cfg_model.n_record == 0)):
    embeddings_results.append(
        [embeddings_fv.detach().cpu().numpy(),
         embeddings_sv.detach().cpu().numpy()])

  # Saves 2-dimensional representation obtained with PCA.
  if cfg_model.pca != 0 and i % cfg_model.pca == 0:
    pca_rep_fv, pca_rep_sv = mmdma_fn.pca(
        embeddings_fv.detach().cpu().numpy(),
        embeddings_sv.detach().cpu().numpy())
    pca_results.append([pca_rep_fv, pca_rep_sv])

  if (i == cfg_model.n_iter - 1
      and (cfg_model.n_record != 0 or cfg_model.n_eval != 0) and workdir):
    summary_writer.close()
  return (evaluation_loss,
          evaluation_matching,
          embeddings_results,
          pca_results,
          summary_writer)


def train_and_evaluate(
    cfg_model: ModelGetterConfig,
    first_view: Union[torch.FloatTensor, torch.utils.data.DataLoader],
    second_view: Union[torch.FloatTensor, torch.utils.data.DataLoader],
    eval_fn: Evaluation,
    workdir: str,
    device: torch.device,
    ) -> Any:
  """Computes the training loop and evaluation steps.

  Arguments:
    cfg_model: ModelGetterConfig, sets parameters for the MMDMA algorithm.
    first_view: torch.Tensor, first view (sample1 x feature1) if primal and
      (sample1 x sample1) if dual.
    second_view: torch.Tensor, second view (sample2 x feature2) if primal and
      (sample2 x sample2) if dual.
    eval_fn: Evaluation dataclass, applies functions for evaluation.
    workdir: str, path to output directory.
    device: torch.device, 'cuda' or 'cpu'.

  Returns:
    optimizer: torch.optim, optimizer.
    model: model parameters.
    evaluation_loss: dictionary, contains loss components.
    evaluation_matching: list of namedtuple, contains metrics values.
  """
  n_sample1, p_feature1 = first_view.shape
  n_sample2, p_feature2 = second_view.shape

  cfg_model.sigmas = torch.FloatTensor([cfg_model.sigmas]).to(device)

  to_evaluate = (cfg_model.pca != 0
                 or cfg_model.n_record != 0 or cfg_model.n_eval != 0)

  def _train_and_evaluate(
      seed: int,
      evaluation: bool,
      inner_workdir: str = ''
      ) -> Any:

    evaluation_loss = ddict(list)
    evaluation_matching = ddict(list)
    pca_results = list()
    embeddings_results = list()

    torch.manual_seed(seed)
    random.seed(seed + 1)
    np.random.seed(seed + 2)

    summary_writer = None

    model = Model(cfg_model,
                  n_sample1,
                  n_sample2,
                  p_feature1,
                  p_feature2,
                  cfg_model.keops,
                  device)
    model = model.to(device)
    logging.info('seed=%s', seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_model.learning_rate)
    model.train()

    # TODO(lpapaxanthos): stopping criterion.
    for i in range(cfg_model.n_iter):
      optimizer.zero_grad()
      loss, loss_components = model(first_view, second_view)
      loss.backward()
      optimizer.step()
      logging.info('Epoch: %s., train loss: %s.', i, loss)

      if evaluation:
        out = _evaluate(i, first_view, second_view, model, eval_fn, loss,
                        loss_components, evaluation_loss,
                        evaluation_matching, embeddings_results, pca_results,
                        cfg_model, device, summary_writer,
                        workdir=inner_workdir)
        evaluation_loss, evaluation_matching, embeddings_results, pca_results, summary_writer = out

    return (loss, optimizer, model, evaluation_loss, evaluation_matching,
            embeddings_results, pca_results)

  loss_output = np.inf
  seed_output = np.inf

  logging.info(cfg_model)

  # Sampling array of seeds.
  np.random.seed(cfg_model.seed)
  assert cfg_model.n_seed < 1e6, 'cfg_model.n_seed must be smaller than 1e6.'
  array_seeds = np.random.choice(int(1e6), cfg_model.n_seed, replace=False)

  if cfg_model.n_seed > 1:
    for k in array_seeds:
      loss, optimizer, model, *_ = _train_and_evaluate(
          k, evaluation=False, inner_workdir='')

      if loss < loss_output:
        loss_output = loss
        seed_output = k

  seed_output = seed_output if cfg_model.n_seed > 1 else cfg_model.seed
  out = _train_and_evaluate(seed_output,
                            evaluation=to_evaluate,
                            inner_workdir=workdir)
  _, optimizer, model, evaluation_loss, evaluation_matching, embeddings_results, pca_results = out
  return (
      optimizer,
      model,
      evaluation_loss,
      evaluation_matching,
      embeddings_results,
      pca_results,
      seed_output
      )
