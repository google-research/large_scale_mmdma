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

"""Evaluation metrics."""
import dataclasses
import logging
from lsmmdma.mmdma_functions import compute_sqpairwise_distances
import numpy as np
from pykeops.torch import LazyTensor
import scipy
import torch


@dataclasses.dataclass()
class OutputSupervised():
  foscttm: float
  top1: float
  top5: float


# Supervised evaluation.
@dataclasses.dataclass()
class SupervisedEvaluation():
  """Computes evaluation metrics, knowing the ground truth alignment.

  Three metrics are being computed:
  - FOSCTTM (Fraction Of Samples Closer Than the True Match): it gives the
  average number of samples from one view that are closer to the true match
  in the other view, in the learned low dimensional space. As this metrics is
  not symmetrical if we exchange the first and second views, the results are
  averaged calculating the FOSCTTM in both directions.
  - topK: it computes the fraction of samples from one view correctly
  assigned among top K% nearest neighbours of the true match from the other
  view. As this metrics is not symmetrical if we exchange the first and
  second views, the results are averaged calculating top1 in both directions.
  K is set to be 1 and 5 by default.
  """
  ground_truth_alignment: np.ndarray
  device: torch.device

  def compute_all_evaluation(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor
      ) -> OutputSupervised:
    """Computes evaluation measures, knowing the ground truth alignment."""
    second_view_aligned = second_view[self.ground_truth_alignment]

    n = first_view.shape[0]
    try:
      foscttm = self._foscttm(first_view, second_view_aligned)
    except:
      logging.warning(
          'FOSCTTM was not computed and most likely led to an OOM issue.')
      foscttm = -1
    if n > 100:
      top1 = self._topx_keops(
          first_view, second_view_aligned, topk=int(1 / 100 * n))
    else:
      logging.warning(
          'Top1 can not be computed with a number of samples <100.')
      top1 = -1
    if n > 20:
      top5 = self._topx_keops(
          first_view, second_view_aligned, topk=int(5 / 100 * n))
    else:
      logging.warning(
          'Top5 can not be computed with a number of samples <20.')
      top5 = -1
    return OutputSupervised(foscttm, top1, top5)

  def _foscttm(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor
      ) -> float:
    """Computes the fraction of samples closer to the true match based on
    squared euclidean distances between samples."""
    # TODO(lpapaxanthos): Memory efficient FOSCTTM.
    n = first_view.shape[0]
    # Assumes the views are aligned.
    distances = compute_sqpairwise_distances(first_view, second_view)
    fraction = (
        torch.sum(distances < torch.diag(distances))
        + torch.sum(torch.t(distances) < torch.diag(distances))
        ) / (2 * n * (n - 1))
    return fraction.item()

  def _topx_keops(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor,
      topk: int = 1
      ) -> float:
    """Computes fraction of samples correctly assigned among topk nearest
    neighbours."""
    def get_count_knn(distance: LazyTensor, dim: int = 0):
      # Grid <-> Samples, (M**2, K) integer tensor.
      indknn = distance.argKmin(topk, dim=dim)
      frac = indknn - torch.arange(n).reshape(-1, 1).to(self.device)
      return torch.count_nonzero(frac == 0).item()
    n = first_view.shape[0]
    first_view_i = LazyTensor(first_view[:, None, :])  # (M**2, 1, 2)
    second_view_j = LazyTensor(second_view[None, :, :])  # (1, N, 2)
    # (M**2, N) symbolic matrix of squared distances.
    distance_ij = ((first_view_i - second_view_j) ** 2).sum(-1)
    count0 = get_count_knn(distance_ij, dim=0)
    count1 = get_count_knn(distance_ij, dim=1)
    return (count0 + count1) / (2 * n)
