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
import math
from typing import Optional, List
from lsmmdma.mmdma_functions import compute_sqpairwise_distances
import numpy as np
from pykeops.torch import LazyTensor
from sklearn.neighbors import KNeighborsClassifier
import torch


@dataclasses.dataclass()
class OutputSupervised():
  foscttm: float
  top1: float
  top5: float
  no: float
  lta: float


# Evaluation.
@dataclasses.dataclass()
class Evaluation():
  """Computes evaluation metrics, knowing the ground truth alignment.

  Five metrics are being computed:
  - FOSCTTM (Fraction Of Samples Closer Than the True Match): it gives the
  average number of samples from one view that are closer to the true match
  in the other view, in the learned low dimensional space. As this metrics is
  not symmetrical if we exchange the first and second views, the results are
  averaged calculating the FOSCTTM in both directions. Two implementations are
  available, `_foscttm` and `_foscttm_keops`, the latter scaling to several
  hundreds of samples.
  - topK (top1 and top5): it computes the fraction of samples from one view
  correctly assigned among top K% nearest neighbours of the true match from the
  other view. As this metrics is not symmetrical if we exchange the first and
  second views, the results are averaged calculating top1 in both directions.
  K is set to be 1 and 5 by default.
  - Neighbourhood Overlap: it computes the fraction of samples from one view
  correctly assigned among the top K nearest neighbours of the true match from
  the other view, for all K. The output is a vector of dimension K.
  - Label Transfer Accuracy: assigns the label to samples from the second_view
  (resp. first_view) based on the 5-nearest neighbours from the first_view
  (resp. seconv_view). Calculates accuracy of the predicted labels.
  """
  ground_truth_alignment: Optional[np.ndarray] = None
  cell_labels: Optional[List[np.ndarray]] = None
  n_neighbours: int = 5
  device: torch.device = torch.device('cuda')
  short: bool = False

  def compute_all_evaluation(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor
      ) -> OutputSupervised:
    """Computes evaluation measures, knowing the ground truth alignment."""
    if self.ground_truth_alignment is not None:
      second_view_aligned = second_view[self.ground_truth_alignment]

    n = first_view.shape[0]
    m = second_view.shape[0]

    foscttm = -1
    top1 = -1
    top5 = -1
    neigh_overlap = -1
    lta = -1

    if self.ground_truth_alignment is not None:
      try:
        if n == m and n < 5000:
          foscttm = self._foscttm(first_view, second_view_aligned)
        elif n == m:
          foscttm = self._foscttm_keops(first_view, second_view_aligned)
        else:
          logging.warning(f'FOSCTTM was not computed because {n} != {m}.')
      except:
        logging.warning(
            'FOSCTTM was not computed and most likely led to an OOM issue.')

      if not self.short:
        try:
          top1 = self._topx_keops(
              first_view, second_view_aligned, topk=1, percentage=True)
        except:
          logging.warning(
              'Top1 can not be computed with a number of samples <100.')

        try:
          top5 = self._topx_keops(
              first_view, second_view_aligned, topk=5, percentage=True)
        except:
          logging.warning(
              'Top5 can not be computed with a number of samples <20.')

        try:
          if n == m:
            if n * m < 1e7:
              neigh_overlap = self._neighbour_overlap(
                  first_view, second_view_aligned)
            else:
              logging.warning("""Switching to batch version""")
              neigh_overlap = self._neighbour_overlap_batch(
                  first_view, second_view_aligned)
        except:
          logging.warning(f"""Neighbourhood overlap was not computed, either
                          because {n} != {m} or because it might have led to an
                          OOM issue""")
      else:
        logging.info("""TopK and NO were not computed.""")

    if self.cell_labels is not None:
      try:
        lta1 = self._label_transfer_accuracy(
            first_view, second_view, self.cell_labels, k=self.n_neighbours)
        lta2 = self._label_transfer_accuracy(
            second_view, first_view, self.cell_labels[::-1],
            k=self.n_neighbours)
        lta = (lta1 + lta2) / 2.0
      except:
        logging.warning("""LTA was not computed because it might have led to
                        an OOM issue""")

    return OutputSupervised(foscttm, top1, top5, neigh_overlap, lta)

  def _foscttm(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor
      ) -> float:
    """Computes the fraction of samples closer to the true match based on
    squared euclidean distances between samples."""
    n = first_view.shape[0]
    # Assumes the views are aligned.
    distances = compute_sqpairwise_distances(first_view, second_view)
    fraction = (
        torch.sum(distances < torch.diag(distances))
        + torch.sum(torch.t(distances) < torch.diag(distances))
        ) / (2 * n * (n - 1))
    return fraction.item()

  def _foscttm_keops(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor,
    ) -> float:
    """Computes the fraction of samples closer to the true match based on
    squared euclidean distances between samples."""
    n, d = first_view.shape

    first_view_i = LazyTensor(first_view.view(n, 1, d))
    second_view_j = LazyTensor(second_view.view(1, n, d))

    distance_ij = ((first_view_i - second_view_j)**2).sum(dim=2)

    diagonal = ((first_view - second_view)**2).sum(axis=1)
    diagonal1 = LazyTensor(diagonal.view(-1, 1, 1))
    diagonal2 = LazyTensor(diagonal.view(1, -1, 1))

    cttm1 = (diagonal1 - distance_ij).sign().relu()
    cttm2 = (diagonal2 - distance_ij).sign().relu()
    cttm1 = cttm1.sum(1)
    cttm2 = cttm2.sum(1)
    return (cttm1.sum().item() + cttm2.sum().item()) / (n * (n - 1) * 2)

  def _topx_keops(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor,
      topk: int = 1,
      percentage: bool = True
      ) -> float:
    """Computes fraction of samples correctly assigned among topk (%) nearest
    neighbours."""
    def get_count_knn(distance: LazyTensor, n_sample, dim: int = 0):
      # Grid <-> Samples, (M**2, K) integer tensor.
      if percentage:
        indknn = distance.argKmin(int(topk * n_sample / 100), dim=dim)
      else:
        indknn = distance.argKmin(topk, dim=dim)
      frac = indknn - torch.arange(n_sample).reshape(-1, 1).to(self.device)
      return torch.count_nonzero(frac == 0).item()

    n = first_view.shape[0]
    m = second_view.shape[0]
    first_view_i = LazyTensor(first_view[:, None, :])  # (M**2, 1, 2)
    second_view_j = LazyTensor(second_view[None, :, :])  # (1, N, 2)
    # (M**2, N) symbolic matrix of squared distances.
    distance_ij = ((first_view_i - second_view_j) ** 2).sum(-1)
    count_nn0 = get_count_knn(distance_ij, n, dim=0)
    count_nn1 = get_count_knn(distance_ij, m, dim=1)
    return (count_nn0 / m + count_nn1 / n) / 2.0

  def _neighbour_overlap(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor,
      ) -> float:
    """Computes the neighbourhood overlap metric.

    The Neihgbourhood Overlap is the fraction of samples that have among
    their k-neighbours the true match. This function assumes that the two
    datasets are aligned (sample1 in first_view corresponds to sample1 in
    second_view) and have the same number of cells.
    """
    n = first_view.shape[0]
    distances = compute_sqpairwise_distances(first_view, second_view)
    ranking = torch.diagonal(torch.argsort(distances, dim=1))
    cumul_rank = [torch.sum(ranking <= rank).item() for rank in range(n)]
    cumul_frac = np.array(cumul_rank) / n
    return cumul_frac

  def _neighbour_overlap_batch(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor,
      batch_size: int = 1024,
      ) -> float:
    """Computes the neighbourhood overlap metrics by batches of the first_view.

    The Neihgbourhood Overlap is the fraction of samples that have among
    their k-neighbours the true match. This function assumes that the two
    datasets are aligned (sample1 in first_view corresponds to sample1 in
    second_view) and have the same number of cells.
    """
    n = first_view.shape[0]
    n_batch = math.ceil(n / batch_size)
    cumul_frac = np.zeros(n)
    for i in range(n_batch):
      distances = compute_sqpairwise_distances(
          first_view[i * batch_size: (i+1) * batch_size], second_view)
      ranking = torch.diagonal(torch.argsort(distances, dim=1))
      cumul_rank = [torch.sum(ranking <= rank).item() for rank in range(n)]
      cumul_frac += np.array(cumul_rank)
    return cumul_frac / n

  def _label_transfer_accuracy(
      self,
      first_view: torch.FloatTensor,
      second_view: torch.FloatTensor,
      cell_labels: List[np.ndarray],
      k: int = 5):
    """Computes the Label Transfer Accuracy metrics."""
    first_view = first_view.detach().cpu()
    second_view = second_view.detach().cpu()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(first_view, cell_labels[0])
    predictions = knn.predict(second_view)
    count = 0
    for label1, label2 in zip(predictions, cell_labels[1]):
      if label1 == label2:
        count += 1
    return count / second_view.shape[0]

