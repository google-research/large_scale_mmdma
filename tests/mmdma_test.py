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

"""Tests for MMDMA algorithm in PyTorch."""
from absl import flags
from absl.testing import absltest
from lsmmdma.data import data_pipeline
from lsmmdma.metrics import Evaluation
import lsmmdma.mmdma_functions as mmdma_fn
import lsmmdma.train as mmdma_core
import numpy as np
import torch

flags.DEFINE_string('output_dir', '', 'Output directory.')
FLAGS = flags.FLAGS


class MMDMATest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(1)
    self.n = 3
    self.p = 10
    self.low_dim = 2
    self.sigmas = torch.FloatTensor([0.25])

    self.view1 = torch.FloatTensor(np.random.randn(self.n, self.p))
    self.view2 = torch.FloatTensor(np.random.randn(self.n, self.p)) + 2

    self.kernel1 = mmdma_core.get_kernel(self.view1)
    self.kernel2 = mmdma_core.get_kernel(self.view2)

    self.param1_dual = torch.FloatTensor(np.random.randn(self.n, self.low_dim))
    self.param1_primal = torch.matmul(self.view1.t(), self.param1_dual)
    self.param2_dual = torch.FloatTensor(np.random.randn(self.n, self.low_dim))
    self.param2_primal = torch.matmul(self.view2.t(), self.param2_dual)

    self.embedding1_primal = torch.matmul(self.view1, self.param1_primal)
    self.embedding1_dual = torch.matmul(self.kernel1, self.param1_dual)

    self.device = torch.device('cpu')

    self.cfg_model = mmdma_core.ModelGetterConfig(
        seed=2,
        low_dim=self.low_dim,
        n_iter=6,
        keops=1,
        mode='dual',
        n_eval=5,
        n_record=5,
        learning_rate=1e-3,
        sigmas=self.sigmas,
        lambda1=0.001,
        lambda2=0.001,
        init='uniform',
        use_unbiased_mmd=True
        )

  def test_distortion(self):
    # n < p
    dis_val_primal = mmdma_fn.dis_primal(self.view1, self.param1_primal)
    dis_val_dual = mmdma_fn.dis_dual(
        self.embedding1_dual, self.kernel1
        ) * 1 / np.sqrt(self.p)

    self.assertAlmostEqual(dis_val_primal, dis_val_dual, delta=1e-1)

    # n > p
    dis_val_primal = mmdma_fn.dis_primal(self.view1[:, :2],
                                         self.param1_primal[:2, :])
    kernel = mmdma_core.get_kernel(self.view1[:, :2])
    embedding = torch.matmul(kernel, self.param1_dual)
    dis_val_dual = mmdma_fn.dis_dual(embedding, kernel) * 1 / np.sqrt(2)

    self.assertAlmostEqual(dis_val_primal, dis_val_dual, delta=1e-3)

  def test_penalty(self):
    pen_val_primal = mmdma_fn.pen_primal(self.param1_primal, self.device)
    pen_val_dual = mmdma_fn.pen_dual(
        self.embedding1_dual, self.param1_dual, self.device
        ) * 1 / np.sqrt(self.p)

    self.assertAlmostEqual(pen_val_primal, pen_val_dual, delta=1e-3)

  def test_squared_mmd(self):
    mmd_keops = mmdma_fn.squared_mmd(
        self.view1, self.view2, self.sigmas, True,
        self.cfg_model.use_unbiased_mmd, self.device)
    mmd = mmdma_fn.squared_mmd(
        self.view1, self.view2, self.sigmas, False,
        self.cfg_model.use_unbiased_mmd, self.device)

    self.assertAlmostEqual(mmd_keops, mmd, delta=1e-5)

    mmd = mmdma_fn.squared_mmd(
        self.view1, self.view1, self.sigmas, True,
        self.cfg_model.use_unbiased_mmd, self.device)

    # self.assertEqual(mmd, 0.0)

  def test_loss(self):
    loss_dual = [0]*2
    loss_primal = [0]*2
    cfg_model = self.cfg_model
    l1 = cfg_model.lambda1
    l2 = cfg_model.lambda2
    for i, keops in enumerate([1, 0]):
      cfg_model.mode = 'dual'
      cfg_model.lambda1 = l1 * 1 / np.sqrt(self.p)
      cfg_model.lambda2 = l2 * 1 / np.sqrt(self.p)
      loss_dual[i] = mmdma_core.loss_mmdma(
          [self.param1_dual, self.param2_dual], self.kernel1, self.kernel2,
          self.n, self.n, cfg_model, keops, self.device)[0]
      cfg_model.mode = 'primal'
      cfg_model.lambda1 = l1
      cfg_model.lambda2 = l2
      loss_primal[i] = mmdma_core.loss_mmdma(
          [self.param1_primal, self.param2_primal], self.view1, self.view2,
          self.n, self.n, cfg_model, keops, self.device)[0]

    self.assertAlmostEqual(loss_dual[0], loss_dual[1], delta=1e-2)
    self.assertAlmostEqual(loss_primal[0], loss_primal[1], delta=1e-5)
    self.assertAlmostEqual((loss_dual[0] - loss_primal[0]) / loss_primal[0],
                           0, delta=1e-4)

  def test_keops_and_unbiased(self):
    n = 200
    p = 10

    view1 = torch.FloatTensor(np.random.randn(n, p))
    view2 = torch.FloatTensor(np.random.randn(n, p)) + 2.0
    kernel1 = mmdma_core.get_kernel(view1)
    kernel2 = mmdma_core.get_kernel(view2)

    rd_vec = np.arange(n)
    eval_fn = Evaluation(ground_truth_alignment=rd_vec,
                         device=self.device)

    cfg_model = self.cfg_model
    cfg_model.n_iter = 50
    for use_unbiased_mmd in [True, False]:
      cfg_model.use_unbiased_mmd = use_unbiased_mmd
      for mode in ['dual', 'primal']:
        cfg_model.mode = mode
        loss = []
        mmd = []
        res = []
        for keops in [1, 0]:
          cfg_model.keops = keops
          # Runs model.
          if mode == 'dual':
            out = mmdma_core.train_and_evaluate(
                cfg_model, kernel1, kernel2, eval_fn,
                workdir='', device=self.device)
          else:
            out = mmdma_core.train_and_evaluate(
                cfg_model, view1, view2, eval_fn, workdir='',
                device=self.device)
          _, _, evaluation_loss, evaluation_matching, _, _, _ = out

          loss.append(evaluation_loss['loss'][-1])
          mmd.append(evaluation_loss['mmd'][-1])
          res.append(evaluation_matching['foscttm'][-1])

        self.assertAlmostEqual((mmd[0] - mmd[1]) / mmd[1], 0, delta=0.2)
        self.assertAlmostEqual(res[0], res[1], delta=1e-3)
        self.assertAlmostEqual((loss[0] - loss[1]) / loss[1], 0, delta=0.2)

  def test_metrics(self):
    n = 200
    view1, _, _ = data_pipeline.generate_data(None, n, 10)
    eval_fn = Evaluation(ground_truth_alignment=np.arange(n),
                         device=self.device)
    out = eval_fn.compute_all_evaluation(view1, view1)

    self.assertEqual(out.foscttm, 0.)
    self.assertEqual(out.top1, 1.)
    self.assertEqual(out.top5, 1.)

  def test_squared_distance(self):
    distance1 = mmdma_fn.compute_sqpairwise_distances(self.view1, self.view2)
    distance2 = torch.sum(
        (self.view1[:, None, :] - self.view2[None, :, :])**2, axis=-1)

    self.assertAlmostEqual((distance1 - distance2).sum().item(), 0., delta=1e-4)
    self.assertAlmostEqual(
        distance1[1, 2].item(), distance2[1, 2].item(), delta=1e-5)
    self.assertAlmostEqual(
        distance1[2, 0].item(), distance2[2, 0].item(), delta=1e-5)

  def test_unbiased_functions(self):
    # Testing biased MMD for keops and without keops are equal
    mmd_a = mmdma_fn.squared_mmd(
        self.view1, self.view2, self.sigmas, False, False, self.device)
    mmd_b = mmdma_fn.squared_mmd(
        self.view1, self.view2, self.sigmas, True, False, self.device)
    self.assertAlmostEqual(mmd_a, mmd_b, delta=1e-4)

    # Testing unbiased MMD for keops and without keops are equal
    mmd_a = mmdma_fn.squared_mmd(
        self.view1, self.view2, self.sigmas, False,
        self.cfg_model.use_unbiased_mmd, self.device)
    mmd_b = mmdma_fn.squared_mmd(
        self.view1, self.view2, self.sigmas, True,
        self.cfg_model.use_unbiased_mmd, self.device)
    self.assertAlmostEqual(mmd_a, mmd_b, delta=1e-4)

  def test_evaluation_short(self):
    n = 10
    p = 2
    view1 = torch.FloatTensor(np.random.randn(n, p))
    view2 = torch.FloatTensor(np.random.randn(n, p)) + 2
    rd_vec = np.arange(n)
    eval_fn = Evaluation(
        ground_truth_alignment=rd_vec, device=self.device, short=True)
    out = eval_fn.compute_all_evaluation(view1, view2)

    self.assertNotEqual(out.foscttm, -1)
    self.assertEqual(out.lta, -1)
    self.assertEqual(out.no, -1)
    self.assertEqual(out.top1, -1)

  def test_evaluation_celllabels(self):
    n = 10
    p = 2
    view1 = torch.FloatTensor(np.random.randn(n, p))
    view2 = torch.FloatTensor(np.random.randn(n, p)) + 2
    cell_labels_fv = np.random.choice(2, n)
    cell_labels_sv = np.random.choice(2, n)
    rd_vec = np.arange(n)
    eval_fn = Evaluation(
        ground_truth_alignment=rd_vec, device=self.device,
        cell_labels=[cell_labels_fv, cell_labels_sv], short=True)
    out = eval_fn.compute_all_evaluation(view1, view2)

    self.assertNotEqual(out.lta, -1)

  def test_evaluation_long(self):
    n = 30
    p = 2
    view1 = torch.FloatTensor(np.random.randn(n, p))
    view2 = torch.FloatTensor(np.random.randn(n, p)) + 2
    rd_vec = np.arange(n)
    eval_fn = Evaluation(
        ground_truth_alignment=rd_vec, device=self.device, short=False)
    out = eval_fn.compute_all_evaluation(view1, view2)

    self.assertNotEqual(out.top5, -1)
    self.assertNotEqual(out.no[0], -1)

  def test_evaluation_foscttm(self):
    n = 10
    p = 2
    view1 = torch.FloatTensor(np.random.randn(n, p))
    view2 = torch.FloatTensor(np.random.randn(n, p)) + 2
    rd_vec = np.arange(n)
    eval_fn = Evaluation(
        ground_truth_alignment=rd_vec, device=self.device, short=False)
    out = eval_fn.compute_all_evaluation(view1, view2)
    foscttm_keops = eval_fn._foscttm_keops(view1, view2)
    self.assertAlmostEqual(foscttm_keops, out.foscttm, delta=1e-3)

if __name__ == '__main__':
  absltest.main()
