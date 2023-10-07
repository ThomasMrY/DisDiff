# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""Tests for beta_vae.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from data.ground_truth import dummy_data
from evaluation.metrics import beta_vae
import numpy as np


class BetaVaeTest(absltest.TestCase):

  def test_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: x
    random_state = np.random.RandomState(0)
    scores = beta_vae.compute_beta_vae_sklearn(
        ground_truth_data, representation_function, random_state, None, 5,
        2000, 2000)
    self.assertBetween(scores["train_accuracy"], 0.9, 1.0)
    self.assertBetween(scores["eval_accuracy"], 0.9, 1.0)


if __name__ == "__main__":
  absltest.main()
