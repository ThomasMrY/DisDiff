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

"""DSprites dataset and new variants with probabilistic decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from data.ground_truth import ground_truth_data
from data.ground_truth import util
import numpy as np
import PIL
from six.moves import range



class Dataset(ground_truth_data.GroundTruthData):
  """DSprites dataset.

  The data set was originally introduced in "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework" and can be downloaded from
  https://github.com/deepmind/dsprites-dataset.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, images, latent_factor_indices=[1, 2, 3, 4, 5], factor_sizes = [ 1,  3,  6, 40, 32, 32]):
    # By default, all factors (including shape) are considered ground truth
    # factors.
    if latent_factor_indices is None:
      latent_factor_indices = list(range(6))
    self.latent_factor_indices = latent_factor_indices
    self.data_shape = [64, 64, 1]
    # Load the data so that we can sample from it.
    self.images = images
    self.factor_sizes = np.array(
        factor_sizes, dtype=np.int64)
    self.full_factor_sizes = factor_sizes
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    return self.sample_observations_from_factors_no_color(factors, random_state)

  def sample_observations_from_factors_no_color(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    return self.images[indices]

  def _sample_factor(self, i, num, random_state):
    return random_state.randint(self.factor_sizes[i], size=num)
