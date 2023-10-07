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

"""Cars3D data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from data.ground_truth import ground_truth_data
from data.ground_truth import util
import numpy as np
import PIL
import scipy.io as sio
from six.moves import range
from sklearn.utils.extmath import cartesian



class Dataset(ground_truth_data.GroundTruthData):
  """Cars3D data set.

  The data set was first used in the paper "Deep Visual Analogy-Making"
  (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
  downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.

  The ground-truth factors of variation are:
  0 - elevation (4 different values)
  1 - azimuth (24 different values)
  2 - object type (183 different values)
  """

  def __init__(self, images):
    self.factor_sizes = [4, 24, 183]
    features = cartesian([np.array(list(range(i))) for i in self.factor_sizes])
    self.latent_factor_indices = [0, 1, 2]
    self.num_total_factors = features.shape[1]
    self.index = util.StateSpaceAtomIndex(self.factor_sizes, features)
    self.factor_bases = self.index.factor_bases
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)
    self.data_shape = [64, 64, 3]
    self.images = images

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return self.factor_sizes

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = self.index.features_to_index(all_factors)
    return self.images[indices]
