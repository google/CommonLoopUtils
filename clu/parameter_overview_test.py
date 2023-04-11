# Copyright 2023 The CLU Authors.
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

"""Tests for parameter overviews."""

from absl.testing import absltest
from clu import parameter_overview
from flax import linen as nn
import jax
import jax.numpy as jnp


EMPTY_PARAMETER_OVERVIEW = """+------+-------+------+------+-----+
| Name | Shape | Size | Mean | Std |
+------+-------+------+------+-----+
+------+-------+------+------+-----+
Total: 0"""

FLAX_CONV2D_PARAMETER_OVERVIEW = """+-------------+--------------+------+
| Name        | Shape        | Size |
+-------------+--------------+------+
| conv/bias   | (2,)         | 2    |
| conv/kernel | (3, 3, 3, 2) | 54   |
+-------------+--------------+------+
Total: 56"""

FLAX_CONV2D_PARAMETER_OVERVIEW_WITH_STATS = """+-------------+--------------+------+------+-----+
| Name        | Shape        | Size | Mean | Std |
+-------------+--------------+------+------+-----+
| conv/bias   | (2,)         | 2    | 1.0  | 0.0 |
| conv/kernel | (3, 3, 3, 2) | 54   | 1.0  | 0.0 |
+-------------+--------------+------+------+-----+
Total: 56"""

FLAX_CONV2D_MAPPING_PARAMETER_OVERVIEW_WITH_STATS = """+--------------------+--------------+------+------+-----+
| Name               | Shape        | Size | Mean | Std |
+--------------------+--------------+------+------+-----+
| params/conv/bias   | (2,)         | 2    | 1.0  | 0.0 |
| params/conv/kernel | (3, 3, 3, 2) | 54   | 1.0  | 0.0 |
+--------------------+--------------+------+------+-----+
Total: 56"""


class CNN(nn.Module):

  @nn.compact
  def __call__(self, x):
    return nn.Conv(features=2, kernel_size=(3, 3), name="conv")(x)


class JaxParameterOverviewTest(absltest.TestCase):

  def test_count_parameters_empty(self):
    self.assertEqual(0, parameter_overview.count_parameters({}))

  def test_count_parameters(self):
    rng = jax.random.PRNGKey(42)
    # Weights of a 2D convolution with 2 filters..
    variables = CNN().init(rng, jnp.zeros((2, 5, 5, 3)))
    # 3 * 3*3 * 2 + 2 (bias) = 56 parameters
    self.assertEqual(56,
                     parameter_overview.count_parameters(variables["params"]))

  def test_get_parameter_overview_empty(self):
    self.assertEqual(EMPTY_PARAMETER_OVERVIEW,
                     parameter_overview.get_parameter_overview({}))

  def test_get_parameter_overview(self):
    rng = jax.random.PRNGKey(42)
    # Weights of a 2D convolution with 2 filters..
    variables = CNN().init(rng, jnp.zeros((2, 5, 5, 3)))
    variables = jax.tree_map(jnp.ones_like, variables)
    self.assertEqual(
        FLAX_CONV2D_PARAMETER_OVERVIEW,
        parameter_overview.get_parameter_overview(
            variables["params"], include_stats=False))
    self.assertEqual(
        FLAX_CONV2D_PARAMETER_OVERVIEW_WITH_STATS,
        parameter_overview.get_parameter_overview(variables["params"]))
    self.assertEqual(
        FLAX_CONV2D_MAPPING_PARAMETER_OVERVIEW_WITH_STATS,
        parameter_overview.get_parameter_overview(variables))

  def test_printing_bool(self):
    self.assertEqual(
        parameter_overview._default_table_value_formatter(True), "True")
    self.assertEqual(
        parameter_overview._default_table_value_formatter(False), "False")


if __name__ == "__main__":
  absltest.main()
