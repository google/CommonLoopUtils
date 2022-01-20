# Copyright 2022 The CLU Authors.
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

"""Defines available types for use by Metrics when written.

A Metric should return one of the following types when compute() is called.
"""

import abc
import dataclasses
from typing import Any, Union

import jax.numpy as jnp
import numpy as np

ArrayType = Union[np.ndarray, jnp.ndarray]
ScalarType = Union[int, float, np.number, np.ndarray, jnp.ndarray]


class Value(abc.ABC):
  """Class defining available metric computation return values.

  Types mirror those available in MetricWriter. See
  clu/metric_writers/interface.py
  """
  pass


@dataclasses.dataclass
class Summary(Value):
  value: ArrayType
  metadata: Any


@dataclasses.dataclass
class Scalar(Value):
  value: ScalarType


@dataclasses.dataclass
class Image(Value):
  """Image type.

  Mapping from image key to images. Images should have the shape [N, H, W, C] or
  [H, W, C], where H is the height, W is the width and C the
  number of channels (1 or 3). N is the number of images that will be
  written. Image dimensions can differ between different image keys but
  not between different steps for the same image key.
  """
  value: ArrayType


@dataclasses.dataclass
class Audio(Value):
  """Audio type.

  Mapping from audio key to audios. Audios should have the shape [N, T, C],
  where T is the time length and C the number of channels (1 -
  mono, 2 - stereo, >= 3 - surround; not all writers support any number of
  channels). N is the number of audios that will be written. Audio
  dimensions can differ between different audio keys but not between
  different steps for the same audio key. Values should be floating-point
  values in [-1, +1].
  """
  value: ArrayType
  sample_rate: int


@dataclasses.dataclass
class Text(Value):
  value: str


@dataclasses.dataclass
class Histogram(Value):
  # value must be an array of counts (integers)
  value: ArrayType
  num_buckets: int


@dataclasses.dataclass
class HyperParam(Value):
  """The name of the hyperparameter should be handled outside this class.

  Value should correspond to a single hyperparameter, while a Mapping[str,
  HyperParam] (name to HyperParam) is maintained independently.
  """
  value: Any
