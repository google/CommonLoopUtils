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

"""Defines a generic write interface.

The write helper accepts a MetricWriter object and a Mapping[str,
clu.metrics.Metric], and automatically writes to the appropriate typed write
method of the writer depending on the type of the metric.
"""

import collections
from typing import Mapping, Union

from clu import values
from clu.metric_writers import interface
import jax.numpy as jnp
import numpy as np


def _is_scalar(value):
  if isinstance(value, values.Scalar) or isinstance(value,
                                                    (int, float, np.number)):
    return True
  if isinstance(value, (np.ndarray, jnp.ndarray)):
    return jnp.ndim(value) == 0
  return False


def write_values(writer: interface.MetricWriter, step: int,
                 metrics: Mapping[str, Union[values.Value, values.ArrayType,
                                             values.ScalarType]]):
  """Writes all provided metrics.

  Allows providing a mapping of name to Value object, where each Value
  specifies a type. The appropriate write method can then be called depending
  on the type.

  Args:
    writer: MetricWriter object
    step: Step at which the arrays were generated.
    metrics: Mapping from name to clu.values.Value object.
  """
  writes = collections.defaultdict(dict)
  for k, v in metrics.items():
    if isinstance(v, values.Summary):
      writes[(writer.write_summaries, frozenset({"metadata": v.metadata
                                                }.items()))][k] = v.value
    elif _is_scalar(v):
      if isinstance(v, values.Scalar):
        writes[(writer.write_scalars, frozenset())][k] = v.value
      else:
        writes[(writer.write_scalars, frozenset())][k] = v
    elif isinstance(v, values.Image):
      writes[(writer.write_images, frozenset())][k] = v.value
    elif isinstance(v, values.Text):
      writes[(writer.write_texts, frozenset())][k] = v.value
    elif isinstance(v, values.HyperParam):
      writes[(writer.write_hparams, frozenset())][k] = v.value
    elif isinstance(v, values.Histogram):
      writes[(writer.write_histograms,
              frozenset({"num_buckets": v.num_buckets}.items()))][k] = v.value
    elif isinstance(v, values.Audio):
      writes[(writer.write_audios,
              frozenset({"sample_rate": v.sample_rate}.items()))][k] = v.value
    else:
      raise ValueError("Metric: ", k, " has unsupported value: ", v)

  for (fn, extra_args), vals in writes.items():
    fn(step, vals, **dict(extra_args))
