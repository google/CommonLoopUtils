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

"""Defines a generic write interface.

The write helper accepts a MetricWriter object and a Mapping[str,
clu.metrics.Metric], and automatically writes to the appropriate typed write
method of the writer depending on the type of the metric.
"""

# pylint: disable=g-importing-member

import collections
import getpass
import os
import re
from typing import Any, List, Mapping, Optional, Tuple, Union

from absl import flags
from absl import logging
from clu import values
from clu.metric_writers.async_writer import AsyncMultiWriter
from clu.metric_writers.interface import MetricWriter
from clu.metric_writers.logging_writer import LoggingWriter
from clu.metric_writers.multi_writer import MultiWriter
from clu.metric_writers.summary_writer import SummaryWriter
from etils import epath
import jax.numpy as jnp
import numpy as np


FLAGS = flags.FLAGS


def _is_scalar(value: Any) -> bool:
  if isinstance(value, values.Scalar) or isinstance(
      value, (int, float, np.number)
  ):
    return True
  if isinstance(value, (np.ndarray, jnp.ndarray)):
    return value.ndim == 0 or value.size <= 1
  return False


def write_values(
    writer: MetricWriter,
    step: int,
    metrics: Mapping[
        str, Union[values.Value, values.ArrayType, values.ScalarType]
    ],
):
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
  histogram_num_buckets = collections.defaultdict(int)
  for k, v in metrics.items():
    if isinstance(v, values.Summary):
      writes[
          (writer.write_summaries, frozenset({"metadata": v.metadata}.items()))
      ][k] = v.value
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
      writes[(writer.write_histograms, frozenset())][k] = v.value
      histogram_num_buckets[k] = v.num_buckets
    elif isinstance(v, values.Audio):
      writes[(
          writer.write_audios,
          frozenset({"sample_rate": v.sample_rate}.items()),
      )][k] = v.value
    else:
      raise ValueError("Metric: ", k, " has unsupported value: ", v)

  for (fn, extra_args), vals in writes.items():
    if fn == writer.write_histograms:
      # for write_histograms, the num_buckets arg is a Dict indexed by name
      writer.write_histograms(step, vals, num_buckets=histogram_num_buckets)
    else:
      fn(step, vals, **dict(extra_args))




def create_default_writer(
    logdir: Optional[epath.PathLike] = None,
    *,
    just_logging: bool = False,
    asynchronous: bool = True,
    collection: Optional[str] = None,
) -> MultiWriter:
  """Create the default writer for the platform.

  On most platforms this will create a MultiWriter that writes to multiple back
  ends (logging, TF summaries etc.).

  Args:
    logdir: Logging dir to use for TF summary files. If empty/None will the
      returned writer will not write TF summary files.
    just_logging: If True only use a LoggingWriter. This is useful in multi-host
      setups when only the first host should write metrics and all other hosts
      should only write to their own logs.
      default (None) will automatically determine if you # GOOGLE-INTERNAL have
    asynchronous: If True return an AsyncMultiWriter to not block when writing
      metrics.
    collection: A string which, if provided, provides an indication that the
      provided metrics should all be written to the same collection, or
      grouping.

  Returns:
    A `MetricWriter` according to the platform and arguments.
  """
  if just_logging:
    if asynchronous:
      return AsyncMultiWriter([LoggingWriter(collection=collection)])
    else:
      return MultiWriter([LoggingWriter(collection=collection)])
  writers = [LoggingWriter(collection=collection)]
  if logdir is not None:
    logdir = epath.Path(logdir)
    if collection is not None:
      logdir /= collection
    writers.append(SummaryWriter(os.fspath(logdir)))
  if asynchronous:
    return AsyncMultiWriter(writers)
  return MultiWriter(writers)
