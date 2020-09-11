# Copyright 2020 The CLU Authors.
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

"""MetricWriter for writing to TF summary files.

Only works in eager mode.
"""

from typing import Any, Mapping, Optional

from clu.google import usage_logging
from clu.metric_writers import interface
import numpy as np
import tensorflow as tf

from tensorboard.plugins.hparams import api as hparams_api


Scalar = interface.Scalar


class SummaryWriter(interface.MetricWriter):
  """MetricWriter that writes TF summary files."""

  def __init__(self, logdir: str):
    super().__init__()
    self._summary_writer = tf.summary.create_file_writer(logdir)
    usage_logging.log_event("create", "metric_writers.SummaryWriter")

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    with self._summary_writer.as_default():
      for key, value in scalars.items():
        tf.summary.scalar(key, value, step=step)

  def write_images(self, step: int, images: Mapping[str, np.ndarray]):
    with self._summary_writer.as_default():
      for key, value in images.items():
        tf.summary.image(key, value, step=step)

  def write_texts(self, step: int, texts: Mapping[str, str]):
    with self._summary_writer.as_default():
      for key, value in texts.items():
        tf.summary.text(key, value, step=step)

  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, np.ndarray],
                       num_buckets: Optional[Mapping[str, int]] = None):
    with self._summary_writer.as_default():
      for key, value in arrays.items():
        tf.summary.histogram(key, value, step=step, buckets=num_buckets)

  def write_hparams(self, hparams: Mapping[str, Any]):
    with self._summary_writer.as_default():
      hparams_api.hparams(hparams)
