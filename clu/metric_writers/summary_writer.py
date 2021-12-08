# Copyright 2021 The CLU Authors.
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

Only works in eager mode. Does not work for Pytorch code, please use
TorchTensorboardWriter instead.
"""

from typing import Any, Mapping, Optional


from clu.internal import utils
from clu.metric_writers import interface
import tensorflow as tf

from tensorboard.plugins.hparams import api as hparams_api


Array = interface.Array
Scalar = interface.Scalar


class SummaryWriter(interface.MetricWriter):
  """MetricWriter that writes TF summary files."""

  def __init__(self, logdir: str):
    super().__init__()
    self._summary_writer = tf.summary.create_file_writer(logdir)


  def write_summaries(
      self, step: int,
      values: Mapping[str, Array],
      metadata: Optional[Mapping[str, Any]] = None):
    with self._summary_writer.as_default():
      for key, value in values.items():
        md = metadata.get(key) if metadata is not None else None
        tf.summary.write(key, value, step=step, metadata=md)

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    with self._summary_writer.as_default():
      for key, value in scalars.items():
        tf.summary.scalar(key, value, step=step)

  def write_images(self, step: int, images: Mapping[str, Array]):
    with self._summary_writer.as_default():
      for key, value in images.items():
        if len(value.shape) == 3:
          value = value[None]
        tf.summary.image(key, value, step=step, max_outputs=value.shape[0])

  def write_audios(
      self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
    with self._summary_writer.as_default():
      for key, value in audios.items():
        tf.summary.audio(key, value, sample_rate=sample_rate, step=step,
                         max_outputs=value.shape[0])

  def write_texts(self, step: int, texts: Mapping[str, str]):
    with self._summary_writer.as_default():
      for key, value in texts.items():
        tf.summary.text(key, value, step=step)

  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, Array],
                       num_buckets: Optional[Mapping[str, int]] = None):
    with self._summary_writer.as_default():
      for key, value in arrays.items():
        buckets = None if num_buckets is None else num_buckets.get(key)
        tf.summary.histogram(key, value, step=step, buckets=buckets)

  def write_hparams(self, hparams: Mapping[str, Any]):
    with self._summary_writer.as_default():
      hparams_api.hparams(dict(utils.flatten_dict(hparams)))

  def flush(self):
    self._summary_writer.flush()

  def close(self):
    self._summary_writer.close()
