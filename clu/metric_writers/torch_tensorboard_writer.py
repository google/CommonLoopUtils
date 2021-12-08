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

"""MetricWriter for Pytorch summary files.

Use this writer for the Pytorch-based code.

"""

from typing import Any, Mapping, Optional
from absl import logging


from clu.metric_writers import interface
from torch.utils import tensorboard

Array = interface.Array
Scalar = interface.Scalar


class TorchTensorboardWriter(interface.MetricWriter):
  """MetricWriter that writes Pytorch summary files."""

  def __init__(self, logdir: str):
    super().__init__()
    self._writer = tensorboard.SummaryWriter(log_dir=logdir)


  def write_summaries(
      self, step: int,
      values: Mapping[str, Array],
      metadata: Optional[Mapping[str, Any]] = None):
    logging.log_first_n(
        logging.WARNING,
        "TorchTensorboardWriter does not support writing raw summaries.", 1)

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    for key, value in scalars.items():
      self._writer.add_scalar(key, value, global_step=step)

  def write_images(self, step: int, images: Mapping[str, Array]):
    for key, value in images.items():
      self._writer.add_image(key, value, global_step=step, dataformats="HWC")

  def write_audios(
      self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
    for key, value in audios.items():
      self._writer.add_audio(
          key, value, global_step=step, sample_rate=sample_rate)

  def write_texts(self, step: int, texts: Mapping[str, str]):
    for key, value in texts.items():
      self._writer.text(key, value, global_step=step)

  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, Array],
                       num_buckets: Optional[Mapping[str, int]] = None):
    for tag, values in arrays.items():
      bins = None if num_buckets is None else num_buckets.get(tag)
      self._writer.add_histogram(
          tag, values, global_step=step, bins="auto", max_bins=bins)

  def write_hparams(self, hparams: Mapping[str, Any]):
    self._writer.add_hparams(hparams, {})

  def flush(self):
    self._writer.flush()

  def close(self):
    self._writer.close()
