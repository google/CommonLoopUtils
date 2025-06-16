# Copyright 2025 The CLU Authors.
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

"""MetricWriter that writes to multiple MetricWriters."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from clu.metric_writers import interface

Array = interface.Array
Scalar = interface.Scalar


class MultiWriter(interface.MetricWriter):
  """MetricWriter that writes to multiple writers at once."""

  def __init__(self, writers: Sequence[interface.MetricWriter]):
    self._writers = tuple(writers)

  def write_summaries(
      self, step: int,
      values: Mapping[str, Array],
      metadata: Optional[Mapping[str, Any]] = None):
    for w in self._writers:
      w.write_summaries(step, values, metadata)

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    for w in self._writers:
      w.write_scalars(step, scalars)

  def write_images(self, step: int, images: Mapping[str, Array]):
    for w in self._writers:
      w.write_images(step, images)

  def write_videos(self, step: int, videos: Mapping[str, Array]):
    for w in self._writers:
      w.write_videos(step, videos)

  def write_audios(
      self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
    for w in self._writers:
      w.write_audios(step, audios, sample_rate=sample_rate)

  def write_texts(self, step: int, texts: Mapping[str, str]):
    for w in self._writers:
      w.write_texts(step, texts)

  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, Array],
                       num_buckets: Optional[Mapping[str, int]] = None):
    for w in self._writers:
      w.write_histograms(step, arrays, num_buckets)

  def write_pointcloud(
      self,
      step: int,
      point_clouds: Mapping[str, Array],
      *,
      point_colors: Mapping[str, Array] | None = None,
      configs: Mapping[str, str | float | bool | None] | None = None,
  ):
    for w in self._writers:
      w.write_pointcloud(
          step, point_clouds, point_colors=point_colors, configs=configs
      )

  def write_hparams(self, hparams: Mapping[str, Any]):
    for w in self._writers:
      w.write_hparams(hparams)

  def flush(self):
    for w in self._writers:
      w.flush()

  def close(self):
    for w in self._writers:
      w.close()
