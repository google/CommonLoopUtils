# Copyright 2024 The CLU Authors.
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

"""Library for unify reporting model metrics across various logging formats.

This library provides a MetricWriter for each logging format (SummyWriter,
LoggingWriter, etc.) and composing MetricWriter to add support for asynchronous
logging or writing to multiple formats.
"""

import abc
from collections.abc import Mapping
from typing import Any, Optional, Union

import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]
Scalar = Union[int, float, np.number, np.ndarray, jnp.ndarray]


class MetricWriter(abc.ABC):
  """MetricWriter inferface."""

  @abc.abstractmethod
  def write_summaries(
      self, step: int,
      values: Mapping[str, Array],
      metadata: Optional[Mapping[str, Any]] = None):
    """Saves an arbitrary tensor summary.

    Useful when working with custom plugins or constructing a summary directly.

    Args:
      step: Step at which the scalar values occurred.
      values: Mapping from tensor keys to tensors.
      metadata: Optional SummaryMetadata, as a proto or serialized bytes.
                Note that markdown formatting is rendered by tensorboard.
    """

  @abc.abstractmethod
  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    """Write scalar values for the step.

    Consecutive calls to this method can provide different sets of scalars.
    Repeated writes for the same metric at the same step are not allowed.

    Args:
      step: Step at which the scalar values occurred.
      scalars: Mapping from metric name to value.
    """

  @abc.abstractmethod
  def write_images(self, step: int, images: Mapping[str, Array]):
    """Write images for the step.

    Consecutive calls to this method can provide different sets of images.
    Repeated writes for the same image key at the same step are not allowed.

    Warning: Not all MetricWriter implementation support writing images!

    Args:
      step: Step at which the images occurred.
      images: Mapping from image key to images. Images should have the shape [N,
        H, W, C] or [H, W, C], where H is the height, W is the width and C the
        number of channels (1 or 3). N is the number of images that will be
        written. Image dimensions can differ between different image keys but
        not between different steps for the same image key.
    """

  @abc.abstractmethod
  def write_videos(self, step: int, videos: Mapping[str, Array]):
    """Write videos for the step.

    Warning: Logging only.
    Not all MetricWriter implementation support writing videos!

    Consecutive calls to this method can provide different sets of videos.
    Repeated writes for the same video key at the same step are not allowed.


    Args:
      step: Step at which the videos occurred.
      videos: Mapping from video key to videos. videos should have the shape
        [N, T, H, W, C] or [T, H, W, C], where T is time, H is the height,
        W is the width and C the number of channels (1 or 3). N is the number
        of videos that will be written. Video dimensions can differ between
        different video keys but not between different steps for the same
        video key.
    """

  @abc.abstractmethod
  def write_audios(
      self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
    """Write audios for the step.

    Consecutive calls to this method can provide different sets of audios.
    Repeated writes for the same audio key at the same step are not allowed.

    Warning: Not all MetricWriter implementation support writing audios!

    Args:
      step: Step at which the audios occurred.
      audios: Mapping from audio key to audios. Audios should have the shape
        [N, T, C], where T is the time length and C the number of channels
        (1 - mono, 2 - stereo, >= 3 - surround; not all writers support any
        number of channels). N is the number of audios that will be written.
        Audio dimensions can differ between different audio keys but not between
        different steps for the same audio key. Values should be floating-point
        values in [-1, +1].
      sample_rate: Sample rate for the audios.
    """

  @abc.abstractmethod
  def write_texts(self, step: int, texts: Mapping[str, str]):
    """Writes text snippets for the step.

    Warning: Not all MetricWriter implementation support writing text!

    Args:
      step: Step at which the text snippets occurred.
      texts: Mapping from name to text snippet.
    """

  @abc.abstractmethod
  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, Array],
                       num_buckets: Optional[Mapping[str, int]] = None):
    """Writes histograms for the step.

    Consecutive calls to this method can provide different sets of scalars.
    Repeated writes for the same metric at the same step are not allowed.

    Warning: Not all MetricWriter implementation support writing histograms!

    Args:
      step: Step at which the arrays were generated.
      arrays: Mapping from name to arrays to summarize.
      num_buckets: Number of buckets used to create the histogram of the arrays.
        The default number of buckets depends on the particular implementation
        of the MetricWriter.
    """

  @abc.abstractmethod
  def write_pointcloud(
      self,
      step: int,
      point_clouds: Mapping[str, Array],
      *,
      point_colors: Optional[Mapping[str, Array]] = None,
      configs: Optional[
          Mapping[str, Union[str, int, float, bool, None]]
      ] = None,
  ):
    """Writes point cloud summaries.

    Args:
      step: Step at which the point cloud was generated.
      point_clouds: Mapping from point clouds key to point cloud of shape [N, 3]
        array of point coordinates.
      point_colors: Mapping from point colors key to [N, 3] array of point
        colors.
      configs: A dictionary of configuration options for the point cloud.
    """

  @abc.abstractmethod
  def write_hparams(self, hparams: Mapping[str, Any]):
    """Write hyper parameters.

    Do not call twice.

    Args:
      hparams: Flat mapping from hyper parameter name to value.
    """

  @abc.abstractmethod
  def flush(self):
    """Tells the MetricWriter to write out any cached values."""

  @abc.abstractmethod
  def close(self):
    """Flushes and closes the MetricWriter.

    Calling any method on MetricWriter after MetricWriter.close()
    is undefined behavior.
    """
