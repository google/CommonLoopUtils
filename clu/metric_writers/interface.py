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

"""Library for unify reporting model metrics across various logging formats.

This library provides a MetricWriter for each logging format (SummyWriter,
LoggingWriter, etc.) and composing MetricWriter to add support for asynchronous
logging or writing to multiple formats.
"""

import abc
from typing import Any, Mapping, Optional, Union

import numpy as np

Scalar = Union[int, float]


class MetricWriter(abc.ABC):
  """MetricWriter inferface."""

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
  def write_images(self, step: int, images: Mapping[str, np.ndarray]):
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
                       arrays: Mapping[str, np.ndarray],
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
