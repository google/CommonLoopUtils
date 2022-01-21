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

"""Tests for interface."""
import itertools
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from clu import values
from clu.metric_writers import utils
from clu.metric_writers.async_writer import AsyncMultiWriter
from clu.metric_writers.async_writer import AsyncWriter
from clu.metric_writers.interface import MetricWriter
from clu.metric_writers.logging_writer import LoggingWriter
from clu.metric_writers.multi_writer import MultiWriter
from clu.metric_writers.summary_writer import SummaryWriter
import clu.metrics
import flax.struct
import jax.numpy as jnp
import tensorflow as tf



@flax.struct.dataclass
class HistogramMetric(clu.metrics.Metric):
  value: jnp.ndarray
  num_buckets: int

  def compute_value(self):
    return values.Histogram(self.value, self.num_buckets)


@flax.struct.dataclass
class ImageMetric(clu.metrics.Metric):
  value: jnp.ndarray

  def compute_value(self):
    return values.Image(self.value)


@flax.struct.dataclass
class AudioMetric(clu.metrics.Metric):
  value: jnp.ndarray
  sample_rate: int

  def compute_value(self):
    return values.Audio(self.value, self.sample_rate)


@flax.struct.dataclass
class TextMetric(clu.metrics.Metric):
  value: str

  def compute_value(self):
    return values.Text(self.value)


@flax.struct.dataclass
class HyperParamMetric(clu.metrics.Metric):
  value: float

  def compute_value(self):
    return values.HyperParam(self.value)


@flax.struct.dataclass
class SummaryMetric(clu.metrics.Metric):
  value: jnp.ndarray
  metadata: Any

  def compute_value(self):
    return values.Summary(self.value, self.metadata)


def _to_summary(metrics):
  return {k: v.value for k, v in metrics.items()}


def _to_list_of_dicts(d):
  return [{k: v} for k, v in d.items()]


class ONEOF(object):
  """ONEOF(options_list) check value in options_list."""

  def __init__(self, container):
    if not hasattr(container, "__contains__"):
      raise TypeError(f"{container!r} is not a container")
    if not container:
      raise ValueError(f"{container!r} is empty")
    self._c = container

  def __eq__(self, o):
    return o in self._c

  def __ne__(self, o):
    return o not in self._c

  def __repr__(self):
    return "<ONEOF({})>".format(",".join(repr(i) for i in self._c))


class MetricWriterTest(tf.test.TestCase, parameterized.TestCase):

  def test_write(self):
    writer = mock.Mock(spec_set=MetricWriter)
    step = 3
    num_buckets = 4
    sample_rate = 10
    scalar_metrics = {
        "loss": clu.metrics.Average.from_model_output(jnp.asarray([1, 2, 3])),
        "accuracy": clu.metrics.LastValue.from_model_output(jnp.asarray([5])),
    }
    image_metrics = {
        "image": ImageMetric(jnp.asarray([[4, 5], [1, 2]])),
    }
    histogram_metrics = {
        "hist":
            HistogramMetric(value=jnp.asarray([7, 8]), num_buckets=num_buckets),
        "hist2":
            HistogramMetric(
                value=jnp.asarray([9, 10]), num_buckets=num_buckets),
    }
    audio_metrics = {
        "audio":
            AudioMetric(value=jnp.asarray([1, 5]), sample_rate=sample_rate),
        "audio2":
            AudioMetric(value=jnp.asarray([1, 5]), sample_rate=sample_rate + 2),
    }
    text_metrics = {
        "text": TextMetric(value="hello"),
    }
    hparam_metrics = {
        "lr": HyperParamMetric(value=0.01),
    }
    summary_metrics = {
        "summary":
            SummaryMetric(value=jnp.asarray([2, 3, 10]), metadata="some info"),
        "summary2":
            SummaryMetric(value=jnp.asarray([2, 3, 10]), metadata=5),
    }
    metrics = {
        **scalar_metrics,
        **image_metrics,
        **histogram_metrics,
        **audio_metrics,
        **text_metrics,
        **hparam_metrics,
        **summary_metrics,
    }
    metrics = {k: m.compute_value() for k, m in metrics.items()}
    utils.write_values(writer, step, metrics)

    writer.write_scalars.assert_called_once_with(
        step, {k: m.compute() for k, m in scalar_metrics.items()})
    writer.write_images.assert_called_once_with(step,
                                                _to_summary(image_metrics))
    writer.write_histograms.assert_called_once_with(
        step, _to_summary(histogram_metrics), num_buckets=num_buckets)
    writer.write_audios.assert_called_with(
        step,
        ONEOF(_to_list_of_dicts(_to_summary(audio_metrics))),
        sample_rate=ONEOF([sample_rate, sample_rate + 2]))
    writer.write_texts.assert_called_once_with(step, _to_summary(text_metrics))
    writer.write_hparams.assert_called_once_with(step,
                                                 _to_summary(hparam_metrics))
    writer.write_summaries.assert_called_with(
        step,
        ONEOF(_to_list_of_dicts(_to_summary(summary_metrics))),
        metadata=ONEOF(["some info", 5]))


  def test_create_default_writer_summary_writer_is_added(self):
    writer = utils.create_default_writer(
        logdir=self.get_temp_dir(), asynchronous=False)
    self.assertTrue(any(isinstance(w, SummaryWriter) for w in writer._writers))
    writer = utils.create_default_writer(logdir=None, asynchronous=False)
    self.assertFalse(any(isinstance(w, SummaryWriter) for w in writer._writers))


if __name__ == "__main__":
  absltest.main()
