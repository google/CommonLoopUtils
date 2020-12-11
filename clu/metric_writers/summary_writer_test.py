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

"""Tests for SummaryWriter."""

import collections
import os

from clu.metric_writers import summary_writer
import numpy as np
import tensorflow as tf


def _load_histograms_data(logdir):
  """Loads tensor summaries from events in a logdir."""
  # Note: new versions of histograms don't use the HistogramProto type, but
  # they are written as tensors representing the bounds and counts of buckets,
  # with plugin_name = "histogram".
  paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
  data = {}
  for path in paths:
    for event in tf.compat.v1.train.summary_iterator(path):
      for value in event.summary.value:
        current_steps, current_tensors = data.get(value.tag, ([], []))
        data[value.tag] = (current_steps + [event.step],
                           current_tensors + [tf.make_ndarray(value.tensor)])
  return {
      tag: (np.stack(steps), np.stack(tensors))
      for tag, (steps, tensors) in data.items()
  }


def _load_scalars_data(logdir: str):
  """Loads scalar summaries from events in a logdir."""
  paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
  data = collections.defaultdict(dict)
  for path in paths:
    for event in tf.compat.v1.train.summary_iterator(path):
      for value in event.summary.value:
        data[event.step][value.tag] = tf.make_ndarray(value.tensor).flat[0]

  return data


class SummaryWriterTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.logdir = self.get_temp_dir()
    self.writer = summary_writer.SummaryWriter(self.logdir)

  def test_write_scalar(self):
    self.writer.write_scalars(11, {"a": 0.6, "b": 15})
    self.writer.write_scalars(20, {"a": 0.8, "b": 12})
    self.writer.flush()
    data = _load_scalars_data(self.logdir)
    self.assertAllClose(data[11], {"a": 0.6, "b": 15})
    self.assertAllClose(data[20], {"a": 0.8, "b": 12})

  def test_write_histograms(self):
    self.writer.write_histograms(
        0, {
            "a": np.asarray([0.3, 0.1, 0.5, 0.7, 0.1]),
            "b": np.asarray([-0.1, 0.3, 0.2, 0.4, 0.4]),
        }, num_buckets=2)
    self.writer.write_histograms(
        2, {
            "a": np.asarray([0.2, 0.4, 0.5, 0.1, -0.1]),
            "b": np.asarray([0.7, 0.3, 0.2, 0.1, 0.0]),
        }, num_buckets=2)
    data = _load_histograms_data(self.logdir)
    # In the histograms, each tuple represents
    # (bucket_min, bucket_max, bucket_count), where bucket_min is inclusive and
    # bucket_max is exclusive (except the last bucket_max which is inclusive).
    expected_histograms_a = [
        # Step 0.
        [(0.1, 0.4, 3), (0.4, 0.7, 2)],
        # Step 1.
        [(-0.1, 0.2, 2), (0.2, 0.5, 3)],
    ]
    self.assertAllClose(data["a"], ([0, 2], expected_histograms_a))
    expected_histograms_b = [
        # Step 0.
        [(-0.1, 0.15, 1), (0.15, 0.4, 4)],
        # Step 1.
        [(0.0, 0.35, 4), (0.35, 0.7, 1)],
    ]
    self.assertAllClose(data["b"], ([0, 2], expected_histograms_b))

if __name__ == "__main__":
  tf.test.main()
