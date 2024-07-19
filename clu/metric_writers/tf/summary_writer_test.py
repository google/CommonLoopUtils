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

"""Tests for SummaryWriter."""

import collections
import os

from clu.metric_writers.tf import summary_writer
import numpy as np
import tensorflow as tf

from tensorboard.plugins.hparams import plugin_data_pb2


def _load_summaries_data(logdir):
  """Loads raw summaries data from events in a logdir."""
  paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
  data = collections.defaultdict(dict)
  metadata = collections.defaultdict(dict)
  for path in paths:
    for event in tf.compat.v1.train.summary_iterator(path):
      for value in event.summary.value:
        data[event.step][value.tag] = tf.make_ndarray(value.tensor)
        if value.HasField("metadata"):
          metadata[event.step][value.tag] = value.metadata.SerializeToString()
  return data, metadata


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


def _load_hparams(logdir: str):
  """Loads hparams summaries from events in a logdir."""
  paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
  # data = collections.defaultdict(dict)
  hparams = []
  for path in paths:
    for event in tf.compat.v1.train.summary_iterator(path):
      for value in event.summary.value:
        if value.metadata.plugin_data.plugin_name == "hparams":
          hparams.append(plugin_data_pb2.HParamsPluginData.FromString(
              value.metadata.plugin_data.content))
  return hparams


class SummaryWriterTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.logdir = self.get_temp_dir()
    self.writer = summary_writer.SummaryWriter(self.logdir)

  def test_write_summaries(self):
    self.writer.write_summaries(
        11,
        {"a": np.eye(3, dtype=np.uint8),
         "b": np.eye(2, dtype=np.float32)},
        {"a": np.ones((2, 3)).tobytes()})
    self.writer.flush()
    data, metadata = _load_summaries_data(self.logdir)
    self.assertAllClose(
        data[11],
        {"a": np.eye(3, dtype=np.uint8), "b": np.eye(2, dtype=np.float32)})
    self.assertIn("a", metadata[11])

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
        }, num_buckets={"a": 2, "b": 2})
    self.writer.write_histograms(
        2, {
            "a": np.asarray([0.2, 0.4, 0.5, 0.1, -0.1]),
            "b": np.asarray([0.7, 0.3, 0.2, 0.1, 0.0]),
        }, num_buckets={"a": 2, "b": 2})
    self.writer.flush()
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

  def test_hparams(self):
    self.writer.write_hparams(dict(batch_size=512, num_epochs=90))
    hparams = _load_hparams(self.logdir)
    self.assertLen(hparams, 1)
    hparams_dict = hparams[0].session_start_info.hparams
    self.assertLen(hparams_dict, 2)
    self.assertEqual(512, hparams_dict["batch_size"].number_value)
    self.assertEqual(90, hparams_dict["num_epochs"].number_value)

  def test_hparams_nested(self):
    config = {
        "list": [1, 2],
        "tuple": (3, 4),
        "subconfig": {
            "value": "a",
            "list": [10, 20],
        },
    }
    self.writer.write_hparams(config)
    hparams = _load_hparams(self.logdir)
    self.assertLen(hparams, 1)
    hparams_dict = hparams[0].session_start_info.hparams
    self.assertLen(hparams_dict, 7)
    self.assertEqual(1, hparams_dict["list.0"].number_value)
    self.assertEqual(2, hparams_dict["list.1"].number_value)
    self.assertEqual(3, hparams_dict["tuple.0"].number_value)
    self.assertEqual(4, hparams_dict["tuple.1"].number_value)
    self.assertEqual("a", hparams_dict["subconfig.value"].string_value)
    self.assertEqual(10, hparams_dict["subconfig.list.0"].number_value)
    self.assertEqual(20, hparams_dict["subconfig.list.1"].number_value)

if __name__ == "__main__":
  tf.test.main()
