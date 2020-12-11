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

"""Tests for TorchTensorboardWriter."""

import collections
import os

from clu.metric_writers import torch_tensorboard_writer
import tensorflow as tf


def _load_scalars_data(logdir: str):
  """Loads scalar summaries from events in a logdir."""
  paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
  data = collections.defaultdict(dict)
  for path in paths:
    for event in tf.compat.v1.train.summary_iterator(path):
      for value in event.summary.value:
        data[event.step][value.tag] = value.simple_value

  return data


class TorchTensorboardWriterTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.logdir = self.get_temp_dir()
    self.writer = torch_tensorboard_writer.TorchTensorboardWriter(self.logdir)

  def test_write_scalar(self):
    self.writer.write_scalars(11, {"a": 0.6, "b": 15})
    self.writer.write_scalars(20, {"a": 0.8, "b": 12})
    self.writer.flush()
    data = _load_scalars_data(self.logdir)
    self.assertAllClose(data[11], {"a": 0.6, "b": 15})
    self.assertAllClose(data[20], {"a": 0.8, "b": 12})


if __name__ == "__main__":
  tf.test.main()
