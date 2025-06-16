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

"""Tests for MultiWriter."""

from unittest import mock

from clu.metric_writers import interface
from clu.metric_writers import multi_writer
import numpy as np
import tensorflow as tf


class MultiWriterTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.writers = [
        mock.create_autospec(interface.MetricWriter),
        mock.create_autospec(interface.MetricWriter)
    ]
    self.writer = multi_writer.MultiWriter(self.writers)

  def test_write_scalars(self):
    self.writer.write_scalars(0, {"a": 3, "b": 0.15})
    self.writer.write_scalars(2, {"a": 5, "b": 0.007})
    self.writer.flush()
    for w in self.writers:
      w.write_scalars.assert_has_calls([
          mock.call(step=0, scalars={
              "a": 3,
              "b": 0.15
          }),
          mock.call(step=2, scalars={
              "a": 5,
              "b": 0.007
          })
      ])
      w.flush.assert_called()

  def test_write_pointcloud(self):
    point_clouds = np.random.normal(0, 1, (1, 1024, 3)).astype(np.float32)
    point_colors = np.random.uniform(0, 1, (1, 1024, 3)).astype(np.float32)
    config = {
        "material": "PointCloudMaterial",
        "size": 0.09,
    }
    self.writer.write_pointcloud(
        step=0,
        point_clouds={"pcd": point_clouds},
        point_colors={"pcd": point_colors},
        configs={"config": config},
    )
    self.writer.flush()
    for w in self.writers:
      w.write_pointcloud.assert_called_with(
          step=0,
          point_clouds={"pcd": point_clouds},
          point_colors={"pcd": point_colors},
          configs={"config": config},
      )
      w.flush.assert_called()


if __name__ == "__main__":
  tf.test.main()
