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

"""Tests for the LoggingWriter."""

from clu.metric_writers import logging_writer
import numpy as np
import tensorflow as tf


class LoggingWriterTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.writer = logging_writer.LoggingWriter()

  def test_write_scalars(self):
    with self.assertLogs(level="INFO") as logs:
      self.writer.write_scalars(0, {"a": 3, "b": 0.15})
      self.writer.write_scalars(2, {"a": 0.0000005, "b": 0.007})
    self.assertEqual(
        logs.output,
        ["INFO:absl:[0] a=3, b=0.15", "INFO:absl:[2] a=5e-07, b=0.007"])

  def test_write_images(self):
    images = np.zeros((2, 28, 28, 3))
    with self.assertLogs(level="INFO") as logs:
      self.writer.write_images(4, {"input_images": images})
    self.assertEqual(
        logs.output,
        ["INFO:absl:[4] Got images: {'input_images': (2, 28, 28, 3)}."])

  def test_write_videos(self):
    videos = np.zeros((2, 4, 28, 28, 3))
    with self.assertLogs(level="INFO") as logs:
      self.writer.write_videos(4, {"input_videos": videos})
    self.assertEqual(
        logs.output,
        ["INFO:absl:[4] Got videos: {'input_videos': (2, 4, 28, 28, 3)}."])

  def test_write_texts(self):
    with self.assertLogs(level="INFO") as logs:
      self.writer.write_texts(4, {"samples": "bla"})
    self.assertEqual(
        logs.output,
        ["INFO:absl:[4] Got texts: {'samples': 'bla'}."])

  def test_write_histogram(self):
    with self.assertLogs(level="INFO") as logs:
      self.writer.write_histograms(
          step=4,
          arrays={
              "a": np.asarray([-0.1, 0.1, 0.3]),
              "b": np.arange(31),
              "c": np.asarray([0.1, 0.1, 0.1, 0.1, 0.1]),
          },
          num_buckets={
              "a": 2,
              "c": 1
          })
    # Note: There are 31 distinct values [0, 1, ..., 30], and 30 buckets by
    # default. Last bucket gets 2 values.
    expected_histo_b = ", ".join([f"[{i}, {i + 1}): 1" for i in range(29)] +
                                 ["[29, 30]: 2"])
    self.assertEqual(logs.output, [
        "INFO:absl:[4] Histogram for 'a' = {[-0.1, 0.1): 1, [0.1, 0.3]: 2}",
        f"INFO:absl:[4] Histogram for 'b' = {{{expected_histo_b}}}",
        "WARNING:absl:num_buckets was automatically changed from 1 to 2",
        "INFO:absl:[4] Histogram for 'c' = {[-0.4, 0.6]: 5}",
    ])

  def test_write_pointcloud(self):
    point_clouds = np.random.normal(0, 1, (1, 1024, 3)).astype(np.float32)
    point_colors = np.random.uniform(0, 1, (1, 1024, 3)).astype(np.float32)
    config = {
        "material": "PointCloudMaterial",
        "size": 0.09,
    }
    with self.assertLogs(level="INFO") as logs:
      self.writer.write_pointcloud(
          step=4,
          point_clouds={"pcd": point_clouds},
          point_colors={"pcd": point_colors},
          configs={"configs": config},
      )
    self.assertEqual(
        logs.output,
        [
            "INFO:absl:[4] Got point clouds: {'pcd': (1, 1024, 3)},"
            " point_colors: {'pcd': (1, 1024, 3)}, configs: {'configs':"
            " {'material': 'PointCloudMaterial', 'size': 0.09}}."
        ],
    )

  def test_write_hparams(self):
    with self.assertLogs(level="INFO") as logs:
      self.writer.write_hparams({"learning_rate": 0.1, "batch_size": 128})
    self.assertEqual(
        logs.output,
        [
            "INFO:absl:[Hyperparameters] {'learning_rate': 0.1, 'batch_size':"
            " 128}"
        ],
    )

  def test_collection(self):
    writer = logging_writer.LoggingWriter(collection="train")
    writer = logging_writer.LoggingWriter(collection="train")
    with self.assertLogs(level="INFO") as logs:
      writer.write_scalars(0, {"a": 3, "b": 0.15})
      writer.write_images(4, {"input_images": np.zeros((2, 28, 28, 3))})
      writer.write_texts(4, {"samples": "bla"})
      writer.write_histograms(
          step=4,
          arrays={
              "a": np.asarray([-0.1, 0.1, 0.3]),
          },
          num_buckets={
              "a": 2,
          })
      writer.write_hparams({"learning_rate": 0.1})

    self.assertEqual(logs.output, [
        "INFO:absl:[0] collection=train a=3, b=0.15",
        "INFO:absl:[4] collection=train Got images: {'input_images': (2, 28, 28, 3)}.",
        "INFO:absl:[4] collection=train Got texts: {'samples': 'bla'}.",
        "INFO:absl:[4] collection=train Histogram for 'a' = {[-0.1, 0.1): 1, [0.1, 0.3]: 2}",
        "INFO:absl:[Hyperparameters] collection=train {'learning_rate': 0.1}",
    ])


if __name__ == "__main__":
  tf.test.main()
