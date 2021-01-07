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

"""Tests for AsyncWriter."""

import time
from unittest import mock

from clu.metric_writers import async_writer
from clu.metric_writers import interface
import numpy as np
import tensorflow as tf


class AsyncWriterTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.sync_writer = mock.create_autospec(interface.MetricWriter)
    self.writer = async_writer.AsyncWriter(self.sync_writer)

  def test_write_scalars_async(self):
    self.writer.write_scalars(0, {"a": 3, "b": 0.15})
    self.writer.write_scalars(2, {"a": 5, "b": 0.007})
    self.writer.flush()
    self.sync_writer.write_scalars.assert_has_calls([
        mock.call(step=0, scalars={
            "a": 3,
            "b": 0.15
        }),
        mock.call(step=2, scalars={
            "a": 5,
            "b": 0.007
        })
    ])

  def test_write_scalars_fails_without_flush(self):
    self.writer.write_scalars(0, {"a": 3, "b": 0.15})
    self.writer.write_scalars(2, {"a": 5, "b": 0.007})
    # No call to flush() makes it very unlikely that the background thread had
    # the sync_writer.
    self.sync_writer.write_scalars.assert_not_called()
    self.sync_writer.flush.assert_not_called()

  def test_write_images(self):
    images = np.zeros((2, 28, 28, 3))
    self.writer.write_images(4, {"input_images": images})
    self.writer.flush()
    self.sync_writer.write_images.assert_called_with(4,
                                                     {"input_images": mock.ANY})

  def test_write_texts(self):
    self.writer.write_texts(4, {"samples": "bla"})
    self.writer.flush()
    self.sync_writer.write_texts.assert_called_with(4, {"samples": "bla"})

  def test_ensure_flushes(self):
    with async_writer.ensure_flushes(self.writer) as writer:
      writer.write_scalars(0, {"a": 3, "b": 0.15})
      writer.write_scalars(2, {"a": 5, "b": 0.007})
    self.sync_writer.write_scalars.assert_has_calls([
        mock.call(step=0, scalars={
            "a": 3,
            "b": 0.15
        }),
        mock.call(step=2, scalars={
            "a": 5,
            "b": 0.007
        })
    ])

  def test_flush_before_close(self):
    self.writer.close()
    self.sync_writer.flush.assert_called()
    self.sync_writer.close.assert_called()

  def test_reraises_exception(self):
    self.sync_writer.write_scalars.side_effect = ValueError("foo")
    self.writer.write_scalars(0, {"a": 3, "b": 0.15})
    time.sleep(0.1)
    with self.assertRaisesRegex(ValueError, "foo"):
      self.writer.write_scalars(2, {"a": 5, "b": 0.007})


if __name__ == "__main__":
  tf.test.main()
