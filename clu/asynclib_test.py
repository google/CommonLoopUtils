# Copyright 2023 The CLU Authors.
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

"""Tests for clu.asynclib."""

from unittest import mock

from clu import asynclib
import tensorflow as tf


class AsyncWriterTest(tf.test.TestCase):

  def test_async_execution(self):
    pool = asynclib.Pool()
    counter = 0

    @pool
    def fn(counter_increment, return_value):
      nonlocal counter
      counter += counter_increment
      return return_value

    future = fn(1, return_value=2)
    self.assertEqual(counter, 1)
    self.assertEqual(future.result(), 2)

  def test_reraise(self):
    pool = asynclib.Pool()

    @pool
    def error():
      raise ValueError("test")

    error()
    self.assertTrue(pool.has_errors)
    with self.assertRaisesRegex(asynclib.AsyncError, "test"):
      pool.join()
    self.assertFalse(pool.has_errors)

    @pool
    def noop():
      ...

    error()
    self.assertTrue(pool.has_errors)
    with self.assertRaisesRegex(asynclib.AsyncError, "test"):
      noop()
    self.assertFalse(pool.has_errors)

    pool.join()

  @mock.patch("concurrent.futures.ThreadPoolExecutor")
  def test_queue_length(self, executor_mock):
    pool_mock = mock.Mock()
    in_flight = []

    def execute_one():
      in_flight.pop(0)()

    def submit(fn, *args, **kwargs):
      in_flight.append(lambda: fn(*args, **kwargs))

    pool_mock.submit = submit
    executor_mock.return_value = pool_mock

    pool = asynclib.Pool()

    @pool
    def noop():
      ...

    self.assertEqual(pool.queue_length, 0)
    noop()
    self.assertEqual(pool.queue_length, 1)
    noop()
    self.assertEqual(pool.queue_length, 2)
    execute_one()
    self.assertEqual(pool.queue_length, 1)
    execute_one()
    self.assertEqual(pool.queue_length, 0)

  @mock.patch("concurrent.futures.ThreadPoolExecutor")
  def test_flush(self, executor_mock):
    pool_mock = mock.Mock()
    pool_mock._in_flight = None

    def execute_one():
      pool_mock._in_flight.pop(0)()

    def submit(fn, *args, **kwargs):
      pool_mock._in_flight.append(lambda: fn(*args, **kwargs))

    def create_pool(max_workers, thread_name_prefix):
      del max_workers
      del thread_name_prefix
      pool_mock._in_flight = []
      return pool_mock

    def shutdown(wait=False):
      if wait:
        while pool_mock._in_flight:
          execute_one()
      pool_mock._in_flight = None

    pool_mock.submit = submit
    executor_mock.side_effect = create_pool
    pool_mock.shutdown.side_effect = shutdown

    pool = asynclib.Pool()

    @pool
    def noop():
      ...

    self.assertEqual(pool.queue_length, 0)
    noop()
    self.assertEqual(pool.queue_length, 1)
    noop()
    pool.join()
    self.assertEqual(pool.queue_length, 0)
    noop()
    self.assertEqual(pool.queue_length, 1)


if __name__ == "__main__":
  tf.test.main()
