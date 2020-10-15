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

"""Tests for perodic actions."""

import time
from unittest import mock

from absl.testing import parameterized
from clu import periodic_actions
import tensorflow as tf


class ReportProgressTest(tf.test.TestCase, parameterized.TestCase):

  def test_every_steps(self):
    hook = periodic_actions.ReportProgress(
        every_steps=4, every_secs=None, num_train_steps=10)
    t = time.time()
    with self.assertLogs(level="INFO") as logs:
      self.assertFalse(hook(1, t))
      t += 0.11
      self.assertFalse(hook(2, t))
      t += 0.13
      self.assertFalse(hook(3, t))
      t += 0.12
      self.assertTrue(hook(4, t))
    # We did 1 step every 0.12s => 8.333 steps/s.
    self.assertEqual(logs.output, [
        "INFO:absl:Setting work unit notes: 40.0% @4, 8.3 steps/s, ETA: 0 min"
    ])

  def test_every_secs(self):
    hook = periodic_actions.ReportProgress(
        every_steps=None, every_secs=0.3, num_train_steps=10)
    t = time.time()
    with self.assertLogs(level="INFO") as logs:
      self.assertFalse(hook(1, t))
      t += 0.11
      self.assertFalse(hook(2, t))
      t += 0.13
      self.assertFalse(hook(3, t))
      t += 0.12
      self.assertTrue(hook(4, t))
    # We did 1 step every 0.12s => 8.333 steps/s.
    self.assertEqual(logs.output, [
        "INFO:absl:Setting work unit notes: 40.0% @4, 8.3 steps/s, ETA: 0 min"
    ])

  def test_called_every_step(self):
    hook = periodic_actions.ReportProgress(every_steps=3, num_train_steps=10)
    t = time.time()
    with self.assertRaisesRegex(ValueError,
                                "EveryNHook must be called after every step"):
      hook(1, t)
      hook(11, t)  # Raises exception.

  @parameterized.named_parameters(
      ("_nowait", False),
      ("_wait", True),
  )
  @mock.patch("time.time")
  def test_named(self, wait_jax_async_dispatch, time_mock):
    time_mock.return_value = 0
    hook = periodic_actions.ReportProgress(
        every_steps=1, every_secs=None, num_train_steps=10)
    def _wait():
      # Here we depend on hook._executor=ThreadPoolExecutor(max_workers=1)
      hook._executor.submit(lambda: None).result()
    self.assertFalse(hook(1))  # Never triggers on first execution.
    with hook.timed("test1", wait_jax_async_dispatch):
      _wait()
      time_mock.return_value = 1
    _wait()
    with hook.timed("test2", wait_jax_async_dispatch):
      _wait()
      time_mock.return_value = 2
    _wait()
    with hook.timed("test1", wait_jax_async_dispatch):
      _wait()
      time_mock.return_value = 3
    _wait()
    time_mock.return_value = 4
    with self.assertLogs(level="INFO") as logs:
      self.assertTrue(hook(2))
    self.assertEqual(logs.output, [
        "INFO:absl:Setting work unit notes: 20.0% @2, 0.2 steps/s, ETA: 1 min"
        " (0 min : 50.0% test1, 25.0% test2)"
    ])


class DummyProfilerSession:
  """Dummy Profiler that records the steps at which sessions started/ended."""

  def __init__(self):
    self.step = None
    self.start_session_call_steps = []
    self.end_session_call_steps = []

  def start_session(self):
    self.start_session_call_steps.append(self.step)

  def end_session_and_get_url(self, tag):
    del tag
    self.end_session_call_steps.append(self.step)


class ProfileTest(tf.test.TestCase):

  @mock.patch.object(periodic_actions, "profiler", autospec=True)
  def test_every_steps(self, mock_profiler):
    start_steps = []
    stop_steps = []
    step = 0

    def add_start_step(logdir):
      del logdir  # unused
      start_steps.append(step)

    def add_stop_step():
      stop_steps.append(step)

    mock_profiler.start.side_effect = add_start_step
    mock_profiler.stop.side_effect = add_stop_step
    hook = periodic_actions.Profile(
        num_profile_steps=2, first_profile=3, every_steps=7)
    for step in range(1, 18):
      hook(step)
    self.assertAllEqual([3, 7, 14], start_steps)
    self.assertAllEqual([5, 9, 16], stop_steps)


if __name__ == "__main__":
  tf.test.main()
