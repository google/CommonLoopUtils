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

"""Tests for perodic actions."""

import tempfile
import time
from unittest import mock

from absl.testing import parameterized
from clu import periodic_actions
import tensorflow as tf


class ReportProgressTest(tf.test.TestCase, parameterized.TestCase):

  def test_every_steps(self):
    hook = periodic_actions.ReportProgress(
        every_steps=4, every_secs=None, num_train_steps=10)
    t = time.monotonic()
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
        "INFO:absl:Setting work unit notes: 8.3 steps/s, 40.0% (4/10), ETA: 0m"
    ])

  def test_every_secs(self):
    hook = periodic_actions.ReportProgress(
        every_steps=None, every_secs=0.3, num_train_steps=10)
    t = time.monotonic()
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
        "INFO:absl:Setting work unit notes: 8.3 steps/s, 40.0% (4/10), ETA: 0m"
    ])

  def test_without_num_train_steps(self):
    report = periodic_actions.ReportProgress(every_steps=2)
    t = time.monotonic()
    with self.assertLogs(level="INFO") as logs:
      self.assertFalse(report(1, t))
      self.assertTrue(report(2, t + 0.12))
    # We did 1 step in 0.12s => 8.333 steps/s.
    self.assertEqual(logs.output, [
        "INFO:absl:Setting work unit notes: 8.3 steps/s"
    ])

  def test_unknown_cardinality(self):
    report = periodic_actions.ReportProgress(
        every_steps=2,
        num_train_steps=tf.data.UNKNOWN_CARDINALITY)
    t = time.monotonic()
    with self.assertLogs(level="INFO") as logs:
      self.assertFalse(report(1, t))
      self.assertTrue(report(2, t + 0.12))
    # We did 1 step in 0.12s => 8.333 steps/s.
    self.assertEqual(logs.output, [
        "INFO:absl:Setting work unit notes: 8.3 steps/s"
    ])

  def test_called_every_step(self):
    hook = periodic_actions.ReportProgress(every_steps=3, num_train_steps=10)
    t = time.monotonic()
    with self.assertRaisesRegex(
        ValueError, "PeriodicAction must be called after every step"):
      hook(1, t)
      hook(11, t)  # Raises exception.

  @parameterized.named_parameters(
      ("_nowait", False),
      ("_wait", True),
  )
  @mock.patch("time.monotonic")
  def test_named(self, wait_jax_async_dispatch, mock_time):
    mock_time.return_value = 0
    hook = periodic_actions.ReportProgress(
        every_steps=1, every_secs=None, num_train_steps=10)
    def _wait():
      # Here we depend on hook._executor=ThreadPoolExecutor(max_workers=1)
      hook._executor.submit(lambda: None).result()
    self.assertFalse(hook(1))  # Never triggers on first execution.
    with hook.timed("test1", wait_jax_async_dispatch):
      _wait()
      mock_time.return_value = 1
    _wait()
    with hook.timed("test2", wait_jax_async_dispatch):
      _wait()
      mock_time.return_value = 2
    _wait()
    with hook.timed("test1", wait_jax_async_dispatch):
      _wait()
      mock_time.return_value = 3
    _wait()
    mock_time.return_value = 4
    with self.assertLogs(level="INFO") as logs:
      self.assertTrue(hook(2))
    self.assertEqual(logs.output, [
        "INFO:absl:Setting work unit notes: 0.2 steps/s, 20.0% (2/10), ETA: 0m"
        " (0m : 50.0% test1, 25.0% test2)"
    ])

  @mock.patch("time.monotonic")
  def test_write_metrics(self, time_mock):
    time_mock.return_value = 0
    writer_mock = mock.Mock()
    hook = periodic_actions.ReportProgress(
        every_steps=2, every_secs=None, writer=writer_mock)
    time_mock.return_value = 1
    hook(1)
    time_mock.return_value = 2
    hook(2)
    self.assertEqual(writer_mock.write_scalars.mock_calls, [
        mock.call(2, {"steps_per_sec": 1}),
        mock.call(2, {"uptime": 2}),
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
  @mock.patch("time.monotonic")
  def test_every_steps(self, mock_time, mock_profiler):
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
        logdir=tempfile.mkdtemp(),
        num_profile_steps=2,
        profile_duration_ms=2_000,
        first_profile=3,
        every_steps=7)
    for step in range(1, 18):
      mock_time.return_value = step - 0.5 if step == 9 else step
      hook(step)
    self.assertAllEqual([3, 7, 14], start_steps)
    # Note: profiling 7..10 instead of 7..9 because 7..9 took only 1.5 seconds.
    self.assertAllEqual([5, 10, 16], stop_steps)


class ProfileAllHostsTest(tf.test.TestCase):

  @mock.patch.object(periodic_actions, "profiler", autospec=True)
  def test_every_steps(self, mock_profiler):
    start_steps = []
    step = 0

    def profile_collect(logdir, callback, hosts, duration_ms):
      del logdir, callback, hosts, duration_ms  # unused
      start_steps.append(step)

    mock_profiler.collect.side_effect = profile_collect
    hook = periodic_actions.ProfileAllHosts(
        logdir=tempfile.mkdtemp(),
        profile_duration_ms=2_000,
        first_profile=3,
        every_steps=7)
    for step in range(1, 18):
      hook(step)
    self.assertAllEqual([3, 7, 14], start_steps)


class PeriodicCallbackTest(tf.test.TestCase):

  def test_every_steps(self):
    callback = mock.Mock()
    hook = periodic_actions.PeriodicCallback(
        every_steps=2, callback_fn=callback)

    for step in range(1, 10):
      hook(step, 3, remainder=step % 3)

    expected_calls = [
        mock.call(remainder=2, step=2, t=3),
        mock.call(remainder=1, step=4, t=3),
        mock.call(remainder=0, step=6, t=3),
        mock.call(remainder=2, step=8, t=3)
    ]
    self.assertListEqual(expected_calls, callback.call_args_list)

  @mock.patch("time.monotonic")
  def test_every_secs(self, mock_time):
    callback = mock.Mock()
    hook = periodic_actions.PeriodicCallback(every_secs=2, callback_fn=callback)

    for step in range(1, 10):
      mock_time.return_value = float(step)
      hook(step, remainder=step % 5)
    # Note: time will be initialized at 1 so hook runs at steps 4 & 7.
    expected_calls = [
        mock.call(remainder=4, step=4, t=4.0),
        mock.call(remainder=2, step=7, t=7.0)
    ]
    self.assertListEqual(expected_calls, callback.call_args_list)

  def test_on_steps(self):
    callback = mock.Mock()
    hook = periodic_actions.PeriodicCallback(on_steps=[8], callback_fn=callback)

    for step in range(1, 10):
      hook(step, remainder=step % 3)

    callback.assert_called_once_with(remainder=2, step=8, t=mock.ANY)

  def test_async_execution(self):
    out = []

    def cb(step, t):
      del t
      out.append(step)

    hook = periodic_actions.PeriodicCallback(
        every_steps=1, callback_fn=cb, execute_async=True)
    hook(0)
    hook(1)
    hook(2)
    hook(3)
    # Block till all the hooks have finished.
    hook.get_last_callback_result().result()
    # Check order of execution is preserved.
    self.assertListEqual(out, [0, 1, 2, 3])

  def test_error_async_is_forwarded(self):

    def cb(step, t):
      del step
      del t
      raise Exception

    hook = periodic_actions.PeriodicCallback(
        every_steps=1, callback_fn=cb, execute_async=True)

    hook(0)

    with self.assertRaises(Exception):
      hook(1)

  def test_function_without_step_and_time(self):

    # This must be used with pass_step_and_time=False.
    def cb():
      return 5

    hook = periodic_actions.PeriodicCallback(
        every_steps=1, callback_fn=cb, pass_step_and_time=False)
    hook(0)
    hook(1)
    self.assertEqual(hook.get_last_callback_result(), 5)


if __name__ == "__main__":
  tf.test.main()
