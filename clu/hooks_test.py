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

"""Tests for internal metric writers."""

import time
from unittest import mock

from clu import hooks
import tensorflow as tf


class ReportProgressTest(tf.test.TestCase):

  def test_every_steps(self):
    hook = hooks.ReportProgress(
        every_steps=4, every_secs=None, num_train_steps=10)
    t = time.time()
    with self.assertLogs(level="INFO") as logs:
      hook(1, t)
      t += 0.11
      hook(2, t)
      t += 0.13
      hook(3, t)
      t += 0.12
      hook(4, t)
    # We did 1 step every 0.12s => 8.333 steps/s.
    self.assertEqual(logs.output, [
        "INFO:absl:Setting work unit notes: 40.0% @4, 8.3 steps/s, ETA: 0 min"
    ])

  def test_every_secs(self):
    hook = hooks.ReportProgress(
        every_steps=None, every_secs=0.3, num_train_steps=10)
    t = time.time()
    with self.assertLogs(level="INFO") as logs:
      hook(1, t)
      t += 0.11
      hook(2, t)
      t += 0.13
      hook(3, t)
      t += 0.12
      hook(4, t)
    # We did 1 step every 0.12s => 8.333 steps/s.
    self.assertEqual(logs.output, [
        "INFO:absl:Setting work unit notes: 40.0% @4, 8.3 steps/s, ETA: 0 min"
    ])

  def test_called_every_step(self):
    hook = hooks.ReportProgress(every_steps=3, num_train_steps=10)
    t = time.time()
    with self.assertRaisesRegex(ValueError,
                                "EveryNHook must be called after every step"):
      hook(1, t)
      # Skipping step 2.
      hook(11, t)


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

  @mock.patch.object(hooks, "profiler", autospec=True)
  def test_every_steps(self, mock_profiler):
    start_steps = []
    stop_steps = []
    step = 0

    def add_start_step():
      start_steps.append(step)

    def add_stop_step():
      stop_steps.append(step)

    mock_profiler.start.side_effect = add_start_step
    mock_profiler.stop.side_effect = add_stop_step
    hook = hooks.Profile(num_profile_steps=2, first_profile=3, every_steps=7)
    for step in range(1, 18):
      hook(step)
    self.assertAllEqual([3, 7, 14], start_steps)
    self.assertAllEqual([5, 9, 16], stop_steps)


if __name__ == "__main__":
  tf.test.main()
