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

"""Hooks are actions that are executed during the training loop."""

import abc
import time
from typing import Optional

from clu import metric_writers
from clu import platform
from clu import profiler
from clu.google import usage_logging

usage_logging.log_import("hooks")

MetricWriter = metric_writers.MetricWriter


class Hook(abc.ABC):
  """Interface for all hooks."""

  @abc.abstractmethod
  def __call__(self, step: int, t: Optional[float] = None):
    pass


class EveryNHook(Hook):
  """Abstract base class for hooks that are executed periodically."""

  def __init__(self,
               *,
               every_steps: Optional[int] = None,
               every_secs: Optional[float] = None):
    self._every_steps = every_steps
    self._every_secs = every_secs
    self._previous_step = None
    self._previous_time = None
    self._last_step = None

  def _apply_condition(self, step: int, t: float):
    if self._every_steps is not None and step % self._every_steps == 0:
      return True
    if (self._every_secs is not None and
        t - self._previous_time > self._every_secs):
      return True
    return False

  def __call__(self, step: int, t: Optional[float] = None):
    """Method to call the hook after every training step."""
    if t is None:
      t = time.time()
    if self._previous_step is None:
      self._previous_step = step
      self._previous_time = t
      self._last_step = step
      return

    if self._every_steps is not None:
      if step - self._last_step != 1:
        raise ValueError("EveryNHook must be called after every step (once).")
    self._last_step = step

    if self._apply_condition(step, t):
      self._apply(step, t)
      self._previous_step = step
      self._previous_time = t

  @abc.abstractmethod
  def _apply(self, step: int, t: float):
    pass


class ReportProgress(EveryNHook):
  """This hook will set the progress note on the work unit."""

  def __init__(self,
               *,
               num_train_steps: int,
               writer: Optional[MetricWriter] = None,
               every_steps: Optional[int] = None,
               every_secs: Optional[float] = 60.0):
    """Creates a new ReportProgress hook.

    Warning: The progress and the reported steps_per_sec are estimates. We
    ignore the asynchronous dispatch for JAX and other operations in the
    training loop (e.g. evaluation).

    Args:
      num_train_steps: The total number of training steps for training.
      writer: Optional MetricWriter to report steps_per_sec measurement. This is
        an estimate for precise values use Xprof.
      every_steps: How often to report the progress in number of training steps.
      every_secs: How often to report progress as time interval.
    """
    super().__init__(every_steps=every_steps, every_secs=every_secs)
    self._num_train_steps = num_train_steps
    self._writer = writer

  def _apply_condition(self, step: int, t: float):
    # Always trigger at last step.
    if step == self._num_train_steps:
      return True
    return super()._apply_condition(step, t)

  def _apply(self, step: int, t: float):
    steps_per_sec = (step - self._previous_step) / (t - self._previous_time)
    eta_seconds = (self._num_train_steps - step) / steps_per_sec
    message = (f"{100 * step / self._num_train_steps:.1f}% @{step}, "
               f"{steps_per_sec:.1f} steps/s, ETA: {eta_seconds / 60:.0f} min")
    # This should be relative cheap so we can do it in the same main thread.
    platform.work_unit().set_notes(message)
    if self._writer is not None:
      self._writer.write_scalars(step, {"steps_per_sec": steps_per_sec})


class Profile(EveryNHook):
  """This hook collects a profile every time it triggers."""

  def __init__(self,
               *,
               num_profile_steps: int = 5,
               first_profile: int = 10,
               every_steps: Optional[int] = None,
               every_secs: Optional[float] = 3600.0):
    super().__init__(every_steps=every_steps, every_secs=every_secs)
    self._num_profile_steps = num_profile_steps
    self._first_profile = first_profile
    self._session_running = False

  def _apply_condition(self, step: int, t: float) -> bool:
    if self._session_running:
      if step >= self._previous_step + self._num_profile_steps:
        self._end_session()
      return False
    if step == self._first_profile:
      return True
    return super()._apply_condition(step, t)

  def _apply(self, step: int, t: float):
    del step, t  # Unused.
    self._start_session()

  def _start_session(self):
    self._session_running = True
    profiler.start()

  def _end_session(self):
    url = profiler.stop()
    if url is not None:
      platform.work_unit().create_artifact(
          platform.ArtifactType.URL,
          url,
          description=f"[{self._previous_step}] Profile")
    self._session_running = False
