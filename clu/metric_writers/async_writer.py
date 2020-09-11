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

"""MetricWriter that writes metrics in a separate thread.

- The order of the write calls is preserved.
- Users need to all `flush()` or use the `ensure_flushes()` context to make sure
  that all metrics have been written.
- Errors while writing in the background thread will be re-raised in the main
  thread on the next write_*() call.
"""

import contextlib
import multiprocessing
import sys
from typing import Any, Mapping, Optional, Sequence

from absl import logging
from clu.google import usage_logging
from clu.metric_writers import interface
from clu.metric_writers import multi_writer
import numpy as np

Scalar = interface.Scalar


class AsyncWriter(interface.MetricWriter):
  """MetricWriter that performs write operations in a separate thread.

  All write operations will be executed in a background thread. If an exceptions
  occurs in the background thread it will be raised on the main thread on the
  call of one of the write_* methods.
  """

  def __init__(self, writer: interface.MetricWriter):
    super().__init__()
    self._writer = writer
    # We have a thread pool with a single worker to ensure that calls to
    # the function are run in order (but in a background thread).
    self._worker_pool = multiprocessing.pool.ThreadPool(1)
    self._errors = [
    ]  # Tuples returned by sys.exc_info(): (type, value, traceback).
    usage_logging.log_event("create", "metric_writers.AsyncWriter")

  def _raise_previous_errors(self):
    while self._errors:
      _, value, traceback = self._errors.pop()
      logging.exception("An error occurred in a previous call.")
      raise value.with_traceback(traceback)

  def _call_async(self, func, **kwargs):
    """Call `func` with `kwargs` in the background thread."""

    def _fn(func, **kwargs):
      try:
        func(**kwargs)
      except Exception as e:
        self._errors.append(sys.exc_info())
        logging.exception(
            "Error in producer thread. Storing exception info for "
            "re-raise in the main thread.")
        raise e

    self._raise_previous_errors()
    self._worker_pool.apply_async(_fn, args=(func,), kwds=kwargs)

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    scalars = {k: np.array(v).item() for k, v in scalars.items()}
    self._call_async(self._writer.write_scalars, step=step, scalars=scalars)

  def write_images(self, step: int, images: Mapping[str, np.ndarray]):
    images = {k: np.array(v) for k, v in images.items()}
    self._call_async(self._writer.write_images, step=step, images=images)

  def write_texts(self, step: int, texts: Mapping[str, str]):
    self._call_async(self._writer.write_texts, step=step, texts=texts)

  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, np.ndarray],
                       num_buckets: Optional[Mapping[str, int]] = None):
    arrays = {k: np.array(v) for k, v in arrays.items()}
    self._call_async(
        self._writer.write_histograms,
        step=step,
        arrays=arrays,
        num_buckets=num_buckets)

  def write_hparams(self, hparams: Mapping[str, Any]):
    self._call_async(self._writer.write_hparams, hparams=hparams)

  def flush(self):
    self._raise_previous_errors()
    self._worker_pool.close()
    self._worker_pool.join()
    self._writer.flush()
    self._worker_pool = multiprocessing.pool.ThreadPool(1)


class AsyncMultiWriter(AsyncWriter):
  """AsyncMultiWriter writes to multiple writes in a separate thread."""

  def __init__(self, writers: Sequence[interface.MetricWriter]):
    super().__init__(multi_writer.MultiWriter(writers))


@contextlib.contextmanager
def ensure_flushes(writer: interface.MetricWriter):
  try:
    yield writer
  finally:
    writer.flush()
