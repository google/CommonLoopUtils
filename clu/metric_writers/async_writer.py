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

"""MetricWriter that writes metrics in a separate thread.

- The order of the write calls is preserved.
- Users need to all `flush()` or use the `ensure_flushes()` context to make sure
  that all metrics have been written.
- Errors while writing in the background thread will be re-raised in the main
  thread on the next write_*() call.
"""

import contextlib
from typing import Any, Mapping, Optional, Sequence

from clu import asynclib

from clu.metric_writers import interface
from clu.metric_writers import multi_writer
import wrapt

Array = interface.Array
Scalar = interface.Scalar


@wrapt.decorator
def _wrap_exceptions(wrapped, instance, args, kwargs):
  del instance
  try:
    return wrapped(*args, **kwargs)
  except asynclib.AsyncError as e:
    raise asynclib.AsyncError(
        "Consider re-running the code without AsyncWriter (e.g. creating a "
        "writer using "
        "`clu.metric_writers.create_default_writer(asynchronous=False)`)"
    ) from e


class AsyncWriter(interface.MetricWriter):
  """MetricWriter that performs write operations in a separate thread.

  All write operations will be executed in a background thread. If an exceptions
  occurs in the background thread it will be raised on the main thread on the
  call of one of the write_* methods.

  Use num_workers > 1 at your own risk, if the underlying writer is not
  thread-safe or does not expect out-of-order events, this can cause problems.
  If num_workers is None then the ThreadPool will use `os.cpu_count()`
  processes.
  """

  def __init__(self,
               writer: interface.MetricWriter,
               *,
               num_workers: Optional[int] = 1):
    super().__init__()
    self._writer = writer
    # By default, we have a thread pool with a single worker to ensure that
    # calls to the function are run in order (but in a background thread).
    self._num_workers = num_workers
    self._pool = asynclib.Pool(
        thread_name_prefix="AsyncWriter", max_workers=num_workers)


  @_wrap_exceptions
  def write_summaries(
      self, step: int,
      values: Mapping[str, Array],
      metadata: Optional[Mapping[str, Any]] = None):
    self._pool(self._writer.write_summaries)(
        step=step, values=values, metadata=metadata)

  @_wrap_exceptions
  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    self._pool(self._writer.write_scalars)(step=step, scalars=scalars)

  @_wrap_exceptions
  def write_images(self, step: int, images: Mapping[str, Array]):
    self._pool(self._writer.write_images)(step=step, images=images)

  @_wrap_exceptions
  def write_videos(self, step: int, videos: Mapping[str, Array]):
    self._pool(self._writer.write_videos)(step=step, videos=videos)

  @_wrap_exceptions
  def write_audios(
      self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
    self._pool(self._writer.write_audios)(
        step=step, audios=audios, sample_rate=sample_rate)

  @_wrap_exceptions
  def write_texts(self, step: int, texts: Mapping[str, str]):
    self._pool(self._writer.write_texts)(step=step, texts=texts)

  @_wrap_exceptions
  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, Array],
                       num_buckets: Optional[Mapping[str, int]] = None):
    self._pool(self._writer.write_histograms)(
        step=step, arrays=arrays, num_buckets=num_buckets)

  @_wrap_exceptions
  def write_hparams(self, hparams: Mapping[str, Any]):
    self._pool(self._writer.write_hparams)(hparams=hparams)

  def flush(self):
    try:
      self._pool.join()
    finally:
      self._writer.flush()

  def close(self):
    try:
      self.flush()
    finally:
      self._writer.close()


class AsyncMultiWriter(multi_writer.MultiWriter):
  """AsyncMultiWriter writes to multiple writes in a separate thread."""

  def __init__(self,
               writers: Sequence[interface.MetricWriter],
               *,
               num_workers: Optional[int] = 1):
    super().__init__([AsyncWriter(w, num_workers=num_workers) for w in writers])


@contextlib.contextmanager
def ensure_flushes(*writers: interface.MetricWriter):
  """Context manager which ensures that one or more writers are flushed."""
  try:
    # The caller should not need to use the yielded value, but we yield
    # the first writer to stay backwards compatible for a single writer.
    yield writers[0]
  finally:
    for writer in writers:
      writer.flush()
