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

"""Utilities for async function calls."""

import collections
import concurrent.futures
import functools
import sys
import threading
from typing import Callable, List, Optional

from absl import logging


class Pool:
  """Pool for wrapping functions to be executed asynchronously.

  Synopsis:

    from clu.internal import asynclib

    pool = asynclib.Pool()
    @pool
    def fn():
      time.sleep(1)

    future = fn()
    print(future.result())
    fn()  # This could re-raise an exception from the first execution.
    print(len(pool))  # Would print "1" because there is one function in flight.
    pool.flush()  # This could re-raise an exception from the second execution.
  """

  def __init__(self, thread_name_prefix: str = "",
               max_workers: Optional[int] = None):
    """Creates a new pool that decorates functions for async execution.

    Args:
      thread_name_prefix: See documentation of `ThreadPoolExecutor`.
      max_workers: See documentation of `ThreadPoolExecutor`. The default `None`
        optimizes for parallelizability using the number of CPU cores. If you
        specify `max_workers=1` you the async calls are executed in the same
        order they have been scheduled.
    """
    self._pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=thread_name_prefix)
    self._max_workers = max_workers
    self._thread_name_prefix = thread_name_prefix
    self._errors = collections.deque()
    self._errors_mutex = threading.Lock()
    self._queue_length = 0

  def _reraise(self) -> None:
    if self._errors:
      with self._errors_mutex:
        exc_info = self._errors.popleft()
      raise exc_info[1].with_traceback(exc_info[2])

  def close(self) -> None:
    """Closes this pool & raise a pending exception (if needed)."""
    self._pool.shutdown(wait=True)
    self._reraise()

  def join(self) -> None:
    """Blocks until all functions are processed.

    The pool can be used to schedule more functions after calling this function,
    but there might be more exceptions

    Side-effect:
      If any of the functions raised an exception, then the first of these
      exceptions is reraised.
    """
    self._pool.shutdown(wait=True)
    self._pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_workers,
        thread_name_prefix=self._thread_name_prefix)
    self._reraise()

  @property
  def queue_length(self) -> int:
    """Returns the number of functions that have not returned yet."""
    return self._queue_length

  @property
  def has_errors(self) -> bool:
    """Returns True if there are any pending errors."""
    return bool(self._errors)

  def clear_errors(self) -> List[Exception]:
    """Clears all pending errors and returns them as a (possibly empty) list."""
    with self._errors_mutex:
      errors, self._errors = self._errors, collections.deque()
    return list(errors)

  def __call__(self, fn: Callable):  # pylint: disable=g-bare-generic
    """Returns an async version of fn.

    The function will be executed by this class's ThreadPoolExecutor. Any errors
    will be stored and re-raised next time any function is called that is
    executed through this pool.

    Note that even if there was a previous error, the function is still
    scheduled upon re-execution of the wrapper returned by this function.

    Args:
      fn: Function to be wrapped.

    Returns:
      An async version of `fn`. The return value of that async version will be
      a future (unless an exception was re-raised).
    """

    def inner(*args, **kwargs):

      def trap_errors(*args, **kwargs):
        try:
          return fn(*args, **kwargs)
        except Exception as e:
          with self._errors_mutex:
            self._errors.append(sys.exc_info())
          logging.exception("Error in producer thread for %s",
                            self._thread_name_prefix)
          raise e
        finally:
          self._queue_length -= 1

      self._queue_length += 1
      if not self.has_errors:
        return self._pool.submit(trap_errors, *args, **kwargs)
      self._pool.submit(trap_errors, *args, **kwargs)
      self._reraise()

    if isinstance(fn.__name__, str):
      # Regular function.
      return functools.wraps(fn)(inner)
    # Mock or another weird function that fails with functools.wraps().
    return inner
