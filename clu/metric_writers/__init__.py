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

"""Public interface for metric writers. See src/metric_writers.py."""

# pylint: disable=unused-import


from clu.metric_writers.async_writer import AsyncMultiWriter
from clu.metric_writers.async_writer import AsyncWriter
from clu.metric_writers.async_writer import ensure_flushes
from clu.metric_writers.interface import MetricWriter
from clu.metric_writers.logging_writer import LoggingWriter
from clu.metric_writers.multi_writer import MultiWriter
from clu.metric_writers.summary_writer import SummaryWriter



def create_default_writer(
    logdir: str,
    *,
    just_logging: bool = False,
    asynchronous: bool = True) -> MetricWriter:
  """Create the default writer for the platform.

  On most platforms this will create a MultiWriter that writes to multiple back
  ends (logging, TF summaries etc.).

  Args:
    logdir: Logging dir to use for TF summary files.
    just_logging: If True only use a LoggingWriter. This is useful in multi-host
      setups when only the first host should write metrics and all other hosts
      should only write to their own logs.
    asynchronous: If True return an AsyncMultiWriter to not block when writing
      metrics.

  Returns:
    A `MetricWriter` according to the platform and arguments.
  """
  if just_logging:
    if asynchronous:
      return AsyncWriter(LoggingWriter())
    else:
      return LoggingWriter()

  writers = [LoggingWriter(), SummaryWriter(logdir)]
  if asynchronous:
    return AsyncMultiWriter(writers)
  return MultiWriter(writers)
