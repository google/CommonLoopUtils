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

"""Metric writers write ML model outputs during model training and evaluation.

This module introduces the MetricWriter interface. MetricWriters allow users
to write out metrics about ML models during training and evaluation (e.g. loss,
accuracy).
There is a MetricWriter implementation for each back end (e.g. TensorFlow
summaries) and classes that work on top other MetricWriter to
write to multiple writes at once or write asynchronously.

Note: The current interface might not contain write() methods for all possible
data types. We are open for extending the interface to other data types
(e.g. audio).

Usage:
  writer = MyMetricWriterImplementation()
  # Before training.
  writer.write_hparams({"learning_rate": 0.001, "batch_size": 64})
  # Start training loop.
  for step in range(num_train_steps):
    loss = train_step()
    if step % 50 == 0:
      writer.write_scalars(step, {"loss": loss})
      accuracy = evaluate()
      writer.write_scalars(step, {"accuracy": accuracy})
  # Make sure all values were written.
  writer.flush()  # or use metric_writers.ensure_flushes() context.
"""

# pylint: disable=unused-import

from typing import Optional, List, Tuple, Union

from absl import flags

from clu.metric_writers.async_writer import AsyncMultiWriter
from clu.metric_writers.async_writer import AsyncWriter
from clu.metric_writers.async_writer import ensure_flushes
from clu.metric_writers.interface import MetricWriter
from clu.metric_writers.logging_writer import LoggingWriter
from clu.metric_writers.multi_writer import MultiWriter
from clu.metric_writers.summary_writer import SummaryWriter
from clu.metric_writers.utils import write_values


# TODO(b/200953513): Migrate away from logging imports (on module level)
#                    to logging the actual usage. See b/200953513.


FLAGS = flags.FLAGS


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
    write_to_xm_measurements: If True uses XmMeasurementsWriter in addition.
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
