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

"""Methods for running triggering a profiler for accelerators.

Where results are stored depends on the platform (e.g. TensorBoard).
"""

import threading
from typing import Optional


import tensorflow as tf



def start(logdir: Optional[str] = None,
          options: Optional[tf.profiler.experimental.ProfilerOptions] = None):
  """Starts profiling."""
  if logdir is None:
    raise ValueError("Must specify logdir for tf.profiler!")
  tf.profiler.experimental.start(logdir=logdir, options=options)


def stop():
  """Stops profiling."""
  tf.profiler.experimental.stop()


