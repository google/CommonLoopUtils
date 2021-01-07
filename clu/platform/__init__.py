# Copyright 2021 The CLU Authors.
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

"""Methods for interacting with the experiment platform.

Use cases include informing the platform of the experiment status and providing
a platform independent interface for interactions.
"""

import threading


from clu.platform.interface import ArtifactType
from clu.platform.interface import WorkUnit
from clu.platform.local import LocalWorkUnit


_work_unit = None
_work_unit_lock = threading.Lock()


def work_unit() -> WorkUnit:
  """Gets the global work unit for this experiment trial."""
  global _work_unit
  if _work_unit is None:
    with _work_unit_lock:
      _work_unit = LocalWorkUnit()
  return _work_unit
