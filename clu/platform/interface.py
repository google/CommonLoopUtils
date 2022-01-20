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

"""Interface work units."""

import abc
import enum
from typing import Any


class ArtifactType(enum.Enum):
  # A URL for dashboards, etc.
  URL = 1
  # File path.
  FILE = 2
  # Directory path.
  DIRECTORY = 3


class WorkUnit(abc.ABC):
  """A work unit represents a single trial in an experiment.

  Experiments will usually have multiple work units with different
  hyperparameters. Each work unit can have multiple jobs (training,
  evaluation, etc.). And jobs can have multiple tasks when the training
  is distributed across multiple machines.
  """

  @property
  @abc.abstractmethod
  def experiment_id(self):
    """ID of the experiment of the work unit."""

  @property
  @abc.abstractmethod
  def id(self):
    """Unique identifier for the work unit."""

  @property
  def name(self):
    """Returns the name of the work unit as <XID>/<WID>.

    XID is a ID of the experiment and WID is the number of the work unit
    within the experiment.

    Returns:
      The work unit name. e.g. 12345/1.
    """
    return f"{self.experiment_id}/{self.id}"

  @abc.abstractmethod
  def set_notes(self, msg: str):
    """Sets the notes for this work unit. These are displayed in the UI."""

  @abc.abstractmethod
  def set_task_status(self, msg: str):
    """Sets the status string for this task."""

  @abc.abstractmethod
  def create_artifact(self, artifact_type: ArtifactType, artifact: Any,
                      description: str):
    """Creates an artifact entry for the work unit."""
