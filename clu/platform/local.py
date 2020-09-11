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

from typing import Any

from absl import logging
from clu.platform import interface

WorkUnit = interface.WorkUnit
ArtifactType = interface.ArtifactType


class LocalWorkUnit(WorkUnit):
  """Dummy work unit for running locally."""

  @property
  def experiment_id(self):
    """ID of the experiment of the work unit."""
    return -1

  @property
  def id(self):
    """Unique identifier for the work unit."""
    return -1

  def set_notes(self, msg: str):
    """Set the notes for this work unit."""
    logging.info("Setting work unit notes: %s", msg)

  def set_task_status(self, msg: str):
    """Set the status string for this task."""
    logging.info("Setting task status: %s", msg)

  def create_artifact(self, artifact_type: ArtifactType, artifact: Any,
                      description: str):
    """Creates an artifact entry for the work unit."""
    logging.info("Created artifact %s of type %s and value %s.", description,
                 artifact_type, artifact)
