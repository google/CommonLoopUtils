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

"""Simple checkpointing library for TF2/Flax.

The class `Checkpoint` is a simple wrapper around `tf.train.Checkpoint` that
also stores a `flax.struct.dataclass` instance in the same directory.

Synopsis:

  from clu import checkpoint
  import flax

  @flax.struct.dataclass
  class TrainState:
    optimizer: flax.optim.Optimizer
    step: int

  ds = load_tf_dataset()
  ds_iter = iter(ds)
  ckpt = checkpoint.MultihostCheckpoint(base_directory, dict(ds_iter=ds_iter))
  optimizer = create_flax_optimizer()
  state = TrainState(optimizer=optimizer, step=0)
  state = ckpt.restore_or_initialize(state)  # Also restores `ds_iter`.
  initial_step = int(state.step) + 1
  # Need to replicate all data when training with multiple accelerators.
  state = flax.jax_utils.replicate(state)

  for step in range(initial_step, steps + 1):
    state = update_step(state, next(ds_iter))
    ckpt.save(flax.jax_utils.unreplicate(state))

Loading the model e.g. in a Colab:

  from clu import checkpoint
  import flax
  from . import mnist_lib

  state_dict = checkpoint.load_state_dict(base_directory)
  params = state_dict['optimizer']['target']['params']
  module = mnist_lib.MyArchitecture.partial(num_classes=10)
  model = flax.nn.Model(module, params)
"""

import collections
import re
from typing import Any, Dict, Optional, TypeVar

from absl import logging

from clu.internal import utils
import flax
import jax
import tensorflow as tf




T = TypeVar("T")


def load_state_dict(base_directory) -> Dict[str, Any]:
  """Restores `state` as dictionary from the latest checkpoint.

  Synopsis:

    data = checkpoint.load_state_dict(base_directory)
    params = data['optimizer']['target']['params']
    module = mnist_lib.MyArchitecture.partial(num_classes=10)
    model = flax.nn.Model(module, params)

  Args:
    base_directory: Directory from which the checkpoints should be restored. See
      `Checkpoint.__init__()`.

  Returns:
    The deserialized Flax data, as a dictionary.

  Raises:
    FileNotFoundError: If there is no checkpoint to restore.
  """
  ckpt = Checkpoint(base_directory)
  if not ckpt.latest_checkpoint:
    raise FileNotFoundError(f"No checkpoint found in {base_directory}")
  with utils.log_activity("load_state_dict"):
    with tf.io.gfile.GFile(ckpt.latest_checkpoint_flax, "rb") as f:
      return flax.serialization.msgpack_restore(f.read())


class CheckpointInfo(
    collections.namedtuple("CheckpointInfo", ("prefix", "number"))):
  """Helper class to parse a TensorFlow checkpoint path."""

  CHECKPOINT_REGEX = r"^(?P<prefix>.*)-(?P<number>\d+)"

  @classmethod
  def initialize(cls, base_directory, checkpoint_name: str) -> "CheckpointInfo":
    """Creates a first CheckpointInfo (number=1)."""
    return cls(f"{base_directory}/{checkpoint_name}", 1)

  @classmethod
  def from_path(cls, checkpoint: str) -> "CheckpointInfo":
    """Parses a checkpoint.

    Args:
      checkpoint: A checkpoint prefix, as can be found in the
        `.latest_checkpoint` property of a `tf.train.CheckpointManager`.

    Returns:
      An instance of `CheckpointInfo` that represents `checkpoint`.
    """
    m = re.match(cls.CHECKPOINT_REGEX, checkpoint)
    if m is None:
      RuntimeError(f"Invalid checkpoint format: {checkpoint}")
    d = m.groupdict()  # pytype: disable=attribute-error
    return cls(d["prefix"], int(d["number"]))

  def increment(self) -> "CheckpointInfo":
    """Returns a new CheckpointInfo with `number` increased by one."""
    return CheckpointInfo(self.prefix, self.number + 1)

  def __str__(self):
    """Does the opposite of `.from_path()`."""
    return f"{self.prefix}-{self.number}"


class Checkpoint:
  """A utility class for storing and loading TF2/Flax checkpoints.

  Both the state of a `tf.data.Dataset` iterator and a `flax.struct.dataclass`
  are stored on disk in the following files:

  - {directory}/checkpoint
  - {directory}/ckpt-{number}.index
  - {directory}/ckpt-{number}.data@*
  - {directory}/ckpt-{number}.flax

  Where {number} starts at 1 is then incremented by 1 for every new checkpoint.
  The last file is the `flax.struct.dataclass`, serialized in Messagepack
  format. The other files are explained in more detail in the Tensorflow
  documentation:

  https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
  """

  def __init__(self,
               base_directory: str,
               tf_state: Optional[Dict[str, Any]] = None,
               *,
               max_to_keep: int = 5,
               checkpoint_name: str = "ckpt"):
    """Initializes a Checkpoint with a dictionary of TensorFlow Trackables.

    Args:
      base_directory: Directory under which the checkpoints will be stored. Use
        a different base_directory in every task.
      tf_state: A dictionary of TensorFlow `Trackable` to be serialized, for
        example a dataset iterator.
      max_to_keep: Number of checkpoints to keep in the directory. If there are
        more checkpoints than specified by this number, then the oldest
        checkpoints are removed.
      checkpoint_name: Prefix of the checkpoint files (before `-{number}`).
    """
    if tf_state is None:
      tf_state = dict()
    self.base_directory = base_directory
    self.max_to_keep = max_to_keep
    self.checkpoint_name = checkpoint_name
    self.tf_checkpoint = tf.train.Checkpoint(**tf_state)
    self.tf_checkpoint_manager = tf.train.CheckpointManager(
        self.tf_checkpoint,
        base_directory,
        max_to_keep=max_to_keep,
        checkpoint_name=checkpoint_name)

  def get_latest_checkpoint_to_restore_from(self):
    """Returns the latest checkpoint to restore from.

    In the current implementation, this method simply returns the attribute
    `latest_checkpoint`.

    Subclasses can override this method to provide an alternative checkpoint to
    restore from, for example for synchronization across multiple checkpoint
    directories.
    """
    return self.latest_checkpoint

  @property
  def latest_checkpoint(self) -> Optional[str]:
    """Latest checkpoint, see `tf.train.CheckpointManager.latest_checkpoint`.

    Returns:
      A string to the latest checkpoint. Note that this string is path-like but
      it does not really describe a file, but rather a set of files that are
      constructed from this string, by appending different file extensions. The
      returned value is `None` if there is no previously stored checkpoint in
      `base_directory` specified to `__init__()`.
    """
    return self.tf_checkpoint_manager.latest_checkpoint

  @property
  def latest_checkpoint_flax(self) -> Optional[str]:
    """Path of the latest serialized `state`.

    Returns:
      Path of the file containing the serialized Flax state. The returned value
      is `None` if there is no previously stored checkpoint in `base_directory`
      specified to `__init__()`.
    """
    if self.latest_checkpoint is None:
      return None
    return self._flax_path(self.latest_checkpoint)

  def _flax_path(self, checkpoint: str) -> str:
    return "{}.flax".format(checkpoint)

  def _next_checkpoint(self, checkpoint: Optional[str]) -> str:
    if checkpoint is None:
      return str(
          CheckpointInfo.initialize(self.base_directory, self.checkpoint_name))
    return str(CheckpointInfo.from_path(checkpoint).increment())

  def _checkpoint_number(self, checkpoint: Optional[str]) -> Optional[int]:
    if checkpoint is not None:
      return CheckpointInfo.from_path(checkpoint).number

  @utils.logged_with("Checkpoint.save()")
  def save(self, state) -> str:
    """Saves a new checkpoints in the directory.

    Args:
      state: Flax checkpoint to be stored.

    Returns:
      The checkpoint identifier ({base_directory}/ckpt-{number}).

    Raises:
      RuntimeError: If tf_checkpoint.save_counter does not match
          tf_checkpoint_manager.latest_checkpoint.
    """
    latest_checkpoint_num = self._checkpoint_number(self.latest_checkpoint) or 0
    if latest_checkpoint_num != self.tf_checkpoint.save_counter.numpy():
      raise RuntimeError(
          f"Expected save_counter={self.tf_checkpoint.save_counter.numpy()} "
          f"to match latest_checkpoint={self.latest_checkpoint}. Make sure "
          f"the checkpoint is initialized via `.restore_or_initialize()` "
          f"before it's stored and that no other process writes to the same "
          f"checkpoint directory.")
    next_checkpoint = self._next_checkpoint(self.latest_checkpoint)
    flax_path = self._flax_path(next_checkpoint)
    if not tf.io.gfile.exists(self.base_directory):
      tf.io.gfile.makedirs(self.base_directory)
    with tf.io.gfile.GFile(flax_path, "wb") as f:
      f.write(flax.serialization.to_bytes(state))
    checkpoints = set(self.tf_checkpoint_manager.checkpoints)
    # Write Tensorflow data last. This way Tensorflow checkpoint generation
    # logic will make sure to only commit checkpoints if they complete
    # successfully. A previously written `flax_path` would then simply be
    # overwritten next time.
    self.tf_checkpoint_manager.save()
    for checkpoint in checkpoints.difference(
        self.tf_checkpoint_manager.checkpoints):
      tf.io.gfile.remove(self._flax_path(checkpoint))
    if next_checkpoint != self.latest_checkpoint:
      raise AssertionError(  # pylint: disable=g-doc-exception
          "Expected next_checkpoint to match latest_checkpoint: "
          f"{next_checkpoint} != {self.latest_checkpoint}")
    return self.latest_checkpoint

  @utils.logged_with("Checkpoint.restore_or_initialize()")
  def restore_or_initialize(self,
                            state: T,
                            checkpoint: Optional[str] = None) -> T:
    """Restores from the latest checkpoint, or creates a first checkpoint.

    Args:
      state : A flax checkpoint to be stored or to serve as a template. If the
        checkoint is restored (and not initialized), then the fields of `state`
        must match the data previously stored.
      checkpoint: A flax checkpoint to be restored. If not specified, the
        latest checkpoint is restored.

    Returns:
      The restored `state` object. Note that all TensorFlow `Trackable`s in
      `tf_state` (see `__init__()`) are also updated.
    """
    if checkpoint:
      checkpoint_to_restore = checkpoint
    else:
      logging.info("No checkpoint specified. Restore the latest checkpoint.")
      checkpoint_to_restore = self.get_latest_checkpoint_to_restore_from()
    if not checkpoint_to_restore:
      logging.info("Checkpoint %s does not exist.", checkpoint_to_restore)
      self.save(state)
      return state
    logging.info("Restoring checkpoint: %s", checkpoint_to_restore)
    self.tf_checkpoint.restore(checkpoint_to_restore)
    flax_path = self._flax_path(checkpoint_to_restore)
    with tf.io.gfile.GFile(flax_path, "rb") as f:
      state = flax.serialization.from_bytes(state, f.read())
    logging.info("Restored save_counter=%d restored_checkpoint=%s",
                 self.tf_checkpoint.save_counter.numpy(),
                 checkpoint_to_restore)
    return state

  def restore(self, state: T, checkpoint: Optional[str] = None) -> T:
    """Restores from the latest checkpoint.

    Similar to `restore_or_initialize()`, but raises a `FileNotFoundError` if
    there is no checkpoint.

    Args:
      state : A flax checkpoint to be stored or to serve as a template. If the
        checkoint is restored (and not initialized), then the fields of `state`
        must match the data previously stored.
      checkpoint: A flax checkpoint path to be restored. If not specified, the
        latest checkpoint is restored.

    Returns:
      The restored `state` object. Note that all TensorFlow `Trackable`s in
      `tf_state` (see `__init__()`) are also updated.

    Raises:
      FileNotFoundError: If specified checkpoint does not exist, or if there
      is no checkpoint to restore in case no checkpoint was specified.
    """
    if checkpoint:
      if not tf.io.gfile.exists(self._flax_path(checkpoint)):
        raise FileNotFoundError(f"Checkpoint {checkpoint} does not exist")
    else:
      checkpoint = self.get_latest_checkpoint_to_restore_from()
      if not checkpoint:
        raise FileNotFoundError(f"No checkpoint found at {self.base_directory}")
    return self.restore_or_initialize(state, checkpoint)


class MultihostCheckpoint(Checkpoint):
  """An subclass of `Checkpoint` that synchronizes between multiple JAX hosts.

  If the training split across multiple hosts, then the following race condition
  can occur : If a host is pre-empted while writing a checkpoint, then the other
  hosts will only be restarted with a small delay, and at that point they
  probably already have finished writing their checkpoint. Upon restart, the
  host that was interrupted while writing the checkpoint will load the latest
  fully written checkpoint, which will be out of sync with the other hosts that
  successfully wrote one more checkpoint.

  This class also allows to specify a `multihost_base_directory` that is
  identical for all hosts and will be used to drive a host-specific directory.
  """

  def __init__(self,
               multihost_base_directory: str,
               tf_state: Optional[Dict[str, Any]] = None,
               *,
               host_id: Optional[int] = None,
               max_to_keep: int = 5,
               checkpoint_name: str = "ckpt"):
    """Initializes a MultihostCheckpoint with a dict of TensorFlow Trackables.

    Args:
      multihost_base_directory: Directory that will be used to construct a
        host-specific `base_directory` under which the checkpoints will be
        stored. Usually a directory *within* the work unit's workdirectory
        (e.g. `f"{workdir}/checkpoints`). One directory per host will be created
        at the same level as this base directory labeled
        `f"{multihost_base_directory}-{host_id}"`.
      tf_state: A dictionary of TensorFlow `Trackable` to be serialized, for
        example a dataset iterator.
      host_id: Host ID used to construct the `base_directory`. Taken from
        `jax.host_id()` if not specified.
      max_to_keep: Number of checkpoints to keep in the directory. If there are
        more checkpoints than specified by this number, then the oldest
        checkpoints are removed.
      checkpoint_name: Prefix of the checkpoint files (before `-{number}`).
    """
    if max_to_keep < 2:
      raise ValueError("Requires multiple checkpoints (max_to_keep>=2).")
    multihost_base_directory = multihost_base_directory.rstrip("/")
    self.multihost_base_directory = multihost_base_directory
    if host_id is None:
      host_id = jax.host_id()
    base_directory = f"{multihost_base_directory}-{host_id}"
    super().__init__(
        base_directory,
        tf_state,
        max_to_keep=max_to_keep,
        checkpoint_name=checkpoint_name)

  @utils.logged_with(
      "MultihostCheckpoint.get_latest_checkpoint_to_restore_from()")
  def get_latest_checkpoint_to_restore_from(self) -> Optional[str]:
    """Returns the latest checkpoint available on all hosts."""
    base_directory_glob = f"{self.multihost_base_directory}-*"
    base_directories = tf.io.gfile.glob(base_directory_glob)
    if self.base_directory not in base_directories:
      return None
    checkpoints = {}
    common_numbers = None
    all_numbers = set()
    for base_directory in base_directories:
      checkpoint_manager = tf.train.CheckpointManager(
          tf.train.Checkpoint(),
          base_directory,
          max_to_keep=self.max_to_keep,
          checkpoint_name=self.checkpoint_name)
      numbers = [
          CheckpointInfo.from_path(checkpoint).number
          for checkpoint in checkpoint_manager.checkpoints
      ]
      checkpoints[base_directory] = dict(
          zip(numbers, checkpoint_manager.checkpoints))
      numbers = set(numbers)
      if common_numbers is None:
        common_numbers = numbers
      else:
        common_numbers &= numbers
      all_numbers |= numbers
    logging.info(
        "Checked checkpoint base_directories: %s - common_numbers=%s "
        "- exclusive_numbers=%s", base_directories, common_numbers,
        all_numbers.difference(common_numbers))
    if not common_numbers:
      return None
    highest_number = sorted(common_numbers)[-1]
    return checkpoints[self.base_directory][highest_number]
