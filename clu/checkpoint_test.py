# Copyright 2025 The CLU Authors.
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

"""Tests for clu.checkpoint."""

import os
import tempfile
from unittest import mock

from clu import checkpoint
import flax
import tensorflow as tf


def _make_dataset():
  inputs = tf.range(10.)[:, None]
  labels = inputs * 5. + tf.range(5.)[None, :]
  features = dict(x=inputs, y=labels)
  return tf.data.Dataset.from_tensor_slices(features).repeat().batch(2)


@flax.struct.dataclass
class TrainState:
  step: int


@flax.struct.dataclass
class TrainStateExtended:
  step: int
  name: str


class NotTrainState:
  pass


def _checkpoint_number(path):
  *parts, number = path.split("-")
  del parts
  return int(number)


class CheckpointTest(tf.test.TestCase):

  def test_safe_normpath(self):
    self.assertEqual(checkpoint.safe_normpath("./test_dir"), "test_dir")
    self.assertEqual(checkpoint.safe_normpath(".//test_dir"), "test_dir")
    self.assertEqual(checkpoint.safe_normpath("gs://test_dir"), "gs://test_dir")
    self.assertEqual(
        checkpoint.safe_normpath("gs://test_dir/"), "gs://test_dir")

  def test_initialize_mkdir(self):
    base_dir = os.path.join(tempfile.mkdtemp(), "test")
    state = TrainState(step=1)
    ckpt = checkpoint.Checkpoint(base_dir)
    self.assertIsNone(ckpt.current_checkpoint)
    self.assertIsNone(ckpt.latest_checkpoint)
    self.assertFalse(os.path.isdir(base_dir))
    state = ckpt.restore_or_initialize(state)
    self.assertIsNotNone(ckpt.latest_checkpoint)
    self.assertEqual(ckpt.latest_checkpoint, ckpt.current_checkpoint)
    self.assertTrue(os.path.isdir(base_dir))

  def test_restores_flax_state(self):
    base_dir = tempfile.mkdtemp()
    state = TrainState(step=1)
    ckpt = checkpoint.Checkpoint(base_dir, max_to_keep=2)
    # Initializes.
    state = ckpt.restore_or_initialize(state)
    state = TrainState(step=0)
    # Restores step=1.
    state = ckpt.restore_or_initialize(state)
    self.assertEqual(state.step, 1)
    state = TrainState(step=2)
    # Stores step=2.
    path = ckpt.save(state)
    self.assertEqual(_checkpoint_number(path), 2)
    state = TrainState(step=0)
    # Restores step=2.
    state = ckpt.restore(state)
    self.assertEqual(state.step, 2)
    state = TrainState(step=3)
    # Stores step=3
    path2 = ckpt.save(state)
    self.assertEqual(_checkpoint_number(path2), 3)
    state = TrainState(step=0)
    # Restores step=2.
    state = ckpt.restore(state, path)
    self.assertEqual(state.step, 2)

  def test_load_state_dict(self):
    base_dir = tempfile.mkdtemp()
    state = TrainState(step=1)
    ckpt = checkpoint.Checkpoint(base_dir)
    # Initializes.
    state = ckpt.restore_or_initialize(state)
    # Load via load_state_dict().
    flax_dict = checkpoint.load_state_dict(base_dir)
    self.assertEqual(flax_dict, dict(step=1))
    with self.assertRaisesRegex(FileNotFoundError, r"^No checkpoint found"):
      checkpoint.load_state_dict(tempfile.mkdtemp())

  def test_fails_when_restoring_subset(self):
    base_dir = tempfile.mkdtemp()
    state = TrainStateExtended(step=1, name="test")
    ckpt = checkpoint.Checkpoint(base_dir)
    # Initialixes with TrainStateExtended.
    state = ckpt.restore_or_initialize(state)
    state = TrainState(step=0)
    # Restores with TrainState.
    with self.assertRaisesRegex(ValueError, r"^Unknown field"):
      state = ckpt.restore_or_initialize(state)

  def test_fails_when_restoring_superset(self):
    base_dir = tempfile.mkdtemp()
    ckpt = checkpoint.Checkpoint(base_dir)
    state = TrainState(step=0)
    # Initialixes with TrainState.
    state = ckpt.restore_or_initialize(state)
    state = TrainStateExtended(step=1, name="test")
    # Restores with TrainStateExtended.
    with self.assertRaisesRegex(ValueError, r"^Missing field"):
      state = ckpt.restore_or_initialize(state)

  def test_restores_tf_state(self):
    base_dir = tempfile.mkdtemp()
    ds_iter = iter(_make_dataset())
    ckpt = checkpoint.Checkpoint(base_dir, dict(ds_iter=ds_iter))
    features0 = next(ds_iter)  # Advance iterator by one.
    del features0
    state = TrainState(step=1)
    # Initialize at features1.
    state = ckpt.restore_or_initialize(state)
    features1 = next(ds_iter)
    features2 = next(ds_iter)
    self.assertNotAllEqual(features1["x"], features2["x"])
    self.assertNotAllEqual(features1["y"], features2["y"])
    # Restore at features1.
    state = ckpt.restore_or_initialize(state)
    features1_restored = next(ds_iter)
    self.assertAllEqual(features1["x"], features1_restored["x"])
    self.assertAllEqual(features1["y"], features1_restored["y"])
    # Save at features2.
    path = ckpt.save(state)
    self.assertEqual(_checkpoint_number(path), 2)
    features2 = next(ds_iter)
    features3 = next(ds_iter)
    self.assertNotAllEqual(features2["x"], features3["x"])
    self.assertNotAllEqual(features2["y"], features3["y"])
    # Restore at features2.
    state = ckpt.restore_or_initialize(state)
    features2_restored = next(ds_iter)
    self.assertAllEqual(features2["x"], features2_restored["x"])
    self.assertAllEqual(features2["y"], features2_restored["y"])
    # Restore at features2 as dictionary.
    state = ckpt.restore_dict()
    features2_restored = next(ds_iter)
    self.assertAllEqual(features2["x"], features2_restored["x"])
    self.assertAllEqual(features2["y"], features2_restored["y"])

  def test_restore_flax_alone(self):
    base_dir = tempfile.mkdtemp()
    ds_iter = iter(_make_dataset())
    ckpt = checkpoint.Checkpoint(base_dir, dict(ds_iter=ds_iter))
    state = TrainState(step=1)
    # Initializes.
    state = ckpt.restore_or_initialize(state)
    state = TrainState(step=0)
    ckpt = checkpoint.Checkpoint(base_dir)
    # Restores step=1.
    state = ckpt.restore_or_initialize(state)
    self.assertEqual(state.step, 1)

  def test_restore_dict(self):
    base_dir = tempfile.mkdtemp()
    ds_iter = iter(_make_dataset())
    ckpt = checkpoint.Checkpoint(base_dir, dict(ds_iter=ds_iter))
    with self.assertRaisesRegex(FileNotFoundError, r"No checkpoint found at"):
      ckpt.restore_dict()
    with self.assertRaisesRegex(FileNotFoundError,
                                r"Checkpoint invalid does not exist"):
      ckpt.restore_dict(checkpoint="invalid")

    state = TrainState(step=1)
    ckpt.save(state)

    state_dict = ckpt.restore_dict()
    self.assertEqual(state_dict, dict(step=1))
    first_checkpoint = ckpt.latest_checkpoint

    new_state = TrainState(step=2)
    ckpt.save(new_state)

    self.assertEqual(
        ckpt.restore_dict(checkpoint=first_checkpoint),
        dict(step=1))
    self.assertEqual(ckpt.restore_dict(), dict(step=2))
    self.assertEqual(
        ckpt.restore_dict(checkpoint=ckpt.latest_checkpoint),
        dict(step=2))

  def test_ignores_incomplete_checkpoint(self):
    base_dir = tempfile.mkdtemp()
    state = TrainState(step=1)
    ckpt = checkpoint.Checkpoint(base_dir)
    # Initializes.
    state = ckpt.restore_or_initialize(state)
    state = TrainState(step=0)
    # Restores step=1.
    state = ckpt.restore_or_initialize(state)
    self.assertEqual(state.step, 1)
    state = TrainState(step=2)
    # Failed save : step=2 is stored, but TensorFlow checkpoint fails.
    ckpt.tf_checkpoint_manager.save = None
    with self.assertRaisesRegex(TypeError,
                                r"'NoneType' object is not callable"):
      ckpt.save(state)
    files = os.listdir(base_dir)
    self.assertIn("ckpt-2.flax", files)
    self.assertNotIn("ckpt-2.index", files)
    ckpt = checkpoint.Checkpoint(base_dir)
    state = TrainState(step=0)
    # Restores step=1.
    state = ckpt.restore_or_initialize(state)
    self.assertEqual(state.step, 1)
    # Stores step=2.
    state = TrainState(step=2)
    path = ckpt.save(state)
    self.assertEqual(_checkpoint_number(path), 2)
    files = os.listdir(base_dir)
    self.assertIn("ckpt-2.flax", files)
    self.assertIn("ckpt-2.index", files)
    state = TrainState(step=0)
    # Restores step=2.
    state = ckpt.restore_or_initialize(state)
    self.assertEqual(state.step, 2)

  def test_max_to_keep(self):
    base_dir = tempfile.mkdtemp()
    state = TrainState(step=1)
    ckpt = checkpoint.Checkpoint(base_dir, max_to_keep=1)
    state = ckpt.restore_or_initialize(state)
    files1 = os.listdir(base_dir)
    state = TrainState(step=2)
    path = ckpt.save(state)
    self.assertEqual(_checkpoint_number(path), 2)
    files2 = os.listdir(base_dir)
    self.assertEqual(len(files1), len(files2))
    self.assertNotEqual(files1, files2)

  def test_checkpoint_name(self):
    base_dir = tempfile.mkdtemp()
    state = TrainState(step=1)
    ckpt = checkpoint.Checkpoint(base_dir, checkpoint_name="test")
    path = ckpt.save(state)
    self.assertIn("test", path)

  def test_fails_if_not_registered(self):
    base_dir = tempfile.mkdtemp()
    not_state = NotTrainState()
    ckpt = checkpoint.Checkpoint(base_dir)
    with self.assertRaisesRegex(TypeError, r"serialize"):
      ckpt.restore_or_initialize(not_state)

  def test_overwrite(self):
    base_dir = tempfile.mkdtemp()
    tf_step = tf.Variable(1)
    state = TrainState(step=1)
    ckpt = checkpoint.Checkpoint(base_dir, dict(step=tf_step))
    # Initialize step=1.
    state = ckpt.restore_or_initialize(state)
    self.assertEqual(state.step, 1)
    self.assertEqual(tf_step.numpy(), 1)
    checkpoint_info = checkpoint.CheckpointInfo.from_path(
        ckpt.current_checkpoint)
    # Stores steps 2, 3, 4, 5
    for _ in range(4):
      tf_step.assign_add(1)
      state = state.replace(step=state.step + 1)
      ckpt.save(state)
    latest_checkpoint = str(checkpoint_info._replace(number=5))
    self.assertEqual(ckpt.current_checkpoint, latest_checkpoint)
    self.assertEqual(ckpt.latest_checkpoint, latest_checkpoint)
    # Restores at step=1
    ckpt = checkpoint.Checkpoint(base_dir, dict(step=tf_step))
    state = ckpt.restore(state, checkpoint=str(checkpoint_info))
    self.assertEqual(state.step, 1)
    self.assertEqual(tf_step.numpy(), 1)
    self.assertNotEqual(ckpt.current_checkpoint, ckpt.latest_checkpoint)
    self.assertEqual(ckpt.current_checkpoint, str(checkpoint_info))
    self.assertEqual(ckpt.latest_checkpoint, latest_checkpoint)
    # Overwrites step=2, deletes 3, 4, 5.
    tf_step.assign_add(1)
    state = state.replace(step=state.step + 1)
    ckpt.save(state)
    latest_checkpoint = str(checkpoint_info._replace(number=2))
    self.assertEqual(ckpt.current_checkpoint, latest_checkpoint)
    self.assertEqual(ckpt.latest_checkpoint, latest_checkpoint)


class MultihostCheckpoint(tf.test.TestCase):

  @mock.patch("jax.process_index")
  def test_initialize_mkdir(self, process_index_mock):
    multihost_base_dir = os.path.join(tempfile.mkdtemp(), "test")
    state = TrainState(step=1)
    process_index_mock.return_value = 0
    base_dir = f"{multihost_base_dir}-0"
    ckpt = checkpoint.MultihostCheckpoint(multihost_base_dir)
    self.assertIsNone(ckpt.latest_checkpoint)
    self.assertFalse(os.path.isdir(base_dir))
    state = ckpt.restore_or_initialize(state)
    self.assertIsNotNone(ckpt.latest_checkpoint)
    self.assertTrue(os.path.isdir(base_dir))

  @mock.patch("jax.process_index")
  def test_synchronize_multiple_hosts(self, process_index_mock):
    multihost_base_dir = os.path.join(tempfile.mkdtemp(), "test")
    state = TrainState(step=1)
    process_index_mock.return_value = 0
    ckpt_0 = checkpoint.MultihostCheckpoint(multihost_base_dir)
    process_index_mock.return_value = 1
    ckpt_1 = checkpoint.MultihostCheckpoint(multihost_base_dir)
    # Initialize both at step=1.
    state_0 = ckpt_0.restore_or_initialize(state)
    state_1 = ckpt_1.restore_or_initialize(state)
    # Update both at step=2.
    state_0 = state_0.replace(step=2)
    ckpt_0.save(state_0)
    state_1 = state_1.replace(step=2)
    ckpt_1.save(state_1)
    # Update ckpt_1 at step=3.
    state_1 = state_1.replace(step=3)
    ckpt_1.save(state_1)
    # Reload both at step=2.
    process_index_mock.return_value = 0
    ckpt_0 = checkpoint.MultihostCheckpoint(multihost_base_dir)
    process_index_mock.return_value = 1
    ckpt_1 = checkpoint.MultihostCheckpoint(multihost_base_dir)
    self.assertEqual(ckpt_0.latest_checkpoint,
                     ckpt_0.get_latest_checkpoint_to_restore_from())
    self.assertNotEqual(ckpt_1.latest_checkpoint,
                        ckpt_1.get_latest_checkpoint_to_restore_from())
    state_0 = ckpt_0.restore_or_initialize(state)
    state_1 = ckpt_1.restore_or_initialize(state)
    self.assertEqual(state_0.step, 2)
    self.assertEqual(state_1.step, 2)

  def test_preemption(self):
    multihost_base_dir = os.path.join(tempfile.mkdtemp(), "test")
    state = TrainState(step=1)
    state0 = state.replace(step=0)
    ckpt_0 = checkpoint.MultihostCheckpoint(multihost_base_dir, host_id=0)
    ckpt_1 = checkpoint.MultihostCheckpoint(multihost_base_dir, host_id=1)
    # Initialize both at step=1.
    state_0 = ckpt_0.restore_or_initialize(state)
    state_1 = ckpt_1.restore_or_initialize(state)
    self.assertEqual(state_0.step, 1)
    self.assertEqual(state_1.step, 1)
    # Restore both at step=1.
    state_0 = ckpt_0.restore_or_initialize(state0)
    state_1 = ckpt_1.restore_or_initialize(state0)
    self.assertEqual(state_0.step, 1)
    self.assertEqual(state_1.step, 1)
    # Update only ckpt_0 to step=2.
    state_0 = state_0.replace(step=2)
    ckpt_0.save(state_0)
    # Load both checkpoints at last common step=1.
    ckpt_0 = checkpoint.MultihostCheckpoint(multihost_base_dir, host_id=0)
    ckpt_1 = checkpoint.MultihostCheckpoint(multihost_base_dir, host_id=1)
    state_0 = ckpt_0.restore_or_initialize(state)
    state_1 = ckpt_1.restore_or_initialize(state)
    self.assertEqual(state_0.step, 1)
    self.assertEqual(state_1.step, 1)
    # Store both at step=2.
    state_0 = state_0.replace(step=2)
    state_1 = state_1.replace(step=2)
    ckpt_0.save(state_0)
    ckpt_1.save(state_1)
    # Restore both at step=2.
    state_0 = ckpt_0.restore_or_initialize(state0)
    state_1 = ckpt_1.restore_or_initialize(state0)
    self.assertEqual(state_0.step, 2)
    self.assertEqual(state_1.step, 2)

if __name__ == "__main__":
  tf.test.main()
