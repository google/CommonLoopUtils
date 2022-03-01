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

"""Unit tests for the index_dataset module."""
import itertools

from absl.testing import parameterized
from clu.experimental import index_dataset
import tensorflow as tf

# Some options for testing determinism and restart behavior.
# For each setting there are 2 options and then we test all combinations.
_RECORDS_PER_DATASET = (7, [7, 5])
_PROPORTIONS = (None, [0.3, 0.4])
_SHUFFLE = (True, False)
_NUM_SHARDS = (1, 3)


class IndexDatasetTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the index_dataset module."""

  def test_simple(self):
    """No shuffling, no sharding, no mixing."""
    dataset = index_dataset.create_index_dataset(6)
    values = list(dataset.take(15).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     # First epoch.
                     [{"index": 0, "record_key": 0},
                      {"index": 1, "record_key": 1},
                      {"index": 2, "record_key": 2},
                      {"index": 3, "record_key": 3},
                      {"index": 4, "record_key": 4},
                      {"index": 5, "record_key": 5},
                      # Second epoch.
                      {"index": 6, "record_key": 0},
                      {"index": 7, "record_key": 1},
                      {"index": 8, "record_key": 2},
                      {"index": 9, "record_key": 3},
                      {"index": 10, "record_key": 4},
                      {"index": 11, "record_key": 5},
                      # Third epoch.
                      {"index": 12, "record_key": 0},
                      {"index": 13, "record_key": 1},
                      {"index": 14, "record_key": 2}])
    # pyformat: enable

  def test_num_epochs(self):
    """Setting the number of epochs yields a finite dataset."""
    dataset = index_dataset.create_index_dataset(4, num_epochs=2)
    values = list(dataset.take(15).as_numpy_iterator())
    self.assertEqual(dataset.cardinality(), 8)
    # pyformat: disable
    self.assertEqual(values,
                     # First epoch.
                     [{"index": 0, "record_key": 0},
                      {"index": 1, "record_key": 1},
                      {"index": 2, "record_key": 2},
                      {"index": 3, "record_key": 3},
                      # Second epoch.
                      {"index": 4, "record_key": 0},
                      {"index": 5, "record_key": 1},
                      {"index": 6, "record_key": 2},
                      {"index": 7, "record_key": 3}])
    # pyformat: enable

  def test_num_epochs_with_mixture_fails(self):
    """Mixing datasets with fixed number epochs is not allowed."""
    with self.assertRaisesRegex(
        ValueError,
        "Using fixed number of epochs is not allowed when mixing datasets."):
      index_dataset.create_index_dataset([4, 8], num_epochs=2)

  def test_shuffle(self):
    """Shuffling, no sharding, no mixing."""
    dataset = index_dataset.create_index_dataset(6, shuffle=True, seed=(32, 73))
    values = list(dataset.take(15).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     # First epoch.
                     [{"index": 0, "record_key": 1},
                      {"index": 1, "record_key": 0},
                      {"index": 2, "record_key": 4},
                      {"index": 3, "record_key": 3},
                      {"index": 4, "record_key": 2},
                      {"index": 5, "record_key": 5},
                      # Second epoch.
                      {"index": 6, "record_key": 1},
                      {"index": 7, "record_key": 4},
                      {"index": 8, "record_key": 2},
                      {"index": 9, "record_key": 0},
                      {"index": 10, "record_key": 3},
                      {"index": 11, "record_key": 5},
                      # Third epoch.
                      {"index": 12, "record_key": 5},
                      {"index": 13, "record_key": 2},
                      {"index": 14, "record_key": 3}])
    # pyformat: enable

  def test_sharding_drop_remainder(self):
    dataset = index_dataset.create_index_dataset(
        8, shard_id=0, num_shards=3, sharding_drop_remainder=True)
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 0, "record_key": 0},
                      {"index": 3, "record_key": 1},
                      {"index": 6, "record_key": 0},
                      {"index": 9, "record_key": 1}])
    # pyformat: enable
    dataset = index_dataset.create_index_dataset(
        8, shard_id=1, num_shards=3, sharding_drop_remainder=True)
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 1, "record_key": 2},
                      {"index": 4, "record_key": 3},
                      {"index": 7, "record_key": 2},
                      {"index": 10, "record_key": 3}])
    # pyformat: enable
    dataset = index_dataset.create_index_dataset(
        8, shard_id=2, num_shards=3, sharding_drop_remainder=True)
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 2, "record_key": 4},
                      {"index": 5, "record_key": 5},
                      {"index": 8, "record_key": 4},
                      {"index": 11, "record_key": 5}])
    # pyformat: enable

  def test_sharding_no_drop_remainder(self):
    dataset = index_dataset.create_index_dataset(
        8, shard_id=0, num_shards=3, sharding_drop_remainder=False)
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 0, "record_key": 0},
                      {"index": 3, "record_key": 1},
                      {"index": 6, "record_key": 2},
                      {"index": 9, "record_key": 0}])
    # pyformat: enable
    dataset = index_dataset.create_index_dataset(
        8, shard_id=1, num_shards=3, sharding_drop_remainder=False)
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 1, "record_key": 3},
                      {"index": 4, "record_key": 4},
                      {"index": 7, "record_key": 5},
                      {"index": 10, "record_key": 3}])
    # pyformat: enable
    dataset = index_dataset.create_index_dataset(
        8, shard_id=2, num_shards=3, sharding_drop_remainder=False)
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 2, "record_key": 6},
                      {"index": 5, "record_key": 7},
                      {"index": 8, "record_key": 6},
                      {"index": 11, "record_key": 7}])
    # pyformat: enable

  def test_mixing_equal_probability(self):
    """Test mixing with 2 datasets with 4 and 6 elements."""
    dataset = index_dataset.create_index_dataset([4, 6])
    values = list(dataset.take(16).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     # First epoch for both datasets.
                     [{"index": 0, "record_key": 0, "dataset_id": 0},
                      {"index": 1, "record_key": 0, "dataset_id": 1},
                      {"index": 2, "record_key": 1, "dataset_id": 0},
                      {"index": 3, "record_key": 1, "dataset_id": 1},
                      {"index": 4, "record_key": 2, "dataset_id": 0},
                      {"index": 5, "record_key": 2, "dataset_id": 1},
                      {"index": 6, "record_key": 3, "dataset_id": 0},
                      {"index": 7, "record_key": 3, "dataset_id": 1},
                      # First dataset is finished and starts second epoch.
                      {"index": 8, "record_key": 0, "dataset_id": 0},
                      {"index": 9, "record_key": 4, "dataset_id": 1},
                      {"index": 10, "record_key": 1, "dataset_id": 0},
                      {"index": 11, "record_key": 5, "dataset_id": 1},
                      # Second dataset also starts second epoch.
                      {"index": 12, "record_key": 2, "dataset_id": 0},
                      {"index": 13, "record_key": 0, "dataset_id": 1},
                      {"index": 14, "record_key": 3, "dataset_id": 0},
                      {"index": 15, "record_key": 1, "dataset_id": 1}])
    # pyformat: enable

  def test_mixing_with_integer_proportions(self):
    """Test mixing with 2 datasets with 4 and 6 elements."""
    dataset = index_dataset.create_index_dataset([3, 4], proportions=[1, 4])
    values = list(dataset.take(16).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     # First epoch for both datasets.
                     [{"index": 0, "record_key": 0, "dataset_id": 0},
                      {"index": 1, "record_key": 0, "dataset_id": 1},
                      {"index": 2, "record_key": 1, "dataset_id": 1},
                      {"index": 3, "record_key": 2, "dataset_id": 1},
                      {"index": 4, "record_key": 3, "dataset_id": 1},
                      {"index": 5, "record_key": 1, "dataset_id": 0},
                      # Second dataset is finished and starts second epoch.
                      {"index": 6, "record_key": 0, "dataset_id": 1},
                      {"index": 7, "record_key": 1, "dataset_id": 1},
                      {"index": 8, "record_key": 2, "dataset_id": 1},
                      {"index": 9, "record_key": 3, "dataset_id": 1},
                      {"index": 10, "record_key": 2, "dataset_id": 0},
                      # Second dataset starts third epoch.
                      {"index": 11, "record_key": 0, "dataset_id": 1},
                      {"index": 12, "record_key": 1, "dataset_id": 1},
                      {"index": 13, "record_key": 2, "dataset_id": 1},
                      {"index": 14, "record_key": 3, "dataset_id": 1},
                      # First dataset is finished and starts second epoch.
                      {"index": 15, "record_key": 0, "dataset_id": 0}])
    # pyformat: enable

  def test_mixing_with_float_proportions(self):
    """Test mixing with 2 datasets with 4 and 6 elements."""
    dataset = index_dataset.create_index_dataset([3, 4], proportions=[0.2, 0.8])
    values = list(dataset.take(16).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     # First epoch for both datasets.
                     [{"index": 0, "record_key": 0, "dataset_id": 0},
                      {"index": 1, "record_key": 0, "dataset_id": 1},
                      {"index": 2, "record_key": 1, "dataset_id": 1},
                      {"index": 3, "record_key": 2, "dataset_id": 1},
                      {"index": 4, "record_key": 3, "dataset_id": 1},
                      {"index": 5, "record_key": 1, "dataset_id": 0},
                      # Second dataset is finished and starts second epoch.
                      {"index": 6, "record_key": 0, "dataset_id": 1},
                      {"index": 7, "record_key": 1, "dataset_id": 1},
                      {"index": 8, "record_key": 2, "dataset_id": 1},
                      {"index": 9, "record_key": 3, "dataset_id": 1},
                      {"index": 10, "record_key": 2, "dataset_id": 0},
                      # Second dataset starts third epoch.
                      {"index": 11, "record_key": 0, "dataset_id": 1},
                      {"index": 12, "record_key": 1, "dataset_id": 1},
                      {"index": 13, "record_key": 2, "dataset_id": 1},
                      {"index": 14, "record_key": 3, "dataset_id": 1},
                      # First dataset is finished and starts second epoch.
                      {"index": 15, "record_key": 0, "dataset_id": 0}])
    # pyformat: enable

  def test_shuffle_and_sharding(self):
    dataset = index_dataset.create_index_dataset(
        6, shuffle=True, seed=(32, 73), shard_id=0, num_shards=2)
    values = list(dataset.take(6).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 0, "record_key": 2},
                      {"index": 2, "record_key": 0},
                      {"index": 4, "record_key": 1},
                      # Second epoch.
                      {"index": 6, "record_key": 2},
                      {"index": 8, "record_key": 0},
                      {"index": 10, "record_key": 1}])
    # pyformat: enable
    dataset = index_dataset.create_index_dataset(
        6, shuffle=True, seed=(42, 73), shard_id=1, num_shards=2)
    values = list(dataset.take(6).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 1, "record_key": 4},
                      {"index": 3, "record_key": 5},
                      {"index": 5, "record_key": 3},
                      # Second epoch.
                      {"index": 7, "record_key": 3},
                      {"index": 9, "record_key": 4},
                      {"index": 11, "record_key": 5}])
    # pyformat: enable

  def test_mixing_and_sharding(self):
    dataset = index_dataset.create_index_dataset([4, 6],
                                                 shard_id=0,
                                                 num_shards=2)
    values = list(dataset.take(10).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 0, "record_key": 0, "dataset_id": 0},
                      {"index": 2, "record_key": 0, "dataset_id": 1},
                      {"index": 4, "record_key": 1, "dataset_id": 0},
                      {"index": 6, "record_key": 1, "dataset_id": 1},
                      # second epoch of first dataset starts.
                      {"index": 8, "record_key": 0, "dataset_id": 0},
                      {"index": 10, "record_key": 2, "dataset_id": 1},
                      {"index": 12, "record_key": 1, "dataset_id": 0},
                      # second epoch of second dataset starts.
                      {"index": 14, "record_key": 0, "dataset_id": 1},
                      {"index": 16, "record_key": 0, "dataset_id": 0},
                      {"index": 18, "record_key": 1, "dataset_id": 1}])
    # pyformat: enable
    dataset = index_dataset.create_index_dataset([4, 6],
                                                 shard_id=1,
                                                 num_shards=2)
    values = list(dataset.take(10).as_numpy_iterator())
    # pyformat: disable
    self.assertEqual(values,
                     [{"index": 1, "record_key": 2, "dataset_id": 0},
                      {"index": 3, "record_key": 3, "dataset_id": 1},
                      {"index": 5, "record_key": 3, "dataset_id": 0},
                      {"index": 7, "record_key": 4, "dataset_id": 1},
                      # second epoch of first dataset starts.
                      {"index": 9, "record_key": 2, "dataset_id": 0},
                      {"index": 11, "record_key": 5, "dataset_id": 1},
                      {"index": 13, "record_key": 3, "dataset_id": 0},
                      # second epoch of second dataset starts.
                      {"index": 15, "record_key": 3, "dataset_id": 1},
                      {"index": 17, "record_key": 2, "dataset_id": 0},
                      {"index": 19, "record_key": 4, "dataset_id": 1}])
    # pyformat: enable

  @parameterized.parameters(
      itertools.product(_RECORDS_PER_DATASET, _PROPORTIONS, _SHUFFLE,
                        _NUM_SHARDS))
  def test_determinism(self, records_per_dataset, proportions, shuffle: bool,
                       num_shards: int):
    """Creating the dataset twice gives the same result."""
    seed = (3, 84) if shuffle else None
    dataset = index_dataset.create_index_dataset(
        records_per_dataset,
        proportions=proportions,
        shuffle=shuffle,
        seed=seed,
        num_shards=num_shards)
    values_1 = list(dataset.take(50).as_numpy_iterator())
    dataset = index_dataset.create_index_dataset(
        records_per_dataset,
        proportions=proportions,
        shuffle=shuffle,
        seed=seed,
        num_shards=num_shards)
    values_2 = list(dataset.take(50).as_numpy_iterator())
    self.assertAllEqual(values_1, values_2)

  @parameterized.parameters(
      itertools.product(_RECORDS_PER_DATASET, _PROPORTIONS, _SHUFFLE,
                        _NUM_SHARDS))
  def test_start_index(self, records_per_dataset, proportions, shuffle: bool,
                       num_shards: int):
    """We can start anyway and get the same elements."""
    seed = (3, 84) if shuffle else None
    dataset = index_dataset.create_index_dataset(
        records_per_dataset,
        proportions=proportions,
        shuffle=shuffle,
        seed=seed,
        num_shards=num_shards)
    all_values = list(dataset.take(50).as_numpy_iterator())
    for start_index in range(1, 30):
      dataset = index_dataset.create_index_dataset(
          records_per_dataset,
          proportions=proportions,
          shuffle=shuffle,
          seed=seed,
          num_shards=num_shards,
          start_index=start_index)
      values = list(dataset.take(50 - start_index).as_numpy_iterator())
      self.assertAllEqual(all_values[start_index:], values)


if __name__ == "__main__":
  tf.test.main()
