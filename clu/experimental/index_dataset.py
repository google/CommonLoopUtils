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

"""This module provides functionality to create index datasets for tf.data.

Index datasets only contain a monotonically increasing index, a dataset_id
(in case multiple datasets are mixed) and an integer key referencing to record
in the dataset on disk.

We use the following terms:
- A **dataset** is a set of files on disk.
- A **record** is an entry stored in a file on disk. A record has a position
  inside a specific file and a **key** in a dataset.
- An **example** is a transformation of a record. Input pipelines
  might iterate over datasets multiple times (also called epoch) in which case
  multilple examples are derived from the same record (usually by applying
  random transformations).
- **Index** is a monotonically increasing integer that indicates how far we have
  advanced our input pipeline (defined using tf.data).

An index dataset contains elements that each have an index, a dataset ID and a
record key. We can use the dataset ID and the record key to look up the
corresponding example on disk. And we can use the index as random seed (if we
have multiple epochs over the dataset the same key will appear multiple times
but with different indices).

In this module we assume that we know the number of records in our datasets
and that records are keyed by an integer in [0, num_records_in_dataset).

Warning: **This is work in progress!** It requires cl/426924178 and a file
format that support efficient random lookup. Contact mrit@ if you are interested
in using this.
"""

from typing import Sequence, List, Tuple, Union, Optional

from absl import logging
import numpy as np
import tensorflow as tf


def _shuffle(index: tf.Tensor, *, seed: tf.Tensor,
             num_records: tf.Tensor) -> tf.Tensor:
  """Shuffles the `index` within the interval [0, num_records - 1]."""
  # We use a different random seed per epoch.
  epoch = index // num_records
  epoch_seed = tf.random.experimental.stateless_fold_in(seed, epoch)
  key = index % num_records
  # max_index is inclusive.
  return tf.random.experimental.index_shuffle(
      key, seed=epoch_seed, max_index=tf.cast(num_records - 1, tf.int64))


def _shard(num_examples: int, *, shard_id: int, num_shards: int,
           drop_remainder: bool) -> Tuple[int, int]:
  """Calculates the interval for this shard when sharding num_examples.

  This splits the interval [0, num_examples - 1] into num_shards intervals
  and returns the shard_id's interval. If drop_remainder is True all intervals
  will have the same size.

  Args:
    num_examples: Number of examples to shard.
    shard_id: The index of the shard for which to calculate the interval. Must
      be in [0, num_shards - 1].
    num_shards: Number of shards to split num_examples into.
    drop_remainder: If True we create even splits and drop the remainder (all
      shards will have the same number of examples). If False will distribute
      the remainder N over the first N shards.

  Returns:
    Tuple with the start and end of the interval. The start is the first
    example that should be included in this interval and end - 1 is the last
    example to be include in the shard.
  """
  if num_shards < 0:
    raise ValueError(
        f"Number of shards must be a positive integer but got {num_shards}.")
  if shard_id < 0 or shard_id >= num_shards:
    raise ValueError(
        f"Shard id must be in [0, num_shards - 1], num_shards was {num_shards} "
        f"and shard_id was {shard_id}.")

  examples_per_shard = num_examples // num_shards
  shard_start = examples_per_shard * shard_id
  shard_end = examples_per_shard * (shard_id + 1)

  # Handle remaining examples.
  num_unused_examples = num_examples % num_shards

  if num_unused_examples > 0:
    if drop_remainder:
      logging.warning("Dropping %d examples of %d examples (shard %d).",
                      num_unused_examples, num_examples, num_shards)
    else:
      shard_start += min(shard_id, num_unused_examples)
      shard_end += min(shard_id + 1, num_unused_examples)
  return shard_start, shard_end


def _float_to_int_proportions(values: Sequence[Union[float, int]],
                              scale_min_to: int = 100) -> List[int]:
  """Scales at values by `scale_min_to/min(proportions)` and cast to int."""
  scale_factor = scale_min_to / min(values)
  return [int(p * scale_factor) for p in values]


def _counts_per_dataset(k: tf.Tensor,
                        proportions: Sequence[int]) -> List[tf.Tensor]:
  """Calculates the counts per dataset at n elements accordings to proportions.

  We are interleaving n infinite datasets into one combined dataset.

  Proportions P is a list of n integers, representing mixing proportions.

  mix(P, k, i) represents the number of examples from component i
  among the first k examples from the mixed sequence. It is given by the
  following formula:

    mix(P, k, 0) = ceiling(k * P[0] / sum(P))
    mix(P, k, i>0) = mix(P[1:], k - mix(P, k, 0), i - 1)

  Element k of the mixed sequence is equal to element m from component i iff:

    mix(P, k + 1, i) == m + 1  AND
    mix(P, k, i) == m

  _counts_per_dataset() computes the "mix" function described above.

  _dataset_and_key_of_next_element() maps from the index in the combined
  dataset to identity of the ID of the source dataset and key in the source
  dataset.

  Args:
    k: Number of elements of the mixed sequence.
    proportions: The mixing proportions for the n dataset.

  Returns:
    Counts of how many elements from each source dataset are used.
  """
  remaining_proportions = sum(proportions)
  result = []
  for p in proportions:
    new_k = (k * (remaining_proportions - p)) // remaining_proportions
    result.append(k - new_k)
    remaining_proportions -= p
    k = new_k
  return result


def _dataset_and_key_of_next_element(
    k: tf.Tensor, proportions: Sequence[int]) -> Tuple[tf.Tensor, tf.Tensor]:
  """Compute the dataset and the key for interleaved datasets at position k.

  We are interleaving n infinite datasets into one combined dataset.

  See the description in _counts_per_dataset() above.

  Args:
    k: Index in the combined dataset.
    proportions: The mixing proportions for the n dataset.

  Returns:
    A tuple with the index of the source dataset and the key in it for the
    element at index `k` of the combined dataset.
  """
  old_counts = tf.stack(_counts_per_dataset(k, proportions))
  new_counts = tf.stack(_counts_per_dataset(k + 1, proportions))
  # For the new dataset the count increased by 1. All other counts should be
  # the same.
  dataset_id = tf.math.argmax(new_counts - old_counts)
  return dataset_id, new_counts[dataset_id] - 1


def _get_shard_size_and_offset(records_in_dataset,
                               **shard_fn_args) -> Tuple[int, int]:
  shard_start, shard_end = _shard(records_in_dataset, **shard_fn_args)
  return shard_end - shard_start, shard_start


def _get_shard_sizes_and_offsets_for_mixture(
    records_per_dataset: List[int],
    **shard_fn_args) -> Tuple[List[int], List[int]]:
  # Shard each dataset separately.
  shard_starts, shard_ends = zip(
      *[_shard(n, **shard_fn_args) for n in records_per_dataset])  # pylint: disable=missing-kwoa
  records_per_dataset = [
      end - start for start, end in zip(shard_starts, shard_ends)
  ]
  return records_per_dataset, shard_starts


def create_index_dataset(
    records_per_dataset: Union[int, Sequence[int]],
    *,
    proportions: Optional[Sequence[Union[int, float]]] = None,
    start_index: int = 0,
    num_epochs: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[Union[tf.Tensor, Tuple[int, int]]] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    sharding_drop_remainder: bool = False) -> tf.data.Dataset:
  """Creates a new index dataset.

  See the module description for an explanation of the idea.

  Args:
    records_per_dataset: Number of examples per dataset. Provide a sequence to
      mix multiple datasets.
    proportions: Proportions when mixing multiple datasets. If not provided all
      datasets will be mixed with equal proportions. Proportions are relative to
      the sum of all proportions each other at both float and integers can be
      mixed. E.g. when mixing two datasets both [0.25, 0.75] and [1, 3] result
      in the same ratio of 1:3 between the first and the second dataset.
    start_index: Where to start the input pipeline. Usually 0 to start from the
      beginning and otherwise the last seen index+1 before the input pipeline
      was stopped.
    num_epochs: Integer if iterating over a fixed number of epochs. The dataset
      will be finite and have known size. Not supported when mixing multiple
      datasets.
    shuffle: Whether to shuffle record keys. If True you need to provide `seed`.
    seed: Random seed to use for shuffling. This should be a tensor of shape
      [2].
    shard_id: Shard number for which to construct the index dataset. Must be in
      [0, num_shards - 1].
    num_shards: Number of shards if sharding an input pipeline. Each shard would
      get continuous regions of record keys.
    sharding_drop_remainder: Whether to drop the remainder when sharding the
      input pipeline. If False shards might have different sizes with the first
      N shards talking the remaining N examples.

  Returns:
    A `tf.data.Dataset` containing Dict[str, tf.Tensor]. The dictionary will
    contain an 'index' and a 'record_key'. When mixing multiple datasets it will
    also contain a 'dataset_id'.
  """
  # Need seed if shuffling.
  if shuffle:
    if seed is None:
      raise ValueError("Shuffling requires specifying a seed.")
    if num_shards > 1:
      seed = tf.random.experimental.stateless_fold_in(seed, shard_id)
  elif seed is not None:
    logging.warning("Provided seed will not be used since shuffling is off.")

  is_mixture = not isinstance(records_per_dataset, int)

  # Preparations for sharding.
  if num_shards > 1:
    if is_mixture:
      shard_fn = _get_shard_sizes_and_offsets_for_mixture
    else:
      shard_fn = _get_shard_size_and_offset
    records_per_dataset, position_offset_per_dataset = shard_fn(
        records_per_dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        drop_remainder=sharding_drop_remainder)

  # Preparations for mixing
  if is_mixture:
    if proportions is None:
      proportions = len(records_per_dataset) * [1]
    else:
      assert len(proportions) == len(records_per_dataset)
      proportions = _float_to_int_proportions(proportions)

  if num_epochs is None:
    end_index = np.iinfo(np.int64).max
  else:
    if is_mixture:
      # What is an epoch when mixing multiple datasetsw with different number
      # of records or proportions?
      raise ValueError(
          "Using fixed number of epochs is not allowed when mixing datasets.")
    end_index = records_per_dataset * num_epochs

  # We define one map function that goes from index to global index, position
  # and dataset_id.
  # Note: Please use tf.int64 everywhere to avoid type mismatch errors.
  if is_mixture:

    # Turn lists into tensors. This way we can do lookup using slicing.
    records_per_dataset = tf.stack(records_per_dataset)
    if num_shards > 1:
      position_offset_per_dataset = tf.stack(position_offset_per_dataset)

    def map_fn(index):
      assert index.dtype == tf.int64
      dataset_id, index_in_dataset = _dataset_and_key_of_next_element(
          index, proportions)
      num_records_in_dataset = tf.cast(records_per_dataset[dataset_id],
                                       tf.int64)
      if shuffle:
        record_key = _shuffle(
            index_in_dataset, seed=seed, num_records=num_records_in_dataset)
      else:
        record_key = index_in_dataset % num_records_in_dataset
      if num_shards > 1:
        # Make index global.
        index = index * num_shards + shard_id
        record_key += tf.cast(position_offset_per_dataset[dataset_id], tf.int64)
      return {
          "index": index,
          "record_key": record_key,
          "dataset_id": dataset_id,
      }
  else:

    def map_fn(index):
      assert index.dtype == tf.int64
      if shuffle:
        record_key = _shuffle(index, seed=seed, num_records=records_per_dataset)
      else:
        record_key = index % records_per_dataset
      if num_shards > 1:
        # Make index global.
        index = index * num_shards + shard_id
        record_key += position_offset_per_dataset
      return {"index": index, "record_key": record_key}

  ds = tf.data.Dataset.range(start_index, end_index)
  ds = ds.map(map_fn)
  return ds
