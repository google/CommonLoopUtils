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

"""Unit tests for the deterministic_data module."""
import itertools
import math

from typing import Dict
from unittest import mock

from absl.testing import parameterized
from clu import deterministic_data
import dataclasses
import jax
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class MyDatasetBuilder:

  name2len: Dict[str, int]  # Number of examples per split.

  def as_dataset(self, split: tfds.core.ReadInstruction, shuffle_files: bool,
                 read_config: tfds.ReadConfig, decoders) -> tf.data.Dataset:
    del shuffle_files, read_config, decoders
    instructions = split.to_absolute(self.name2len)
    assert len(instructions) == 1
    from_ = instructions[0].from_ or 0
    to = instructions[0].to or self.name2len[instructions[0].splitname]
    return tf.data.Dataset.range(from_, to).map(lambda i: {"index": i})


class DeterministicDataTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for deterministic_data module."""

  @parameterized.parameters(
      (9, 0, 1, True, "test[0:9]"),
      (9, 0, 2, True, "test[0:4]"),
      (9, 1, 2, True, "test[4:8]"),  # Last example gets dropped.
      (9, 0, 3, True, "test[0:3]"),
      (9, 1, 3, True, "test[3:6]"),
      (9, 2, 3, True, "test[6:9]"),
      (9, 0, 1, False, "test[0:9]"),
      (9, 0, 2, False, "test[0:5]"),  # First host gets an extra example.
      (9, 1, 2, False, "test[5:9]"),
      (8, 0, 3, False, "test[0:3]"),  # First 2 hosts get 1 example each.
      (8, 1, 3, False, "test[3:6]"),
      (8, 2, 3, False, "test[6:8]"),
  )
  def test_get_read_instruction_for_host(self, num_examples: int, host_id: int,
                                         host_count: int, drop_remainder: bool,
                                         expected_spec: str):
    expected = tfds.core.ReadInstruction.from_spec(expected_spec)
    actual = deterministic_data.get_read_instruction_for_host(
        "test",
        num_examples,
        host_id=host_id,
        host_count=host_count,
        drop_remainder=drop_remainder)
    name2len = {"test": 9}
    self.assertEqual(
        expected.to_absolute(name2len), actual.to_absolute(name2len))

  @parameterized.parameters(
      (0, 0),  # No hosts.
      (1, 1),  # Only one host (host_id is zero-based.
      (-1, 1),  # Negative host_id.
      (5, 2),  # host_id bigger than number of hosts.
  )
  def test_get_read_instruction_for_host_fails(self, host_id: int,
                                               host_count: int):
    with self.assertRaises(ValueError):
      deterministic_data.get_read_instruction_for_host(
          "test", 11, host_id=host_id, host_count=host_count)

  def test_preprocess_with_per_example_rng(self):

    def preprocess_fn(features):
      features["b"] = tf.random.stateless_uniform([], features["rng"])
      return features

    rng = jax.random.PRNGKey(42)
    ds_in = tf.data.Dataset.from_tensor_slices({"a": [37.2, 31.2, 39.0]})
    ds_out = deterministic_data._preprocess_with_per_example_rng(
        ds_in, preprocess_fn, rng=rng)
    self.assertAllClose([
        {
            "a": 37.2,
            "b": 0.79542184
        },
        {
            "a": 31.2,
            "b": 0.45482683
        },
        {
            "a": 39.0,
            "b": 0.85335636
        },
    ], list(ds_out))

  @parameterized.named_parameters(
      ("cardinality", True),
      ("nocardinality", False),
  )
  def test_create_dataset_padding(self, cardinality):
    dataset_builder = mock.Mock()
    dataset = tf.data.Dataset.from_tensor_slices(
        dict(x=tf.ones((12, 10)), y=tf.ones(12)))
    dataset_builder.as_dataset.return_value = dataset
    batch_dims = (2, 5)
    ds = deterministic_data.create_dataset(
        dataset_builder,
        split="(ignored)",
        batch_dims=batch_dims,
        num_epochs=1,
        shuffle=False,
        pad_up_to_batches=2,
        cardinality=12 if cardinality else None,
    )
    ds_iter = iter(ds)
    self.assertAllClose(
        dict(
            x=tf.ones((2, 5, 10)),
            y=tf.ones((2, 5)),
            mask=tf.ones((2, 5), bool),
        ), next(ds_iter))
    self.assertAllClose(
        dict(
            x=tf.reshape(
                tf.concat([tf.ones(
                    (2, 10)), tf.zeros((8, 10))], axis=0), (2, 5, 10)),
            y=tf.reshape(tf.concat([tf.ones(2), tf.zeros(8)], axis=0), (2, 5)),
            mask=tf.reshape(
                tf.concat(
                    [tf.ones(2, bool), tf.zeros(8, bool)], axis=0), (2, 5)),
        ), next(ds_iter))
    with self.assertRaises(StopIteration):
      next(ds_iter)

  def test_create_dataset_padding_raises_error_cardinality(self):
    dataset_builder = mock.Mock()
    dataset = tf.data.Dataset.from_tensor_slices(
        dict(x=tf.ones((12, 10)), y=tf.ones(12)))
    dataset = dataset.filter(lambda x: True)
    dataset_builder.as_dataset.return_value = dataset
    batch_dims = (2, 5)
    with self.assertRaisesRegex(
        ValueError,
        r"^Cannot determine dataset cardinality."):
      deterministic_data.create_dataset(
          dataset_builder,
          split="(ignored)",
          batch_dims=batch_dims,
          num_epochs=1,
          shuffle=False,
          pad_up_to_batches=2,
          cardinality=None,
      )

  def test_pad_dataset(self):
    dataset = tf.data.Dataset.from_tensor_slices(
        dict(x=tf.ones((12, 10)), y=tf.ones(12)))
    padded_dataset = deterministic_data.pad_dataset(
        dataset, batch_dims=[20], pad_up_to_batches=2, cardinality=12)
    self.assertAllClose(
        dict(
            x=tf.concat([tf.ones(
                (12, 10)), tf.zeros((8, 10))], axis=0),
            y=tf.concat([tf.ones(12), tf.zeros(8)], axis=0),
            mask=tf.concat(
                [tf.ones(12, bool), tf.zeros(8, bool)], axis=0)),
        next(iter(padded_dataset.batch(20))))

  def test_pad_nested_dataset(self):
    dataset = tf.data.Dataset.from_tensor_slices(
        {"x": {"z": (tf.ones((12, 10)), tf.ones(12))},
         "y": tf.ones((12, 4))})

    def expected(*dims):
      return tf.concat([tf.ones((12,) + dims), tf.zeros((8,) + dims)], axis=0)

    padded_dataset = deterministic_data.pad_dataset(
        dataset, batch_dims=[20], pad_up_to_batches=2, cardinality=12)
    self.assertAllClose(
        {"x": {"z": (expected(10), expected())},
         "y": expected(4),
         "mask": tf.concat([tf.ones(12, bool), tf.zeros(8, bool)], axis=0)},
        next(iter(padded_dataset.batch(20))))

  @parameterized.parameters(*itertools.product(range(20), range(1, 4)))
  def test_same_cardinality_on_all_hosts(self, num_examples: int,
                                         host_count: int):
    builder = MyDatasetBuilder({"train": num_examples})
    cardinalities = []
    for host_id in range(host_count):
      split = deterministic_data.get_read_instruction_for_host(
          split="train",
          num_examples=num_examples,
          host_id=host_id,
          host_count=host_count,
          drop_remainder=True)
      ds = deterministic_data.create_dataset(
          builder, split=split, batch_dims=[2], shuffle=False, num_epochs=1)
      cardinalities.append(ds.cardinality().numpy().item())
    self.assertLen(set(cardinalities), 1)

  @parameterized.parameters(*itertools.product(range(20), range(1, 4)))
  def test_same_cardinality_on_all_hosts_with_pad(self, num_examples: int,
                                                  host_count: int):
    builder = MyDatasetBuilder({"train": num_examples})
    # All hosts should have the same number of batches.
    batch_size = 2
    pad_up_to_batches = int(math.ceil(num_examples / (batch_size * host_count)))
    assert pad_up_to_batches * batch_size * host_count >= num_examples
    cardinalities = []
    for host_id in range(host_count):
      split = deterministic_data.get_read_instruction_for_host(
          split="train",
          num_examples=num_examples,
          host_id=host_id,
          host_count=host_count,
          drop_remainder=False)
      ds = deterministic_data.create_dataset(
          builder,
          split=split,
          batch_dims=[batch_size],
          shuffle=False,
          num_epochs=1,
          pad_up_to_batches=pad_up_to_batches)
      cardinalities.append(ds.cardinality().numpy().item())
    self.assertLen(set(cardinalities), 1)

  def test_create_dataset_with_padded_shapes(self):
    dataset_builder = mock.Mock()
    dataset = tf.data.Dataset.from_tensor_slices(
        dict(x=tf.ones((12, 3, 2)), y=tf.ones((12, 3))))
    dataset_builder.as_dataset.return_value = dataset
    batch_dims = (4, 3)
    padded_shapes = {"x": (5, 4), "y": (5,)}
    ds = deterministic_data.create_dataset(
        dataset_builder,
        split="(ignored)",
        batch_dims=batch_dims,
        shuffle=False,
        padded_shapes=padded_shapes,
        )
    expected_shapes = {
        "x": batch_dims + padded_shapes["x"],
        "y": batch_dims + padded_shapes["y"],
    }
    self.assertEqual(
        expected_shapes, {k: v.shape for k, v in ds.element_spec.items()})


if __name__ == "__main__":
  tf.test.main()
