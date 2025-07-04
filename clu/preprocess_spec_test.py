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

import dataclasses

from absl import logging
from absl.testing import parameterized
from clu import preprocess_spec
import tensorflow as tf

Features = preprocess_spec.Features
SEED_KEY = preprocess_spec.SEED_KEY


@dataclasses.dataclass(frozen=True)
class ToFloat:

  def __call__(self, features: Features) -> Features:
    return {k: tf.cast(v, tf.float32) / 255.0 for k, v in features.items()}


@dataclasses.dataclass(frozen=True)
class Rescale:

  scale: int = 1

  def __call__(self, features: Features) -> Features:
    features["image"] *= self.scale
    features["segmentation_mask"] *= self.scale
    return features


@dataclasses.dataclass(frozen=True)
class AddRandomInteger(preprocess_spec.RandomMapTransform):

  def _transform(self, features, seed):
    features["x"] = tf.random.stateless_uniform([], seed)
    return features


all_ops = lambda: preprocess_spec.get_all_ops(__name__)


class PreprocessSpecTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for parsing preprocessing op spec."""

  def test_no_arguments(self):
    op = preprocess_spec._parse_single_preprocess_op("rescale", dict(all_ops()))
    logging.info("op: %r", op)
    self.assertEqual(str(op), "Rescale(scale=1)")

  def test_positional_argument(self):
    op = preprocess_spec._parse_single_preprocess_op("rescale(2)",
                                                     dict(all_ops()))
    logging.info("op: %r", op)
    self.assertEqual(str(op), "Rescale(scale=2)")

  def test_keyword_argument(self):
    op = preprocess_spec._parse_single_preprocess_op("rescale(scale=3)",
                                                     dict(all_ops()))
    logging.info("op: %r", op)
    self.assertEqual(str(op), "Rescale(scale=3)")

  def test_invalid_op_name(self):
    with self.assertRaisesRegex(
        ValueError, r"'does_not_exist' is not available \(available ops: "
        r"\['add_random_integer', 'rescale', 'to_float'\]\)."):
      preprocess_spec._parse_single_preprocess_op("does_not_exist",
                                                  dict(all_ops()))

  def test_invalid_spec(self):
    with self.assertRaisesRegex(
        ValueError, r"'rescale\)' is not a valid preprocess op spec."):
      preprocess_spec._parse_single_preprocess_op("rescale)", dict(all_ops()))

  def test_pos_and_kw_arg(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Rescale'> given both as positional argument \(value: 2\) and keyword "
        r"argument \(value: 3\)."):
      preprocess_spec._parse_single_preprocess_op("rescale(2, scale=3)",
                                                  dict(all_ops()))

  def test_parsing_empty_string(self):
    preprocess_fn = preprocess_spec.parse("", all_ops())
    self.assertEqual(
        str(preprocess_fn), "PreprocessFn(ops=[], only_jax_types=True)")

  def test_multi_op_spec(self):
    preprocess_fn = preprocess_spec.parse("to_float|rescale(3)", all_ops())
    logging.info("preprocess_fn: %r", preprocess_fn)
    self.assertEqual(str(preprocess_fn.ops), "[ToFloat(), Rescale(scale=3)]")

  def test_two_tensors(self):
    preprocess_fn = preprocess_spec.parse("rescale(scale=7)", all_ops())
    x = {"image": tf.constant(3), "segmentation_mask": tf.constant(2)}
    y = preprocess_fn(x)
    self.assertEqual(y, {
        "image": tf.constant(21),
        "segmentation_mask": tf.constant(14),
    })

  def test_only_jax_types(self):
    preprocess_fn = preprocess_spec.parse("", all_ops())
    x = {
        "image": tf.constant(2),
        # Strings are not supported.
        "label": tf.constant("bla"),
        # Sparse tensors are not supported.
        "foo": tf.sparse.eye(4),
        # Ragged tensors are not supported.
        "bar": tf.RaggedTensor.from_tensor([[1, 2, 3], [4, 5, 6]]),
    }
    y = preprocess_fn(x)
    self.assertEqual(y, {"image": tf.constant(2)})

  def test_only_jax_types_nested_inputs(self):
    preprocess_fn = preprocess_spec.parse("", all_ops())
    x = {
        "nested": {
            "not_allowed": tf.constant("bla"),
            "allowed": tf.constant(2),
        }
    }
    y = preprocess_fn(x)
    self.assertEqual(y, {"nested": {"allowed": tf.constant(2)}})

  def test_not_only_jax_types(self):
    preprocess_fn = preprocess_spec.parse("", all_ops(), only_jax_types=False)
    x = {"image": tf.constant(2), "label": tf.constant("bla")}
    y = preprocess_fn(x)
    self.assertEqual(y, x)

  def test_add_preprocess_fn(self):
    op1 = ToFloat()
    op2 = ToFloat()
    op3 = ToFloat()
    fn1 = preprocess_spec.PreprocessFn(ops=(op1, op2), only_jax_types=False)
    fn2 = preprocess_spec.PreprocessFn(ops=(op3,), only_jax_types=True)
    fn12 = fn1 + fn2
    # Note: `+` is not supported on Sequence[PreprocessOp]; need to use `list`.
    self.assertSequenceEqual(fn12.ops, list(fn1.ops) + list(fn2.ops))
    self.assertTrue(fn12.only_jax_types)

  def test_slice_preprocess_fn(self):
    op1 = ToFloat()
    op2 = Rescale()
    op3 = ToFloat()
    fn = preprocess_spec.PreprocessFn(ops=(op1, op2, op3), only_jax_types=True)
    self.assertEqual(fn[:-1].ops, (op1, op2))
    self.assertTrue(fn[:-1].only_jax_types)
    self.assertEqual(fn[1].ops, [op2])
    self.assertTrue(fn[1].only_jax_types)

  def test_random_map_transform(self):
    ds = tf.data.Dataset.from_tensor_slices(
        {SEED_KEY: [[1, 2], [3, 4], [1, 2]]})
    ds = ds.map(AddRandomInteger())
    actual = list(ds)
    print("actual:", actual)
    expect = [
        # Random number was generated and random seed changed.
        {
            "x": 0.8838011,
            SEED_KEY: [1105988140, 1738052849]
        },
        {
            "x": 0.33396423,
            SEED_KEY: [-1860230133, -671226999]
        },
        # Same random seed as first element creates same outcome.
        {
            "x": 0.8838011,
            SEED_KEY: [1105988140, 1738052849]
        },
    ]
    self.assertAllClose(actual, expect)


if __name__ == "__main__":
  tf.test.main()
