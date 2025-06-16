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

"""Tests for dataset_iterator."""
import itertools
import pathlib
import tempfile

from absl.testing import parameterized
from clu.data import dataset_iterator
import numpy as np
import tensorflow as tf

INDEX = "_index"


class DatasetIteratorTest(parameterized.TestCase, tf.test.TestCase):

  def _create_iterator(self, start_index: int, checkpoint: bool = True):
    """Create an iterator over some prime numbers with index."""
    primes = tf.constant([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    ds = tf.data.Dataset.range(start_index, 10)
    ds = ds.map(lambda i: {INDEX: i, "prime": primes[i]})
    # Remove index 1 and 3.
    ds = ds.filter(lambda x: tf.logical_and(x["prime"] != 3, x["prime"] != 7))
    ds = ds.batch(2, drop_remainder=True)
    return dataset_iterator.TfDatasetIterator(ds, checkpoint=checkpoint)

  def test_tf_iterator(self):
    it = self._create_iterator(0)
    self.assertEqual(
        it.element_spec, {
            INDEX: dataset_iterator.ArraySpec(np.int64, (2,)),
            "prime": dataset_iterator.ArraySpec(np.int32, (2,))
        })
    self.assertEqual(next(it), {INDEX: [0, 2], "prime": [2, 5]})
    self.assertEqual(next(it), {INDEX: [4, 5], "prime": [11, 13]})
    it.reset()
    # Iterator starts from the beginning.
    self.assertEqual(next(it), {INDEX: [0, 2], "prime": [2, 5]})

  def test_tf_iterator_save_and_load(self):
    it = self._create_iterator(0)
    next(it)
    next(it)
    next(it)
    work_dir = pathlib.Path(tempfile.mkdtemp())
    filename = work_dir / "ckpt"
    it.save(filename)
    self.assertTrue((work_dir / "ckpt.index").exists())

    it = self._create_iterator(0)
    # Iterator is at the beginning (batch 1).
    self.assertEqual(next(it), {INDEX: [0, 2], "prime": [2, 5]})
    it.load(filename)
    # Iterator is at the end (batch 4).
    self.assertEqual(next(it), {INDEX: [8, 9], "prime": [23, 29]})

  def test_tf_iterator_save_and_load_no_checkpoint(self):
    it = self._create_iterator(0, checkpoint=False)
    self.assertEqual(next(it), {INDEX: [0, 2], "prime": [2, 5]})
    self.assertEqual(next(it), {INDEX: [4, 5], "prime": [11, 13]})
    work_dir = pathlib.Path(tempfile.mkdtemp())
    filename = work_dir / "ckpt"
    it.save(filename)  # Should be a no-op and not create a checkpoint.
    self.assertFalse((work_dir / "ckpt.index").exists())

    it = self._create_iterator(0, checkpoint=False)
    self.assertEqual(next(it), {INDEX: [0, 2], "prime": [2, 5]})
    it.restore(filename)  # Should be a no-op, iterator just continues.
    self.assertEqual(next(it), {INDEX: [4, 5], "prime": [11, 13]})

  def test_peekable_dataset_iterator(self):
    it = self._create_iterator(0)
    it = dataset_iterator.PeekableDatasetIterator(it)
    self.assertEqual(it.peek(), {INDEX: [0, 2], "prime": [2, 5]})
    self.assertEqual(next(it), {INDEX: [0, 2], "prime": [2, 5]})
    self.assertEqual(next(it), {INDEX: [4, 5], "prime": [11, 13]})

  @parameterized.parameters(itertools.product([True, False], [True, False]))
  def test_peekable_dataset_iterator_async(self, wait: bool, peek_first: bool):
    it = self._create_iterator(0)
    it = dataset_iterator.PeekableDatasetIterator(it)
    future = it.peek_async()
    self.assertIsNone(it._peek)
    if wait:
      future.result()
      self.assertIsNotNone(it._peek)
    if peek_first:
      self.assertEqual(it.peek(), {INDEX: [0, 2], "prime": [2, 5]})
    self.assertEqual(next(it), {INDEX: [0, 2], "prime": [2, 5]})
    self.assertEqual(next(it), {INDEX: [4, 5], "prime": [11, 13]})


if __name__ == "__main__":
  tf.test.main()
