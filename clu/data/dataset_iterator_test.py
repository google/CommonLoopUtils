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

"""Tests for dataset_iterator."""
import pathlib
import tempfile

from clu.data import dataset_iterator
import numpy as np
import tensorflow as tf


class DatasetIteratorTest(tf.test.TestCase):

  def _create_iterator(self, start_index: int):
    """Create an iterator over some prime numbers with index."""
    primes = tf.constant([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    ds = tf.data.Dataset.range(start_index, 10)
    ds = ds.map(lambda i: {"index": i, "prime": primes[i]})
    # Remove index 1 and 3.
    ds = ds.filter(lambda x: tf.logical_and(x["prime"] != 3, x["prime"] != 7))
    ds = ds.batch(2, drop_remainder=True)
    return dataset_iterator.TfDatasetIterator(ds)

  def test_tf_iterator(self):
    it = self._create_iterator(0)
    self.assertEqual(
        it.element_spec, {
            "index": dataset_iterator.ArraySpec(np.int64, (2,)),
            "prime": dataset_iterator.ArraySpec(np.int32, (2,))
        })
    self.assertEqual(it.get_next(), {"index": [0, 2], "prime": [2, 5]})
    self.assertEqual(it.get_next(), {"index": [4, 5], "prime": [11, 13]})
    it.reset()
    # Iterator starts from the beginning.
    self.assertEqual(it.get_next(), {"index": [0, 2], "prime": [2, 5]})

  def test_tf_iterator_save_and_load(self):
    it = self._create_iterator(0)
    it.get_next()
    it.get_next()
    it.get_next()
    work_dir = pathlib.Path(tempfile.mkdtemp())
    filename = work_dir / "ckpt"
    it.save(filename)
    self.assertTrue((work_dir / "ckpt.index").exists())

    it = self._create_iterator(0)
    # Iterator is at the beginning (batch 1).
    self.assertEqual(it.get_next(), {"index": [0, 2], "prime": [2, 5]})
    it.load(filename)
    # Iterator is at the end (batch 4).
    self.assertEqual(it.get_next(), {"index": [8, 9], "prime": [23, 29]})

  def test_index_iterator(self):
    it = dataset_iterator.IndexBasedDatasetIterator(self._create_iterator)
    self.assertEqual(
        it.element_spec, {
            "index": dataset_iterator.ArraySpec(np.int64, (2,)),
            "prime": dataset_iterator.ArraySpec(np.int32, (2,))
        })
    self.assertEqual(it.get_next(), {"index": [0, 2], "prime": [2, 5]})
    self.assertEqual(it.get_next(), {"index": [4, 5], "prime": [11, 13]})
    it.reset()
    # Iterator starts from the beginning.
    self.assertEqual(it.get_next(), {"index": [0, 2], "prime": [2, 5]})

  def test_index_iterator_save_and_load(self):
    it = dataset_iterator.IndexBasedDatasetIterator(self._create_iterator)
    it.get_next()
    it.get_next()
    it.get_next()
    work_dir = pathlib.Path(tempfile.mkdtemp())
    filename = work_dir / "ckpt"
    it.save(filename)
    self.assertTrue(filename.exists())

    it = dataset_iterator.IndexBasedDatasetIterator(self._create_iterator)
    # Iterator is at the beginning (batch 1).
    self.assertEqual(it.get_next(), {"index": [0, 2], "prime": [2, 5]})
    it.load(filename)
    # Iterator is at the end (batch 4).
    self.assertEqual(it.get_next(), {"index": [8, 9], "prime": [23, 29]})


if __name__ == "__main__":
  tf.test.main()
