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

"""Interface for dataset iterators.

This module provides the DatasetIterator interface. This intention is that
several frameworks providing datasets can implement this interface without
knowing anything about the framework used for the model and the training loop.
Likewise can training loops assume to get an DatasetIterator object and do not
need to care about the specifics of the input pipelines.

This modules does not depend on TensorFlow. The interface is generic and users
don't have to use `tf.data` to construct a DatasetIterator. However, if they
use `tf.data` they can simply wrap their `tf.data.Dataset` object with
`TfDatasetIterator` to satisfy the interface.
"""
from __future__ import annotations

import abc
import dataclasses
import functools
import json
import typing
from typing import Any, Callable, Dict, Optional, Tuple, Union

from etils import epath
import jax.numpy as jnp  # Just for type checking.
import numpy as np

DType = np.dtype
# Sizes of dimensions, None means the dimension size is unknown.
Shape = Tuple[Optional[int], ...]

INDEX = "index"


@dataclasses.dataclass
class ArraySpec:
  """Describes an array via it's dtype and shape."""
  dtype: DType
  shape: Shape


# Elements are dictionaries with NumPy/JAX arrays.
Array = Union[np.ndarray, jnp.ndarray]
Element = Dict[str, Array]
ElementSpec = Dict[str, ArraySpec]


class DatasetIterator(abc.ABC):
  """Generic interface for iterating over a dataset.

  This does not support __getitem__ since it cannot be implemented efficiently
  for many datasets. However datasets should allow starting the iterator from
  an arbitrary position.

  The element_spec property helps consumers to validate the input without
  reading data. This is similar to `tf.data.Dataset.element_spec`.

  Subclasses may decided to not read/write checkpoints if their state is
  sufficiently tracked externally (e.g. input pipelines that can be correctly
  restarted from the step number).
  """

  @abc.abstractmethod
  def get_next(self) -> Element:
    """Returns the next element."""

  def __next__(self) -> Element:
    return self.get_next()

  @abc.abstractmethod
  def reset(self):
    """Resets the iterator back to the beginning."""

  @property
  @abc.abstractmethod
  def element_spec(self) -> ElementSpec:
    """Returns the spec elements."""

  def save(self, filename: epath.PathLike):
    """Saves the state of the iterator to a file.

    This should only handle this iterator - not iterators in other processes.

    Args:
      filename: Name of the checkpoint. The file must be created by the method.
    """
    raise NotImplementedError

  def load(self, filename: epath.PathLike):
    """Restores the iterator from a file (if available).

    This should only handle this iterator - not iterators in other processes.

    Args:
      filename: Name of the checkpoint. The method can assume that the file
        exists.
    """
    raise NotImplementedError


class TfDatasetIterator(DatasetIterator):
  """DatasetIterator for wrapping a `tf.data.Dataset`."""

  def __init__(self, dataset):
    try:
      if typing.TYPE_CHECKING:
        tf = Any
      else:
        import tensorflow as tf  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      raise RuntimeError("When using TfDatasetIterator your binary must "
                         "depend on //third_party/py/tensorflow.") from e
    self._tf = tf

    if not isinstance(dataset, tf.data.Dataset):
      raise ValueError("`dataset` must be an instance of `tf.data.Dataset` "
                       f"but got {type(dataset)}.")
    self._dataset = dataset
    assert self.element_spec  # Verify element spec.
    self.iterator = iter(dataset)
    self._ckpt = tf.train.Checkpoint(ds=self.iterator)

  def get_next(self) -> Element:
    return {k: np.asarray(v) for k, v in next(self.iterator).items()}

  def reset(self):
    self.iterator = iter(self._dataset)
    self._ckpt = self._tf.train.Checkpoint(ds=self.iterator)

  @functools.cached_property
  def element_spec(self) -> ElementSpec:
    element_spec = self._dataset.element_spec
    if not isinstance(element_spec, dict):
      raise ValueError("Dataset elements must be flat dictionaries but got "
                       f"{element_spec}.")
    invalid_features = [
        k for k, v in element_spec.items()
        if not isinstance(v, self._tf.TensorSpec)
    ]
    if invalid_features:
      raise ValueError(f"Features {invalid_features} are not tensors. Dataset "
                       "elements must be flat dictionaries of tensors.")
    return {
        k: ArraySpec(dtype=v.dtype.as_numpy_dtype, shape=tuple(v.shape))
        for k, v in element_spec.items()
    }

  def save(self, filename: epath.PathLike):
    self._ckpt.write(str(filename))

  def load(self, filename: epath.PathLike):
    self._ckpt.read(str(filename)).assert_consumed()


class IndexBasedDatasetIterator(DatasetIterator):
  """Checkpointable iterator that restores state based on the last seen index.

  This iterator enables deterministic input pipelines without materialising the
  dataset by keeping track of the last seen "index". See go/preemptable-tf-data.
  """

  def __init__(self, start_index_to_iterator: Callable[[int], DatasetIterator]):
    self._start_index_to_iterator = start_index_to_iterator
    self._iterator = self._start_index_to_iterator(0)

  def get_next(self) -> Element:
    element = self._iterator.get_next()
    self._last_seen_index = element[INDEX].max().item()
    return element

  def reset(self):
    self._last_seen_index = -1
    self._iterator = self._start_index_to_iterator(0)

  @property
  def element_spec(self) -> ElementSpec:
    return self._iterator.element_spec

  def save(self, filename: epath.PathLike):
    ckpt = {"last_seen_index": self._last_seen_index}
    epath.Path(filename).write_text(json.dumps(ckpt))

  def load(self, filename: epath.PathLike):
    ckpt = json.loads(epath.Path(filename).read_text())
    self._last_seen_index = ckpt["last_seen_index"]
    self._iterator = self._start_index_to_iterator(self._last_seen_index + 1)
