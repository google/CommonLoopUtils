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

"""Library for parsing a preprocessing spec.

A preprocessing spec is a list of preprocessing ops separated by '|' that can be
applied sequentially as a preprocessing function. The preprocessing ops are
provided as input and must implement the PreprocessOp protocol. While not
strictly required we also recommend annotating preprocess ops as dataclasses.

By convention the preprocessing function operates on dictionaries of features.
Each op can change the dictionary by modifying, adding or removing dictionary
entries. Dictionary entries should be tensors, keys should be strings.
(For common data types we recommend using the feature keys used in TFDS.)

Example spec: 'fn1|fn2(3)|fn3(keyword=5)'
This will construct the following preprocessing function:
def preprocess_fn(features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  features = fn1(features)
  features = fn2(features, 3)
  features = fn3(features, keyword=5)
  return features

See preprocess_spec_test.py for some simple examples.
"""

import ast
import inspect
import re
import sys
from typing import Dict, List, Sequence, Tuple, Type, Union

from absl import logging
import dataclasses
from flax import traverse_util
import jax.numpy as jnp
import tensorflow as tf
import typing_extensions

# Feature dictionary. Arbitrary nested dictionary with string keys and
# tf.Tensor as leaves.
Tensor = Union[tf.Tensor, tf.RaggedTensor, tf.SparseTensor]
Features = Dict[str, Union[Tensor, "Features"]]  # pytype: disable=not-supported-yet
# Feature name for the random seed for tf.random.stateless_* ops.
SEED_KEY = "seed"
# Regex that finds upper case characters.
_CAMEL_CASE_RGX = re.compile(r"(?<!^)(?=[A-Z])")


def _describe_features(features: Features) -> str:
  description = {}
  for k, v in features.items():
    if isinstance(v, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
      description[k] = f"{v.dtype.name}{list(v.shape)}"
    elif isinstance(v, dict):
      description[k] = _describe_features(v)
    else:
      raise ValueError(f"Unsupported type {type(v)} at feature {k}.")
  return str(description)


@typing_extensions.runtime_checkable
class PreprocessOp(typing_extensions.Protocol):
  """Interface for all preprocess ops.

  You don't have to inherit from this protocol. Your class only needs to provide
  the same function signature for __call__().
  While not strictly required we strongly recommend annotating the preprocess
  ops with `@dataclasses.dataclass`. This shortens the code and creates a nice
  __str__().
  get_all_ops() will only return dataclasses but all other methods work with
  any class implementing this protocol.
  """

  def __call__(self, features: Features) -> Features:
    """Apply preprocessing op."""
    ...


def get_all_ops(module_name: str) -> List[Tuple[str, Type[PreprocessOp]]]:
  """Helper to return all preprocess ops in a module.

  Modules that define processing ops can simply define:
  all_ops = lambda: process_spec.get_all_ops(__name__)
  all_ops() will then return a list with all dataclasses implementing the
  PreprocessOp protocol.

  Args:
    module_name: Name of the module. The module must already be imported.

  Returns:
    List of tuples of process ops. The first tuple element is the class name
    converted to snake case (MyAwesomeTransform => my_awesome_transform) and
    the second element is the class.
  """
  is_op = lambda x: dataclasses.is_dataclass(x) and issubclass(x, PreprocessOp)
  op_name = lambda n: _CAMEL_CASE_RGX.sub("_", n).lower()
  members = inspect.getmembers(sys.modules[module_name])
  return [(op_name(name), op) for name, op in members if is_op(op)]


def _jax_supported_tf_types():
  types = [
      x for _, x in inspect.getmembers(tf.dtypes)
      if isinstance(x, tf.dtypes.DType) and hasattr(jnp, x.name)
  ]
  # bool is called bool_ in jax and won't be found by the expression above.
  return types + [tf.bool]


@dataclasses.dataclass
class OnlyJaxTypes:
  """Removes all features which types are not supported by JAX.

  This filters dense tensors by dtype and removes sparse and ragged tensors.
  The latter don't have an equivalent in JAX.

  Attr:
    types: List of allowed types. Defaults to all TF types that can be have an
      equivalant in jax.numpy.
  """

  types: List[tf.dtypes.DType] = dataclasses.field(
      default_factory=_jax_supported_tf_types)

  def __call__(self, features: Features) -> Features:
    features = traverse_util.flatten_dict(features)
    for name in list(features):
      dtype = features[name].dtype
      if dtype not in self.types:
        del features[name]
        logging.warning(
            "Removing feature %r because dtype %s is not supported in JAX.",
            name, dtype)
      elif isinstance(features[name], tf.SparseTensor):
        del features[name]
        logging.warning(
            "Removing feature %r because sparse tensors are not "
            "supported in JAX.", name)
      elif isinstance(features[name], tf.RaggedTensor):
        del features[name]
        logging.warning(
            "Removing feature %r because ragged tensors are not support in"
            "JAX.", name)
    features = traverse_util.unflatten_dict(features)
    return features  # pytype: disable=bad-return-type


@dataclasses.dataclass
class PreprocessFn:
  """Chain of preprocessing ops combined to a single preprocessing function."""

  ops: Sequence[PreprocessOp]
  only_jax_types: bool

  def __call__(self, features: Features) -> Features:
    """Sequentially applies all `self.ops` and returns the result."""
    logging.info("Features before preprocessing: %s",
                 _describe_features(features))
    features = features.copy()
    for op in self.ops:
      features = op(features)
      logging.info("Features after op %s:\n%s", op,
                   _describe_features(features))
    logging.info("Features after preprocessing: %s",
                 _describe_features(features))
    if self.only_jax_types:
      features = OnlyJaxTypes()(features)
    return features


def _get_op_class(
    expr: List[ast.stmt],
    available_ops: Dict[str, Type[PreprocessOp]]) -> Type[PreprocessOp]:
  """Gets the process op fn from the given expression."""
  if isinstance(expr, ast.Call):
    fn_name = expr.func.id
  elif isinstance(expr, ast.Name):
    fn_name = expr.id
  else:
    raise ValueError(
        f"Could not parse function name from expression: {expr!r}.")
  if fn_name in available_ops:
    return available_ops[fn_name]
  raise ValueError(
      f"'{fn_name}' is not available (available ops: {list(available_ops)}).")


def parse_single_preprocess_op(
    spec: str, available_ops: Dict[str, Type[PreprocessOp]]) -> PreprocessOp:
  """Parsing the spec for a single preprocess op.

  The op can just be the method name or the method name followed by any
  arguments (both positional and keyword) to the method.
  See the test cases for some valid examples.

  Args:
    spec: String specifying a single processing operations.
    available_ops: Available preprocessing ops.

  Returns:
    The ProcessOp corresponding to the spec.
  """
  try:
    expr = ast.parse(spec, mode="eval").body  # pytype: disable=attribute-error
  except SyntaxError:
    raise ValueError(f"{spec!r} is not a valid preprocess op spec.")
  op_class = _get_op_class(expr, available_ops)

  # Simple case without arguments.
  if isinstance(expr, ast.Name):
    return op_class()

  assert isinstance(expr, ast.Call)
  args = [ast.literal_eval(arg) for arg in expr.args]
  kwargs = {kv.arg: ast.literal_eval(kv.value) for kv in expr.keywords}
  if not args:
    return op_class(**kwargs)

  # Translate positional arguments into keyword arguments.
  available_arg_names = [f.name for f in dataclasses.fields(op_class)]
  for i, arg in enumerate(args):
    name = available_arg_names[i]
    if name in kwargs:
      raise ValueError(
          f"Argument {name} to {op_class} given both as positional argument "
          f"(value: {arg}) and keyword argument (value: {kwargs[name]}).")
    kwargs[name] = arg

  return op_class(**kwargs)


def parse(spec: str,
          available_ops: List[Tuple[str, Type[PreprocessOp]]],
          *,
          only_jax_types: bool = True) -> PreprocessFn:
  """Parses a preprocess spec; a '|' separated list of preprocess ops."""
  available_ops = dict(available_ops)
  if not spec.strip():
    ops = []
  else:
    ops = [
        parse_single_preprocess_op(s, available_ops) for s in spec.split("|")
    ]
  return PreprocessFn(ops, only_jax_types=only_jax_types)
