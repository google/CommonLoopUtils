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

"""Helper function for creating and logging TF/JAX variable overviews."""

import dataclasses
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from absl import logging

import flax
import jax
import numpy as np
import tensorflow as tf


# TODO(b/200953513): Migrate away from logging imports (on module level)
#                    to logging the actual usage. See b/200953513.


ModuleOrVariables = Union[tf.Module, List[tf.Variable]]
ParamsContainer = Union[tf.Module, Dict[str, np.ndarray],
                        Mapping[str, Mapping[str, Any]]]


@dataclasses.dataclass
class ParamRow:
  name: str
  shape: Tuple[int]
  size: int


@dataclasses.dataclass
class ParamRowWithStats(ParamRow):
  mean: float
  std: float


def flatten_dict(input_dict: Dict[str, Any],
                 *,
                 prefix: str = "",
                 delimiter: str = "/") -> Dict[str, Any]:
  """Flattens the keys of a nested dictionary."""
  output_dict = {}
  for key, value in input_dict.items():
    nested_key = f"{prefix}{delimiter}{key}" if prefix else key
    if isinstance(value, (dict, flax.core.FrozenDict)):
      output_dict.update(
          flatten_dict(value, prefix=nested_key, delimiter=delimiter))
    else:
      output_dict[nested_key] = value
  return output_dict


def count_parameters(params: ParamsContainer) -> int:
  """Returns the count of variables for the module or parameter dictionary."""
  if isinstance(params, tf.Module):
    return sum(np.prod(v.shape) for v in params.trainable_variables)  # pytype: disable=attribute-error
  params = flatten_dict(params)
  return sum(np.prod(v.shape) for v in params.values())


def get_params(module: tf.Module) -> Tuple[List[str], List[np.ndarray]]:
  """Returns the trainable variables of a module as flattened dictionary."""
  assert isinstance(module, tf.Module), module
  variables = sorted(module.trainable_variables, key=lambda v: v.name)
  return [v.name for v in variables], [v.numpy() for v in variables]


def get_parameter_rows(
    params: ParamsContainer,
    *,
    include_stats: bool = False,
) -> List[Union[ParamRow, ParamRowWithStats]]:
  """Returns information about parameters as a list of dictionaries.

  Args:
    params: Dictionary with parameters as NumPy arrays. The dictionary can be
      nested. Alternatively a `tf.Module` can be provided, in which case the
      `trainable_variables` of the module will be used.
    include_stats: If True add columns with mean and std for each variable. Note
      that this can be considerably more compute intensive and cause a lot of
      memory to be transferred to the host (with `tf.Module`).

  Returns:
    A list of `ParamRow`, or `ParamRowWithStats`, depending on the passed value
    of `include_stats`.
  """
  if isinstance(params, tf.Module):
    names, values = get_params(params)
  else:
    assert isinstance(params, (dict, flax.core.FrozenDict))
    if params:
      params = flatten_dict(params)
      names, values = map(list, tuple(zip(*sorted(params.items()))))
    else:
      names, values = [], []

  def make_row(name, value):
    if include_stats:
      return ParamRowWithStats(
          name=name,
          shape=value.shape,
          size=int(np.prod(value.shape)),
          mean=float(value.mean()),
          std=float(value.std()),
      )
    else:
      return ParamRow(
          name=name, shape=value.shape, size=int(np.prod(value.shape)))

  return [make_row(name, value) for name, value in zip(names, values)]


def _default_table_value_formatter(value):
  """Formats ints with "," between thousands and floats to 3 digits."""
  if isinstance(value, bool):
    return str(value)
  elif isinstance(value, int):
    return "{:,}".format(value)
  elif isinstance(value, float):
    return "{:.3}".format(value)
  else:
    return str(value)


def make_table(
    rows: List[Any],
    *,
    column_names: Optional[Sequence[str]] = None,
    value_formatter: Callable[[Any], str] = _default_table_value_formatter,
    max_lines: Optional[int] = None,
) -> str:
  """Renders a list of rows to a table.

  Args:
    rows: List of dataclass instances of a single type (e.g. `ParamRow`).
    column_names: List of columns that that should be included in the output. If
      not provided, then the columns are taken from keys of the first row.
    value_formatter: Callable used to format cell values.
    max_lines: Don't render a table longer than this.

  Returns:
    A string representation of the table in the form:

    +---------+---------+
    | Col1    | Col2    |
    +---------+---------+
    | value11 | value12 |
    | value21 | value22 |
    +---------+---------+
  """

  if any(not dataclasses.is_dataclass(row) for row in rows):
    raise ValueError("Expected `rows` to be list of dataclasses")
  if len(set(map(type, rows))) > 1:
    raise ValueError("Expected elements of `rows` be of same type.")

  class Column:

    def __init__(self, name, values):
      self.name = name.capitalize()
      self.values = values
      self.width = max(len(v) for v in values + [name])

  if column_names is None:
    if not rows:
      return "(empty table)"
    column_names = [field.name for field in dataclasses.fields(rows[0])]

  columns = [
      Column(name, [value_formatter(getattr(row, name))
                    for row in rows])
      for name in column_names
  ]

  var_line_format = "|" + "".join(f" {{: <{c.width}s}} |" for c in columns)
  sep_line_format = var_line_format.replace(" ", "-").replace("|", "+")
  header = var_line_format.replace(">", "<").format(*[c.name for c in columns])
  separator = sep_line_format.format(*["" for c in columns])

  lines = [separator, header, separator]
  for i in range(len(rows)):
    if max_lines and len(lines) >= max_lines - 3:
      lines.append("[...]")
      break
    lines.append(var_line_format.format(*[c.values[i] for c in columns]))
  lines.append(separator)

  return "\n".join(lines)


def get_parameter_overview(params: ParamsContainer,
                           *,
                           include_stats: bool = True,
                           max_lines: Optional[int] = None) -> str:
  """Returns a string with variables names, their shapes, count.

  Args:
    params: Dictionary with parameters as NumPy arrays. The dictionary can be
      nested. Alternatively a `tf.Module` can be provided, in which case the
      `trainable_variables` of the module will be used.
    include_stats: If True, add columns with mean and std for each variable.
    max_lines: If not `None`, the maximum number of variables to include.

  Returns:
    A string with a table like in the example.

  +----------------+---------------+------------+
  | Name           | Shape         | Size       |
  +----------------+---------------+------------+
  | FC_1/weights:0 | (63612, 1024) | 65,138,688 |
  | FC_1/biases:0  |       (1024,) |      1,024 |
  | FC_2/weights:0 |    (1024, 32) |     32,768 |
  | FC_2/biases:0  |         (32,) |         32 |
  +----------------+---------------+------------+
  Total: 65,172,512
  """
  if include_stats and isinstance(params, (dict, flax.core.FrozenDict)):
    params = jax.tree_map(np.asarray, params)
  rows = get_parameter_rows(params, include_stats=include_stats)
  total_weights = count_parameters(params)
  RowType = ParamRowWithStats if include_stats else ParamRow
  # Pass in `column_names` to enable rendering empty tables.
  column_names = [field.name for field in dataclasses.fields(RowType)]
  table = make_table(rows, max_lines=max_lines, column_names=column_names)
  return table + f"\nTotal: {total_weights:,}"


def log_parameter_overview(params: ParamsContainer,
                           *,
                           include_stats: bool = True,
                           max_lines: Optional[int] = None,
                           msg: Optional[str] = None):
  """Writes a table with variables name and shapes to INFO log.

  See get_parameter_overview for details.

  Args:
    params: Dictionary with parameters as NumPy arrays. The dictionary can be
      nested. Alternatively a `tf.Module` can be provided, in which case the
      `trainable_variables` of the module will be used.
    include_stats: If True, add columns with mean and std for each variable.
    max_lines: If not `None`, the maximum number of variables to include.
    msg: Message to be logged before the overview.
  """
  table = get_parameter_overview(params, include_stats=include_stats,
                                 max_lines=max_lines)
  lines = [msg] if msg else []
  lines += table.split("\n")
  # The table can be too large to fit into one log entry.
  for i in range(0, len(lines), 80):
    logging.info("\n%s", "\n".join(lines[i:i + 80]))
