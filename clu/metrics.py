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

"""Functional metric computation library.

This library defines a functional metric computation interface `Metric` that
relies on metrics accumulating intermediate values (in a possibly distributed
manner), and then computes the final metric value from these intermediate
values. Note that most metrics can be created via `Average.from_fun()`. See also
`CollectingMetric` that collects all model outputs with a given name and lends
itself to metric computation in Python.

Some common metrics, such as accuracy and loss average/standard deviation, and a
`Collection` with the same interface, are also provided.

The "model output" is a dictionary of values with unique keys that all have a
specific meaning (such as `loss`, `logits`, and `labels`) and every metric
depends on at least one such model output by name. These outputs are usually
expected to be instances of `jnp.array`.

Synopsis:

  from clu import metrics
  import flax
  import jax

  @flax.struct.dataclass  # required for jax.tree_*
  class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")
    loss_std: metrics.Std.from_output("loss")

  def eval_step(ms, model, variables, inputs, labels):
    loss, logits = get_loss_and_logits(model, variables, inputs, labels)
    return ms.merge(Metrics.gather_from_model_output(
        loss=loss, logits=logits, labels=labels))

  p_eval_step = jax.pmap(
      eval_step, axis_name="batch", static_broadcasted_argnums=0)

  def evaluate(model, p_variables, test_ds):
    ms = flax.jax_utils.replicate(Metrics.empty())
    for inputs, labels in test_ds:
      ms = p_eval_step(ms, model, p_variables, inputs, labels)
    return ms.unreplicate().compute()
"""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type

from absl import logging

from clu.internal import utils
import clu.values
import flax
import jax
import jax.numpy as jnp
import numpy as np

# TODO(b/200953513): Migrate away from logging imports (on module level)
#                    to logging the actual usage. See b/200953513.



def _assert_same_shape(a: jnp.array, b: jnp.array):
  """Raises a `ValueError` if shapes of `a` and `b` don't match."""
  if a.shape != b.shape:
    raise ValueError(f"Expected same shape: {a.shape} != {b.shape}")


class Metric:
  """Interface for computing metrics from intermediate values.

  Refer to `Collection` for computing multipel metrics at the same time.

  Synopsis:

    import jax.numpy as jnp
    import flax

    @flax.struct.dataclass
    class Average(Metric):
      total: jnp.array
      count: jnp.array

      @classmethod
      def from_model_output(cls, value: jnp.array, **_) -> Metric:
        return cls(total=value.sum(), count=np.prod(value.shape))

      def merge(self, other: Metric) -> Metric:
        return type(self)(
          total=self.total + other.total,
          count=self.count + other.count,
        )

      def compute(self):
        return self.total / self.count

    average = None
    for value in range(data):
      update = Average.from_model_output(value)
      average = update if average is None else average.merge(update)
    print(average.compute())
  """

  @classmethod
  def from_model_output(cls, *args, **kwargs) -> "Metric":
    """Creates a `Metric` from model outputs."""
    raise NotImplementedError("Must override from_model_output()")

  def merge(self, other: "Metric") -> "Metric":
    """Returns `Metric` that is the accumulation of `self` and `other`.

    Args:
      other: A `Metric` whose inermediate values should be accumulated onto the
        values of `self`. Note that in a distributed setting, `other` will
        typically be the output of a `jax.lax` parallel operator and thus have a
        dimension added to the dataclass returned by `.from_model_output()`.

    Returns:
      A new `Metric` that accumulates the value from both `self` and `other`.
    """
    raise NotImplementedError("Must override merge()")

  def merge_reduce(self, other: "Metric") -> "Metric":
    return self.merge(other)

  def compute(self) -> jnp.array:
    """Computes final metrics from intermediate values."""
    raise NotImplementedError("Must override compute()")

  @classmethod
  def empty(cls) -> "Metric":
    """Returns an empty instance (i.e. `.merge(Metric.empty())` is a no-op)."""
    raise NotImplementedError("Must override empty()")

  def compute_value(self) -> clu.values.Value:
    """Wraps compute() and returns a values.Value."""
    return clu.values.Scalar(self.compute())

  def reduce(self) -> "Metric":
    """Reduces the metric along it first axis by calling `reduce_merge()`.

    This function primary use case is to aggregate metrics collected across
    multiple devices, rather than "merging" metrics across multiple steps.

    In many cases these have the same semantics (such as `Average`), but
    in some such as LastValue's batch averaging, reduction across devices is
    averaging, while reduction across steps is taking the last value.

    See `Collection.reduce`, for usage patterns.

    Returns:
      reduced metric.
    """

    def reduce_step(reduced: Metric, metric: Metric) -> Tuple[Metric, None]:
      return reduced.merge_reduce(metric), None

    first = jax.tree_map(lambda x: x[0], self)
    remainder = jax.tree_map(lambda x: x[1:], self)
    # TODO(b/160868467) Verify this adds no significant computational cost.
    return jax.lax.scan(reduce_step, first, remainder)[0]

  @classmethod
  def from_fun(cls, fun: Callable):  # pylint: disable=g-bare-generic
    """Calls `cls.from_model_output` with the return value from `fun`.

    Returns a `Metric` derived from `cls` whose `.from_model_output` (1) calls
    `fun` with keyword arguments from `model_output` and (2) supplies the output
    of `fun` to `cls.from_model_output`.

    If the return value of `fun` is a `Mapping`, then it will be expanded to
    create keyword arguments for `cls.from_model_output`. Otherwise, the output
    of `fun` is supplied as a single argument to `cls.from_model_output`.

    Note that the model output "mask" will also be forwarded to the metric, but
    only if it has the same first dimension as the value returned by `fun` (or
    the first value in the `Mapping` returned by `fun`). This allows metrics
    created by this function to be used both with values that exist per-example,
    as well as with values that only exist per batch.

    NOTE: If `fun` returns a `Mapping` with key "mask", then this mask will
    override a "mask" key passed to `from_model_output`.  This allows
    `fun` to read custom mask fields from `model_output`.

    Example:

    ```
    def get_head1(head1_loss, head1_mask, **_):
      return dict(loss=head1_loss, mask=head1_mask)

    @flax.struct.dataclass
    class MultiHeadMetrics(metrics.Collection):
      head1_loss: metrics.Average.from_output("loss").from_fun(get_head1)
      ...

    ms = MultiHeadMetrics.single_from_model_output(
        head1_loss=..., head1_mask=..., ...)
    ```

    Args:
      fun: Function to be applied to model output.

    Returns:
      A `Metric` derived from `cls` that calls `.from_model_output()` with
      the output returned by `fun` when called with keyword arguments from
      `model_output`.
    """

    @flax.struct.dataclass
    class FromFun(cls):
      """Wrapper Metric class that collects output after applying `fun`."""

      @classmethod
      def from_model_output(cls, **model_output) -> Metric:
        mask = model_output.get("mask")
        output = fun(**model_output)
        if isinstance(output, Mapping) and "mask" in output:
          output = dict(output)
          # pop mask to avoid multiple arg error later.
          output_mask = output.pop("mask", None)
          if mask is not None:
            raise ValueError(
                "fun %s provided a mask, but a 'mask' field was already "
                "given in the model output" % (fun,))
          mask = output_mask
        # Ignore the mask if its first dimension doesn't match that of the
        # output of `fun`.
        if mask is not None:
          if isinstance(output, Mapping):
            first_output = next(iter(output.values()))
          else:
            first_output = output
          if (first_output.shape or [0])[0] != mask.shape[0]:
            logging.warning(
                "Ignoring mask for fun(**model output) because of shape "
                "mismatch: output.shape=%s vs. mask.shape=%s",
                first_output.shape, mask.shape)
            mask = None
        if isinstance(output, Mapping):
          return super().from_model_output(**output, mask=mask)
        else:
          return super().from_model_output(output, mask=mask)

    return FromFun

  @classmethod
  def from_output(cls, name: str):  # pylint: disable=g-bare-generic
    """Calls `cls.from_model_output` with model output named `name`.

    Synopsis:

      @flax.struct.dataclass
      class Metrics(Collection):
        loss: Average.from_output('loss')

    Note that the model output "mask" will also be forwarded to the metric, but
    only if it has the same first dimension as the model output specified by
    `name`. This allows to use metrics created by this function both with named
    outputs that exist per-example, as well as with model outputs that only
    exist per batch (as for example "loss" often does).

    Args:
      name: Name of the model output that should be passed as first argument to
        `cls.from_model_output()`.

    Returns:
      A `Metric` derived from `cls` that calls `.from_model_output()` with as
      a first argument the model output specified by `name`.
    """

    @flax.struct.dataclass
    class FromOutput(cls):
      """Wrapper Metric class that collects output named `name`."""

      @classmethod
      def from_model_output(cls, **model_output) -> Metric:
        output = jnp.array(model_output[name])
        mask = model_output.get("mask")
        if mask is not None and (output.shape or [0])[0] != mask.shape[0]:
          logging.warning(
              "Ignoring mask for model output '%s' because of shape mismatch: "
              "output.shape=%s vs. mask.shape=%s", name, output.shape,
              mask.shape)
          mask = None
        return super().from_model_output(output, mask=mask)

    return FromOutput


@flax.struct.dataclass
class CollectingMetric(Metric):
  """A special metric that collects model outputs.

  This metric can NOT be used inside JIT-compiled eval steps (like the pattern
  described in the pydoc of this module). Instead, you will need to call
  `.merge()` in the Python evaluation loop that calls the compiled evaluation
  step. Metric accumulation happens on the host memory. For an efficient use
  of this metric that interleaves JAX computation with Python execution, see the
  async snippet below.

  This metric transfers arrays to host memory (converting to `np.ndarray`) for
  later use in computations on CPU. The references to individual arrays are
  stored in tuples, and a final call to `.compute()` concatenates these arrays.
  If not needed, this final copy can be avoided by overriding `.compute()`.

  Note though that these metrics use much more memory and compute somewhat more
  slowly.

  Also note that `mask` output is not applied automatically. Rather it should
  be collected and used in the final computation from the collected data.

  Example to use compute average precision using `sklearn`:

    @flax.struct.dataclass
    class AveragePrecision(
        metrics.CollectingMetric.from_outputs(("labels", "logits"))):

      def compute(self):
        values = super().compute()
        return sklearn.metrics.average_precision_score(
            values["labels"], values["logits"][:, 1])

  Note that this metric causes a sync barrier when the data is transfered to
  the host. But this can be avoided by using `asynclib`:

    from clu import asynclib

    def evaluate(params):
      ms = MyCollection.empty()
      pool = asynclib.Pool()

      @pool
      def merge(update):
        nonlocal ms
        ms = ms.merge(update)

      for batch in eval_ds:
        merge(eval_step(params, batch))

      pool.join()
      return ms.compute()
  """

  values: Dict[str, Tuple[np.ndarray, ...]]

  @classmethod
  def empty(cls) -> "CollectingMetric":
    return cls(values={})

  def merge(self, other: "CollectingMetric") -> "CollectingMetric":
    values = {
        name: (*value, *other.values[name])
        for name, value in self.values.items()
    }
    if any(
        isinstance(vv, jax.core.Tracer) for v in values.values() for vv in v):  # pylint: disable=g-complex-comprehension
      raise RuntimeError(
          "Tracer detected! CollectingMetric cannot be JIT compiled.")
    if other.values and not self.values:
      return other
    if self.values and not other.values:
      return self
    return type(self)(jax.tree_map(np.asarray, values))

  def reduce(self) -> "CollectingMetric":
    # Note that this is usually called from inside a `pmap()` via
    # `Collection.gather_from_model_output()` so we concatenate using jnp.
    return type(self)(
        {name: jnp.concatenate(values) for name, values in self.values.items()})

  def compute(self) -> Dict[str, np.ndarray]:
    return {k: np.concatenate(v) for k, v in self.values.items()}

  @classmethod
  def from_outputs(cls, names: Sequence[str]):
    """Returns a metric class that collects all model outputs named `names`."""

    @flax.struct.dataclass
    class FromOutputs(cls):  # pylint:disable=missing-class-docstring

      @classmethod
      def from_model_output(cls, **model_output) -> Metric:

        def make_array(value):
          value = jnp.array(value)
          # Can't jnp.concatenate() scalars, promote to shape=(1,) in that case.
          return value[None] if value.ndim == 0 else value

        return cls({name: (make_array(model_output[name]),) for name in names})

    return FromOutputs


@flax.struct.dataclass
class _ReductionCounter(Metric):
  """Pseudo metric that keeps track of the total number of `.merge()`."""

  value: jnp.array

  @classmethod
  def empty(cls):
    return cls(value=jnp.array(1, jnp.int32))

  def merge(self, other: "_ReductionCounter") -> "_ReductionCounter":
    return _ReductionCounter(self.value + other.value)


def _check_reduction_counter_ndim(reduction_counter: _ReductionCounter):
  ndim = reduction_counter.value.ndim
  if ndim != 0:
    raise ValueError(
        f"Collection is still replicated (ndim={ndim}). Maybe you forgot to "
        f"call a flax.jax_utils.unreplicate() or a Collections.reduce()?")


@flax.struct.dataclass
class Collection:
  """Updates a collection of `Metric` from model outputs.

  Refer to the module documentation for a complete example.

  Synopsis:

    @flax.struct.dataclass
    class Metrics(Collection):
      accuracy: Accuracy

    metrics = None
    for inputs, labels in data:
      logits = model(inputs)
      update = Metrics.single_from_model_output(logits=logits, labels=labels)
      metrics = update if metrics is None else metrics.merge(update)
    print(metrics.compute())
  """

  _reduction_counter: _ReductionCounter

  @classmethod
  def create(cls, **metrics: Type[Metric]) -> Type["Collection"]:
    """Handy short-cut to define a `Collection` inline.

    Instead declaring a `Collection` dataclass:

      @flax.struct.dataclass
      class MyMetrics(metrics.Collection):
        accuracy: metrics.Accuracy

    You can use this function to generate it dynamically:

      MyMetrics = metrics.Collection.create(accuracy=metrics.Accuracy)

    To simulataneously create the class and initialize an instance use
    `Collection.create_collection` instead.

    Args:
      **metrics: Names and metric classes to use include in the collection.

    Returns:
      A subclass of Collection with fields defined by provided `metrics`.
    """
    return flax.struct.dataclass(
        type("_InlineCollection", (Collection,), {"__annotations__": metrics}))

  @classmethod
  def create_collection(cls, **metrics: Metric) -> "Collection":
    """Creates a custom collection object with fields metrics.

    This object will be an instance of custom subclass of `Collection` with
    all fields in **metric declared as appropriate dataset fields. For example:

       my_metrics = metrics.Collection.create_collection(
            accuracy=metrics.Accuracy(0, 0))

      is equivalent to:

        @flax.struct.dataclass
        class MyMetrics(metrics.Collection):
          accuracy: metrics.Accuracy
        my_metrics = MyMetrics(_ReductionCounter(jnp.array(1)),
                               accuracy=metric.Accuracy(0, 0))

    Args:
      **metrics: metrics to incroporate into this object.

    Returns:
      An instance of Collection initialized with provided `metrics`
    """
    collection_class = cls.create(**{k: type(v) for k, v in metrics.items()})
    counter = _ReductionCounter(jnp.array(1, dtype=jnp.int32))
    return collection_class(_reduction_counter=counter, **metrics)

  @classmethod
  def empty(cls) -> "Collection":
    return cls(
        _reduction_counter=_ReductionCounter(jnp.array(1, dtype=jnp.int32)),
        **{
            metric_name: metric.empty()
            for metric_name, metric in cls.__annotations__.items()
        })

  @classmethod
  def _from_model_output(cls, **kwargs) -> "Collection":
    """Creates a `Collection` from model outputs."""
    return cls(
        _reduction_counter=_ReductionCounter(jnp.array(1, dtype=jnp.int32)),
        **{
            metric_name: metric.from_model_output(**kwargs)
            for metric_name, metric in cls.__annotations__.items()
        })

  @classmethod
  def single_from_model_output(cls, **kwargs) -> "Collection":
    """Creates a `Collection` from model outputs.

    Note: This function should only be called when metrics are collected in a
    non-distributed setting (i.e. outside a `pmap()`).

    Args:
      **kwargs: Model outputs used by individual metrics.

    Returns:
      A metric collection from provided `kwargs` model outputs.
    """
    return cls._from_model_output(**kwargs)

  @classmethod
  def gather_from_model_output(cls,
                               axis_name="batch",
                               **kwargs) -> "Collection":
    """Creates a `Collection` from model outputs in a distributed setting.

    Args:
      axis_name: Name of the axis along which the values are to be gathered.
        Should be the same as the `axis_name` argument to the `pmap()`.
      **kwargs: Model outputs used by individual metrics.

    Returns:
      A metric collection from provided `kwargs` model outputs that contains
      metrics for all devices across all hosts.
    """
    return jax.lax.all_gather(
        cls._from_model_output(**kwargs), axis_name=axis_name).reduce()

  def merge(self, other: "Collection") -> "Collection":
    """Returns `Collection` that is the accumulation of `self` and `other`."""
    return type(self)(**{
        metric_name: metric.merge(getattr(other, metric_name))
        for metric_name, metric in vars(self).items()
    })

  def reduce(self) -> "Collection":
    """Reduces the collection by calling `Metric.reduce()` on each metric.

    The primary use case is to reduce collection that was gathered
    from  multiple devices into one collection: For instance inside pmap

    ```
      col = jax.lax.all_gather(col, axis_name='foo').reduce()
    ```
    or, if computed directly from model_outputs:

    ```
      col = col.merge(col.gather_from_model_output(**outputs)))
    ```

    will sync collections across all devices to create a replicated collection
    that include statistics from all devices.
    Outside pmap, this metric can then be safely unreplicated using for
    `collection.unreplicate()`.

    If `collection.unreplicate()` is called without gathering it will only
    contain the statistics from the first device, which is rarely a desired
    behavior.

    Returns:
      Reduced collection.
    """
    return type(self)(**{
        metric_name: metric.reduce()
        for metric_name, metric in vars(self).items()
    })

  def compute(self) -> Dict[str, jnp.array]:
    """Returns a dictionary mapping metric field name to `Metric.compute()`."""
    _check_reduction_counter_ndim(self._reduction_counter)
    return {
        metric_name: metric.compute()
        for metric_name, metric in vars(self).items()
        if metric_name != "_reduction_counter"
    }

  def compute_values(self) -> Dict[str, clu.values.Value]:
    """Computes metrics and returns them as clu.values.Value."""
    _check_reduction_counter_ndim(self._reduction_counter)
    return {
        metric_name: metric.compute_value()
        for metric_name, metric in vars(self).items()
        if metric_name != "_reduction_counter"
    }

  def unreplicate(self) -> "Collection":
    """Short-hand for `flax.jax_utils.unreplicate(self)`.

    The collection should be gathered and `reduce`d inside pmap,
    using `gather_from_model_output` or all_gather / reduce for this
    function to return correct values. See `Collection.reduce` for details.

    Returns:
      Unreplicated collection
    """
    return flax.jax_utils.unreplicate(self)


@flax.struct.dataclass
class LastValue(Metric):
  """Keeps the last average global batch value.

  This is useful to log values such as learning rate and losses during training.

  This class mirrors Average, because it needs to maintain total/count
  in cases when batch is distributed across multiple devices and need
  to be averaged later. However, we don't inherit from Average to
  maintain backward compatibility in case of isinstance(metric, Average)
  check.  For backward compatibility this class can be initialized using the
  keyword `LastValue(value=10)`  or  `total` and `count`.
  """
  total: jnp.array
  count: jnp.array

  def __init__(self, total: Optional[jnp.array] = None,
               count: Optional[jnp.array] = None,
               value: Optional[jnp.array] = None,
               ):
    """Constructor which supports keyword argument value as initializer.

    If  "value" is provided, then  "total" should *not* be provided.

    Args:
      total: Total value.
      count: Count of examples, 1 if not provided
      value: Value, if provided, will be assumed to be "count" of values.
    """
    count = count if count is not None else jnp.array(1, dtype=jnp.int32)
    if value is not None:
      if total is not None:
        raise ValueError("Only one of 'total' and 'value' should be None. "
                         f'Got {total}, {value}')
      total = value * count
    object.__setattr__(self, "total", total)
    object.__setattr__(self, "count", count)

  @classmethod
  def empty(cls):
    return cls(jnp.array(0, jnp.float32), count=jnp.array(0, jnp.int32))

  @classmethod
  def from_model_output(cls,
                        value: jnp.array,
                        mask: Optional[jnp.array] = None,
                        **_) -> Metric:
    if mask is None:
      mask = jnp.ones((value.shape or [()])[0])
    return cls(
        total=jnp.where(mask, value, jnp.zeros_like(value)).sum(),
        count=mask.sum().astype(jnp.int32),
    )

  def merge(self, other: "LastValue") -> "LastValue":
    _assert_same_shape(self.value, other.value)
    return other

  def merge_reduce(self, other: "LastValue") -> "LastValue":
    # We need to average during reduction
    _assert_same_shape(self.total, other.total)
    return type(self)(
        total=self.total + other.total,
        count=self.count + other.count,
    )

  @property
  def value(self) -> jnp.array:
    # Explicitly allow NaN division as it is part of normal computation here.
    with jax.debug_nans(False):
      return self.total / self.count

  def compute(self) -> Any:
    return self.value


@flax.struct.dataclass
class Average(Metric):
  """Computes the average of a scalar or a batch of tensors.

  Supports the following types of masks:

  - A one-dimensional mask with the same leading dimension as the scalars, or,
  - A multi-dimensional mask with the exact same dimensions as the scalars.
    This allows the use of per-example masks for examples in a batch, as well as
    per-target masks for targets for examples in a batch.

  The result is always a scalar.

  See also documentation of `Metric`.
  """

  total: jnp.array
  count: jnp.array

  @classmethod
  def empty(cls) -> Metric:
    return cls(total=jnp.array(0, jnp.float32), count=jnp.array(0, jnp.int32))

  @classmethod
  def from_model_output(cls,
                        values: jnp.array,
                        mask: Optional[jnp.array] = None,
                        **_) -> Metric:
    if values.ndim == 0:
      values = values[None]
    if mask is None:
      mask = jnp.ones_like(values)
    # Leading dimensions of mask and values must match.
    if mask.shape[0] != values.shape[0]:
      raise ValueError(
          f"Argument `mask` must have the same leading dimension as `values`. "
          f"Received mask of dimension {mask.shape} "
          f"and values of dimension {values.shape}.")
    # Broadcast mask to the same number of dimensions as values.
    if mask.ndim < values.ndim:
      mask = jnp.expand_dims(
          mask, axis=tuple(np.arange(mask.ndim, values.ndim)))
    mask = mask.astype(bool)
    utils.check_param(mask, dtype=bool, ndim=values.ndim)
    return cls(
        total=jnp.where(mask, values, jnp.zeros_like(values)).sum(),
        count=jnp.where(mask, jnp.ones_like(values, dtype=jnp.int32),
                        jnp.zeros_like(values, dtype=jnp.int32)).sum(),
    )

  def merge(self, other: "Average") -> "Average":
    _assert_same_shape(self.total, other.total)
    return type(self)(
        total=self.total + other.total,
        count=self.count + other.count,
    )

  def compute(self) -> Any:
    return self.total / self.count


@flax.struct.dataclass
class Std(Metric):
  """Computes the standard deviation of a scalar or a batch of scalars.

  See also documentation of `Metric`.
  """

  total: jnp.array
  sum_of_squares: jnp.array
  count: jnp.array

  @classmethod
  def empty(cls):
    return cls(
        total=jnp.array(0, jnp.float32),
        sum_of_squares=jnp.array(0, jnp.float32),
        count=jnp.array(0, jnp.int32))

  @classmethod
  def from_model_output(cls,
                        values: jnp.array,
                        mask: Optional[jnp.array] = None,
                        **_) -> Metric:
    if values.ndim == 0:
      values = values[None]
    utils.check_param(values, ndim=1)
    if mask is None:
      mask = jnp.ones(values.shape[0], dtype=jnp.int32)
    return cls(
        total=values.sum(),
        sum_of_squares=jnp.where(mask, values**2, jnp.zeros_like(values)).sum(),
        count=mask.sum(),
    )

  def merge(self, other: "Std") -> "Std":
    _assert_same_shape(self.total, other.total)
    return type(self)(
        total=self.total + other.total,
        sum_of_squares=self.sum_of_squares + other.sum_of_squares,
        count=self.count + other.count,
    )

  def compute(self) -> Any:
    # var(X) = 1/N \sum_i (x_i - mean)^2
    #        = 1/N \sum_i (x_i^2 - 2 x_i mean + mean^2)
    #        = 1/N ( \sum_i x_i^2 - 2 mean \sum_i x_i + N * mean^2 )
    #        = 1/N ( \sum_i x_i^2 - 2 mean N mean + N * mean^2 )
    #        = 1/N ( \sum_i x_i^2 - N * mean^2 )
    #        = \sum_i x_i^2 / N - mean^2
    mean = self.total / self.count
    variance = self.sum_of_squares / self.count - mean**2
    # Mathematically variance can never be negative but in reality we may run
    # into such issues due to numeric reasons.
    variance = jnp.clip(variance, a_min=0.0)
    return variance**.5


@flax.struct.dataclass
class Accuracy(Average):
  """Computes the accuracy from model outputs `logits` and `labels`.

  `labels` is expected to be of dtype=int32 and to have 0 <= ndim <= 2, and
  `logits` is expected to have ndim = labels.ndim + 1.

  See also documentation of `Metric`.
  """

  @classmethod
  def from_model_output(cls, *, logits: jnp.array, labels: jnp.array,
                        **kwargs) -> Metric:
    if logits.ndim != labels.ndim + 1 or labels.dtype != jnp.int32:
      raise ValueError(
          f"Expected labels.dtype==jnp.int32 and logits.ndim={logits.ndim}=="
          f"labels.ndim+1={labels.ndim + 1}")
    return super().from_model_output(
        values=(logits.argmax(axis=-1) == labels).astype(jnp.float32), **kwargs)
