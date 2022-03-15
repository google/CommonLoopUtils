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

  def eval_step(model, variables, inputs, labels):
    loss, logits = get_loss_and_logits(model, variables, inputs, labels)
    return Metrics.gather_from_model_output(
        loss=loss, logits=logits, labels=labels)

  p_eval_step = jax.pmap(
      eval_step, axis_name="batch", static_broadcasted_argnums=0)

  def evaluate(model, p_variables, test_ds):
    metrics = None
    for inputs, labels in test_ds:
      update = flax.jax_utils.unreplicate(
          p_eval_step(model, p_variables, inputs, labels))
      metrics = update if metrics is None else metrics.merge(update)
    return metrics.compute()
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

from absl import logging

from clu.internal import utils
import clu.values
import flax
import jax
import jax.numpy as jnp

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
        return cls(total=value.sum(), count=jnp.prod(value.shape))

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

  def compute(self) -> jnp.array:
    """Computes final metrics from intermediate values."""
    raise NotImplementedError("Must override compute()")

  def compute_value(self) -> clu.values.Value:
    """Wraps compute() and returns a values.Value."""
    return clu.values.Scalar(self.compute())

  def reduce(self) -> "Metric":
    """Reduces the metric along it first axis by calling `merge()`."""

    def reduce_step(reduced: Metric, metric: Metric) -> Tuple[Metric, None]:
      return reduced.merge(metric), None

    first = jax.tree_map(lambda x: x[0], self)
    remainder = jax.tree_map(lambda x: x[1:], self)
    # TODO(b/160868467) Verify this adds no significant computational cost.
    return jax.lax.scan(reduce_step, first, remainder)[0]

  @classmethod
  def from_fun(cls, fun: Callable):  # pylint: disable=g-bare-generic
    """Calls `cls.from_model_output` with the return value from `fun`.

    Note that the model output "mask" will also be forwarded to the metric, but
    only if it has the same first dimension as the value returned by `fun` when
    called with keyword arguments from `model_output`.
    This allows to use metrics created by this function both with values
    that exist per-example, as well as with values that only
    exist per batch.

    Args:
      fun: Function to be applied to model output.

    Returns:
      A `Metric` derived from `cls` that calls `.from_model_output()` with
      the first argument as the value returned by `fun`
      when called with keyword arguments from `model_output`.
    """

    @flax.struct.dataclass
    class FromFun(cls):
      """Wrapper Metric class that collects output after applying `fun`."""

      @classmethod
      def from_model_output(cls, **model_output) -> Metric:
        mask = model_output.get("mask")
        output = fun(**model_output)
        if mask is not None and (output.shape or [0])[0] != mask.shape[0]:
          logging.warning(
              "Ignoring mask for fun(**model output)"
              "because of shape mismatch: "
              "output.shape=%s vs. mask.shape=%s", output.shape, mask.shape)
          mask = None
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
              "output.shape=%s vs. mask.shape=%s", name,
              output.shape, mask.shape)
          mask = None
        return super().from_model_output(output, mask=mask)

    return FromOutput


@flax.struct.dataclass
class CollectingMetric(Metric):
  """A special metric that collects model outputs.

  This metric keeps all the values and when metrics are `.merge()`-ed or
  `.reduce()`-ed, their values are simply concatenated. Useful for implementing
  metric computation in Python (e.g. using some external libraries).

  Note though that these metrics use much more memory and compute somewhat more
  slowly.

  Also note that `mask` output is not applied automatically. Rather it should
  be collected and used in the final computation from the collected data.

  Example to use compute average precision using `sklearn`:

    @flax.struct.dataclass
    class AveragePrecision(
        metrics.CollectingMetric.from_outputs(("labels", "logits"))):

      def compute(self):
        return sklearn.metrics.average_precision_score(
            self.values["labels"], self.values["logits"][:, 1])
  """

  values: Dict[str, jnp.array]

  def merge(self, other: "CollectingMetric") -> "CollectingMetric":
    return type(self)({
        name: jnp.concatenate((value, other.values[name]))
        for name, value in self.values.items()
    })

  def reduce(self) -> "CollectingMetric":
    return type(self)(
        {name: jnp.concatenate(values) for name, values in self.values.items()})

  def compute(self) -> Dict[str, Tuple[jnp.array]]:
    return self.values

  @classmethod
  def from_outputs(cls, names: Sequence[str]):
    """Returns a metric class that collects all model outputs named `names`."""

    @flax.struct.dataclass
    class FromOutputs(cls):

      @classmethod
      def from_model_output(cls, **model_output) -> Metric:
        def make_array(value):
          value = jnp.array(value)
          # Can't jnp.concatenate() scalars, promote to shape=(1,) in that case.
          return value[None] if value.ndim == 0 else value
        return cls({name: make_array(model_output[name]) for name in names})

    return FromOutputs


@flax.struct.dataclass
class _ReductionCounter(Metric):
  """Pseudo metric that keeps track of the total number of `.merge()`."""

  value: jnp.array

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
    counter = _ReductionCounter(jnp.array(1))
    return collection_class(_reduction_counter=counter, **metrics)

  @classmethod
  def _from_model_output(cls, **kwargs) -> "Collection":
    """Creates a `Collection` from model outputs."""
    return cls(
        _reduction_counter=_ReductionCounter(jnp.array(1)),
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
    """Reduces the collection by calling `Metric.reduce()` on each metric."""
    return type(self)(**{
        metric_name: metric.reduce()
        for metric_name, metric in vars(self).items()
    })

  def compute(self) -> Dict[str, jnp.array]:
    """Computes metrics and returns them as Python numbers/lists."""
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
    """Short-hand for `flax.jax_utils.unreplicate(self)`."""
    return flax.jax_utils.unreplicate(self)


@flax.struct.dataclass
class LastValue(Metric):
  """Keeps the last value.

  See also documentation of `Metric`.
  """

  value: jnp.array

  @classmethod
  def from_model_output(cls,
                        value: jnp.array,
                        mask: Optional[jnp.array] = None,
                        **_) -> Metric:
    if mask is None:
      mask = jnp.ones((value.shape or [()])[0])
    return cls(jnp.where(mask, value, jnp.zeros_like(value)).sum() / mask.sum())

  def merge(self, other: "LastValue") -> "LastValue":
    _assert_same_shape(self.value, other.value)
    return other

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
          mask, axis=tuple(jnp.arange(mask.ndim, values.ndim)))
    mask = mask.astype(bool)
    utils.check_param(mask, dtype=bool, ndim=values.ndim)
    return cls(
        total=jnp.where(mask, values, jnp.zeros_like(values)).sum(),
        count=jnp.where(mask, jnp.ones_like(values),
                        jnp.zeros_like(values)).sum(),
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
  def from_model_output(cls,
                        values: jnp.array,
                        mask: Optional[jnp.array] = None,
                        **_) -> Metric:
    if values.ndim == 0:
      values = values[None]
    utils.check_param(values, ndim=1)
    if mask is None:
      mask = jnp.ones(values.shape[0])
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
