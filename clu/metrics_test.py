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

"""Tests for clu.metrics."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
from clu import asynclib
from clu import metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class CollectingMetricAccuracy(
    metrics.CollectingMetric.from_outputs(("logits", "labels"))):

  def compute(self):
    values = super().compute()
    logits = values["logits"]
    labels = values["labels"]
    assert logits.ndim == 2, logits.shape
    assert labels.ndim == 1, labels.shape
    return (logits.argmax(axis=-1) == labels).mean()


@flax.struct.dataclass
class Collection(metrics.Collection):
  train_accuracy: metrics.Accuracy
  learning_rate: metrics.LastValue.from_output("learning_rate")


@flax.struct.dataclass
class CollectionMixed(metrics.Collection):
  collecting_metric_accuracy: CollectingMetricAccuracy
  train_accuracy: metrics.Accuracy


class MetricsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Clear the trace counter
    chex.clear_trace_counter()

    # Two batches of model output.
    self.model_outputs = (
        dict(
            logits=jnp.array([[1., 0.], [0., 1.]]),
            labels=jnp.array([0, 0]),
            example_loss=jnp.array([0, 4.2]),
            learning_rate=0.02,
            loss=jnp.array(4.2),
        ),
        dict(
            logits=jnp.array([[1., 2.], [3., 4.]]),
            labels=jnp.array([1, 1]),
            example_loss=jnp.array([1.7, 0]),
            learning_rate=0.01,
            loss=jnp.array(1.7),
        ),
    )
    masks = (
        jnp.array([False, True]),
        jnp.array([True, False]),
    )
    self.model_outputs_masked = tuple(
        dict(mask=mask, **model_output)
        for mask, model_output in zip(masks, self.model_outputs))

    self.results = {
        "train_accuracy": 0.75,
        "learning_rate": 0.01,
    }
    self.results_masked = {
        "train_accuracy": 0.5,
        "learning_rate": 0.01,
    }
    # Stack all values. Can for example be used in a pmap().
    self.model_outputs_stacked = jax.tree_map(lambda *args: jnp.stack(args),
                                              *self.model_outputs)
    self.model_outputs_masked_stacked = jax.tree_map(
        lambda *args: jnp.stack(args), *self.model_outputs_masked)

  def make_compute_metric(self, metric_class, reduce, jit=True):
    """Returns a jitted function to compute metrics.

    Args:
      metric_class: Metric class to instantiate.
      reduce: If set to `True`.
      jit: Whether the returned function should be jitted.

    Returns:
      A function that takes `model_outputs` (list of dictionaries of values) as
      an input and returns the value from `metric.compute()`.
    """

    def compute_metric(model_outputs):
      if reduce:
        metric_list = [
            metric_class.from_model_output(**model_output)
            for model_output in model_outputs
        ]
        metric_stacked = jax.tree_map(lambda *args: jnp.stack(args),
                                      *metric_list)
        metric = metric_stacked.reduce()
      else:
        metric = metric_class.empty()
        for model_output in model_outputs:
          update = metric_class.from_model_output(**model_output)
          metric = metric.merge(update)
      return metric.compute()

    if jit:
      compute_metric = jax.jit(compute_metric)
    return compute_metric

  def test_metric_reduce(self):
    metric1 = metrics.LastValue.from_model_output(jnp.array([1, 2]))
    metric2 = metrics.LastValue.from_model_output(jnp.array([3, 4]))
    metric12 = jax.tree_map(lambda *args: jnp.stack(args), metric1, metric2)
    chex.assert_trees_all_equal(metric12.reduce().compute(), metric2.compute())

  def test_from_fun_with_single_output(self):

    def accuracy(*, logits, labels, **_):
      return (logits.argmax(axis=-1) == labels).astype(jnp.float32)

    chex.assert_trees_all_close(
        self.make_compute_metric(
            metrics.Average.from_fun(accuracy),
            reduce=False)(self.model_outputs), self.results["train_accuracy"])

    chex.assert_trees_all_close(
        self.make_compute_metric(
            metrics.Average.from_fun(accuracy),
            reduce=False)(self.model_outputs_masked),
        self.results_masked["train_accuracy"])

  def test_from_fun_with_mapping_output(self):

    # This tests .from_fun() with a function that returns a mapping. Accuracy
    # accepts logits and labels already, so this function just passes them
    # along. (This isn't needed in real code that uses Accuracy, just to test
    # `from_fun`.)
    def make_accuracy_args_map(*, logits, labels, **_):
      return dict(logits=logits, labels=labels)

    chex.assert_trees_all_close(
        self.make_compute_metric(
            metrics.Accuracy.from_fun(make_accuracy_args_map),
            reduce=False)(self.model_outputs), self.results["train_accuracy"])

    chex.assert_trees_all_close(
        self.make_compute_metric(
            metrics.Accuracy.from_fun(make_accuracy_args_map),
            reduce=False)(self.model_outputs_masked),
        self.results_masked["train_accuracy"])

  @parameterized.named_parameters(
      ("0d_values_no_mask", 1, None, 1.),
      ("1d_values_no_mask", [1, 2, 3], None, 2.),
      ("1d_values_1d_mask", [1, 2, 3], [True, True, False], 1.5),
      ("2d_values_no_mask", [[1, 2], [2, 3], [3, 4]], None, 2.5),
      ("2d_values_1d_mask", [[1, 2], [2, 3], [3, 4]], [False, True, True], 3.),
      ("2d_values_2d_mask", [[1, 2], [2, 3], [3, 4]],
       [[False, True], [True, True], [True, True]], 2.8),
      ("3d_values_no_mask", [[[1, 2], [2, 3]], [[2, 1], [3, 4]],
                             [[3, 1], [4, 1]]], None, 2.25),
      ("3d_values_1d_mask", [[[1, 2], [2, 3]], [[2, 1], [3, 4]],
                             [[3, 1], [4, 1]]], [False, True, True], 2.375),
  )
  def test_average_masked(self, values, mask, expected_result):
    values = jnp.asarray(values)
    if mask is not None:
      mask = jnp.asarray(mask)
    chex.assert_trees_all_close(
        metrics.Average.from_model_output(values, mask=mask).compute(),
        expected_result)
    chex.assert_trees_all_close(
        (metrics.Average
         .from_output("values", mask_name="my_mask")
         .from_model_output(values=values, my_mask=mask)
         .compute()),
        expected_result)

  @parameterized.named_parameters(
      ("Average", metrics.Average),
      ("Std", metrics.Std),
      ("LastValue", metrics.LastValue),
  )
  def test_merge_asserts_shape(self, metric_cls):
    metric1 = metric_cls.from_model_output(jnp.arange(3.))
    metric2 = jax.tree_map(lambda *args: jnp.stack(args), metric1, metric1)
    with self.assertRaisesRegex(ValueError, r"^Expected same shape"):
      metric1.merge(metric2)

  @parameterized.named_parameters(
      ("", False),
      ("_reduce", True),
  )
  def test_accuracy(self, reduce):
    chex.assert_trees_all_close(
        self.make_compute_metric(metrics.Accuracy, reduce)(self.model_outputs),
        self.results["train_accuracy"])

  def test_last_value_asserts_shape(self):
    metric1 = metrics.LastValue.from_model_output(jnp.arange(3.))
    metric2 = jax.tree_map(lambda *args: jnp.stack(args), metric1, metric1)
    with self.assertRaisesRegex(ValueError, r"^Expected same shape"):
      metric1.merge(metric2)

  @parameterized.named_parameters(
      ("", False),
      ("_reduce", True),
  )
  def test_loss_average(self, reduce):
    chex.assert_trees_all_close(
        self.make_compute_metric(metrics.Average.from_output("loss"),
                                 reduce)(self.model_outputs_masked),
        self.model_outputs_stacked["loss"].mean())
    chex.assert_trees_all_close(
        self.make_compute_metric(
            metrics.Average.from_output("example_loss"),
            reduce)(self.model_outputs_masked),
        self.model_outputs_stacked["loss"].mean())

  @parameterized.named_parameters(
      ("", False),
      ("_reduce", True),
  )
  def test_loss_std(self, reduce):
    chex.assert_trees_all_close(
        self.make_compute_metric(metrics.Std.from_output("loss"),
                                 reduce)(self.model_outputs_masked),
        self.model_outputs_stacked["loss"].std(),
        atol=1e-4)
    chex.assert_trees_all_close(
        self.make_compute_metric(
            metrics.Std.from_output("example_loss"),
            reduce)(self.model_outputs_masked),
        self.model_outputs_stacked["loss"].std(),
        atol=1e-4)

  def test_collection_create(self):
    collection = metrics.Collection.create(accuracy=metrics.Accuracy)
    chex.assert_trees_all_close(
        collection.single_from_model_output(
            logits=jnp.array([[-1., 1.], [1., -1.]]),
            labels=jnp.array([0, 0]),  # i.e. 1st incorrect, 2nd correct
        ).compute(),
        {"accuracy": 0.5})

  def test_collection_create_collection(self):
    collection = metrics.Collection.create_collection(
        accuracy=metrics.Accuracy.from_model_output(
            logits=jnp.array([[-1., 1.], [1., -1.]]),
            labels=jnp.array([0, 0])),  # i.e. 1st incorrect, 2nd correct)
        loss=metrics.Average.from_model_output(jnp.array([0, 1, 2])))
    chex.assert_trees_all_close(collection.compute(), {
        "accuracy": 0.5,
        "loss": 1
    })
    chex.assert_trees_all_close(
        {k: v.value for k, v in collection.compute_values().items()}, {
            "accuracy": 0.5,
            "loss": 1
        })

  @parameterized.named_parameters(
      ("", False),
      ("_masked", True),
  )
  def test_collection_single(self, masked):

    def compute_collection(model_outputs):
      collection = Collection.empty()
      for model_output in model_outputs:
        update = Collection.single_from_model_output(**model_output)
        collection = collection.merge(update)
      return collection.compute()

    chex.assert_trees_all_close(
        jax.jit(compute_collection)(
            self.model_outputs_masked if masked else self.model_outputs),
        self.results_masked if masked else self.results)

  @parameterized.named_parameters(
      ("", False),
      ("_masked", True),
  )
  @mock.patch("jax.lax.all_gather")
  def test_collection_gather(self, masked, all_gather_mock):

    collections = [
        Collection.single_from_model_output(**model_output)
        for model_output in (
            self.model_outputs_masked if masked else self.model_outputs)
    ]
    all_gather_mock.return_value = jax.tree_map(lambda *args: jnp.stack(args),
                                                *collections)

    def compute_collection(model_outputs):
      collection = Collection.gather_from_model_output(**model_outputs[0])
      return collection.compute()

    chex.assert_trees_all_close(
        jax.jit(compute_collection)(
            self.model_outputs_masked if masked else self.model_outputs),
        self.results_masked if masked else self.results)

  @parameterized.named_parameters(
      ("", False),
      ("_masked", True),
  )
  def test_collection_gather_pmap(self, masked):

    @functools.partial(jax.pmap, axis_name="batch")
    def compute_collection(model_outputs):
      return Collection.gather_from_model_output(**model_outputs)

    if jax.local_device_count() > 1:
      chex.assert_trees_all_close(
          compute_collection(
              self.model_outputs_masked_stacked if masked else self
              .model_outputs_stacked).unreplicate().compute(),
          self.results_masked if masked else self.results)

  def test_collection_asserts_replication(self):
    collections = [
        Collection.single_from_model_output(**model_output)
        for model_output in self.model_outputs
    ]
    collection = jax.tree_map(lambda *args: jnp.stack(args), *collections)
    with self.assertRaisesRegex(ValueError, r"^Collection is still replicated"):
      collection.compute()

  def test_collecting_metric(self):
    metric_class = metrics.CollectingMetric.from_outputs(("logits", "loss"))
    logits = np.concatenate(
        [model_output["logits"] for model_output in self.model_outputs])
    loss = np.array(
        [model_output["loss"] for model_output in self.model_outputs])
    result = self.make_compute_metric(
        metric_class, reduce=False, jit=False)(
            self.model_outputs)
    chex.assert_trees_all_close(result, {
        "logits": logits,
        "loss": loss,
    })

  def test_collecting_metric_reduce(self):
    metric_class = metrics.CollectingMetric.from_outputs(("value",))
    metric = jax.jit(metric_class.from_model_output)(value=jnp.ones([8, 2, 4]))
    reduced = metric.reduce()
    chex.assert_trees_all_close(reduced.compute(), {"value": np.ones([16, 4])})

  def test_collecting_metric_async(self):
    metric = CollectingMetricAccuracy.empty()
    pool = asynclib.Pool()

    @pool
    def merge(update):
      nonlocal metric
      metric = metric.merge(update)

    for model_output in self.model_outputs:
      merge(jax.jit(CollectingMetricAccuracy.from_model_output)(**model_output))
    pool.join()
    result = metric.compute()
    chex.assert_trees_all_close(result, 0.75)

  def test_collecting_metric_tracer(self):
    metric_class = metrics.CollectingMetric.from_outputs(("logits",))
    with self.assertRaisesRegex(RuntimeError, r"^Tracer detected!"):
      _ = self.make_compute_metric(
          metric_class, reduce=False, jit=True)(
              self.model_outputs)

  def test_collection_mixed_async(self):
    metric = CollectionMixed.empty()
    pool = asynclib.Pool()

    @pool
    def merge(update):
      nonlocal metric
      metric = metric.merge(update)

    for model_output in self.model_outputs:
      merge(jax.jit(CollectionMixed.single_from_model_output)(**model_output))
    pool.join()
    result = metric.compute()
    chex.assert_trees_all_close(result, {
        "collecting_metric_accuracy": 0.75,
        "train_accuracy": 0.75,
    })

  def test_metric_empty_types_doesnt_cause_retrace(self):

    @jax.jit
    @chex.assert_max_traces(n=1)
    def merge_collection(model_output, collection):
      update = Collection.single_from_model_output(**model_output)
      return collection.merge(update)

    # Metric will be initialized with a strong type
    # Can only use non-collecting metrics as the shape of collecting
    # metrics changes every iteration.
    collection = Collection.empty()
    for model_output in self.model_outputs:
      # The merged metric _should not_ have weak types
      # If it does have a weak type the second call will cause a re-trace
      collection = merge_collection(model_output, collection)


if __name__ == "__main__":
  absltest.main()
