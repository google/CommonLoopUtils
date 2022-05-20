# Changelog

## v0.0.1-alpha.1

Initial PyPi Release

Current list of modules:

-   `clu.checkpoint`
-   `clu.deterministic_training`
-   `clu.metric_writers`
-   `clu.periodic_actions`
-   `clu.platform`
-   `clu.profiler`

## v0.0.1-alpha.2

-   Adds `metrics` module and some minor changes.

## v0.0.1a3

-   Added `metric_writers.TorchTensorboardWriter`

## v0.0.2

-   Added preprocess_spec.
-   Improvements to periodic_actions.

## v0.0.3

-   `metric_writers`: Lets `SummaryWriter` write nested dictionaries.
-   `internal`: Adds `async.Pool`.
-   `preprocess_spec`: Support nested dictionaries.
-   `profile`: Use JAX profiler APIs instead of TF profiler APIs.

## v0.0.4

`deterministic_data`

-   Support non-positive input value for pad_up_to_batches.
-   Support padding dataset when data dimension is unknown.
-   Support TFDS specs in get_read_instruction_for_host.
-   Allow changing drop_remainder for batching.
-   Add RemainderOptions in deterministic_data.

`metric_writers`

-   Support multiple writers in metric_writers.ensure_flushes.

`metrics`

-   Makes internal.flatten_dict() work with ConfigDicts.
-   Forwards mask model output to metrics created via `Metric.from_output()`.
-   Forwards mask model output to metrics created via `Metric.from_fun()`.
-   Added `Collections.unreplicate()`, `Collections.create()`.

`periodic_actions`

-   Formats long time strings in '{days}d{hours}h{mins}m' format.

`preprocess_spec`

-   Make feature description of features in PreprocessFn more compact.
-   Better type check in `preprocess_spec.get_all_ops()`.

Documentation:

-   Added `clu_synopsis.ipynb` Colab

## v0.0.5

-   Log error instead of failing when `profiler.start()` raises an exception.
-   Makes `periodic_actions.ProgressUpdate` show total number of steps.
-   Makes `AsyncWriter` non-blocking wrt JAX async computations.
-   Adds `clu_synopsis.ipynb` Colab as initial documentation.
-   Restore Checkpoint without providing the state
-   Makes `PreprocessFn` addable.
-   Allow n-dimensional arrays (and masks) to be passed to Metrics.Average().
-   Support slicing `PreprocessFn`.

## v0.0.6

-   Makes `deterministic_data` work with `tfds>4.4.0` and `tfds<=4.4.0`.

This will be the last release supporting Python 3.6.

## v0.0.7

-   Moves `clu.internal.asynclib` to `clu.asynclib`.
-   Adds methods for writing raw tensors and audio to `MetricWriter`.
-   Adds `clu.values` to annotate arrays with a modality.
-   Adds `clu.data.DatasetIterator` - a generic interface between input
    pipelines and training loops.
-   Fixes various issues with `clu.metrics`.

This will be the last release supporting Python 3.7.
