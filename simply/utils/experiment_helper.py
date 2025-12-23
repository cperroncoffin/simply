# Copyright 2024 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper class for experiments."""

import collections
from collections.abc import Sequence
import dataclasses
import functools
import json
import logging
from typing import Any, Mapping, Protocol

from clu import metric_writers
from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
from simply.utils import checkpoint_lib as ckpt_lib
from simply.utils import common
from simply.utils import pytree
import yaml


class MetricWriter(Protocol):
  """Protocol for metric writers."""

  def write_scalars(self, step: int, scalars: Mapping[str, Any]) -> None:
    ...

  def write_texts(self, step: int, texts: Mapping[str, str]) -> None:
    ...

  def flush(self) -> None:
    ...

  def close(self) -> None:
    ...


class WandbMetricWriter:
  """Weights & Biases metric writer.

  This class provides an interface compatible with clu.metric_writers
  for logging to Weights & Biases.
  """

  def __init__(
      self,
      project: str = '',
      entity: str = '',
      name: str = '',
      tags: Sequence[str] = (),
      config: Mapping[str, Any] | None = None,
      dir: str | None = None,
  ):
    """Initialize wandb run.

    Args:
      project: W&B project name. If empty, uses WANDB_PROJECT env var.
      entity: W&B entity (username or team). If empty, uses default.
      name: Run name. If empty, wandb generates one.
      tags: List of tags for the run.
      config: Configuration dict to log.
      dir: Directory to store wandb files.
    """
    try:
      import wandb
    except ImportError:
      raise ImportError(
          'wandb is required for WandbMetricWriter. '
          'Install it with: pip install wandb'
      )

    self._wandb = wandb

    init_kwargs = {}
    if project:
      init_kwargs['project'] = project
    if entity:
      init_kwargs['entity'] = entity
    if name:
      init_kwargs['name'] = name
    if tags:
      init_kwargs['tags'] = list(tags)
    if config:
      init_kwargs['config'] = dict(config)
    if dir:
      init_kwargs['dir'] = dir

    self._run = wandb.init(**init_kwargs)
    logging.info('Initialized wandb run: %s', self._run.name)

  def write_scalars(self, step: int, scalars: Mapping[str, Any]) -> None:
    """Log scalar metrics to wandb."""
    # Convert numpy arrays to Python scalars
    logged_scalars = {}
    for k, v in scalars.items():
      if isinstance(v, np.ndarray):
        v = v.item() if v.size == 1 else float(v.mean())
      logged_scalars[k] = v
    self._wandb.log(logged_scalars, step=step)

  def write_texts(self, step: int, texts: Mapping[str, str]) -> None:
    """Log text to wandb as a table or summary."""
    for key, text in texts.items():
      # Log as wandb summary for text data
      self._run.summary[key] = text

  def flush(self) -> None:
    """Flush is a no-op for wandb (it handles this automatically)."""
    pass

  def close(self) -> None:
    """Finish the wandb run."""
    self._wandb.finish()


def is_primary_process() -> bool:
  """Returns if the current process is the primary one."""
  return jax.process_index() == 0


def convert_to_scalar(x: Any) -> Any:
  """Convert x to a single Python scalar."""
  try:
    if np.size(x) == 1:
      # Use np.asanyarray and reshape to convert to 0-dimensional array.
      return np.asanyarray(x).reshape(())
    else:
      return None
  except TypeError:
    return None


def setup_work_unit() -> None:
  print('Setup work unit.')


@dataclasses.dataclass(frozen=True)
class ExperimentHelper:
  """A utility class that saves all the experiment related data."""

  experiment_dir: str
  ckpt_interval: int = 0  # 0 means no save
  ckpt_max_to_keep: int = 1
  ckpt_keep_period: int = 0  # 0 means no keep
  metric_log_interval: int = 0
  num_train_steps: int = 0
  log_additional_info: bool = False
  should_save_ckpt: bool = True
  # Metric writer configuration
  metric_writer_type: str = 'wandb'  # 'tensorboard' or 'wandb'
  wandb_project: str = ''
  wandb_entity: str = ''
  wandb_name: str = ''
  wandb_tags: tuple[str, ...] = ()
  wandb_config: Mapping[str, Any] | None = None

  @property
  def should_save_data(self) -> bool:
    return is_primary_process() and bool(self.experiment_dir)

  @property
  def ckpt_dir(self) -> str:
    return (epath.Path(self.experiment_dir) / 'checkpoints').as_posix()

  @property
  def ckpt_save_policy(self) -> ocp.checkpoint_managers.SaveDecisionPolicy:
    """Creates a checkpoint save policy."""
    policies = []
    if self.ckpt_interval > 0:
      policies.append(
          ocp.checkpoint_managers.FixedIntervalPolicy(
              interval=self.ckpt_interval,
          )
      )
    if self.num_train_steps >= 0:
      policies.append(
          ocp.checkpoint_managers.SpecificStepsPolicy(
              steps=[self.num_train_steps],
          )
      )
    return ocp.checkpoint_managers.AnySavePolicy(policies)

  @property
  def ckpt_preservation_policy(
      self,
  ) -> ocp.checkpoint_managers.PreservationPolicy:
    policies = [ocp.checkpoint_managers.LatestN(self.ckpt_max_to_keep)]
    if self.ckpt_keep_period:
      policies.append(
          ocp.checkpoint_managers.EveryNSteps(
              interval_steps=self.ckpt_keep_period,
          )
      )
    return ocp.checkpoint_managers.AnyPreservationPolicy(policies)

  @functools.cached_property
  def ckpt_mngr(self) -> ocp.CheckpointManager | None:
    """Creates a checkpoint manager."""
    if not (self.should_save_ckpt and self.experiment_dir):
      return None
    if (
        self.ckpt_keep_period
        and (self.ckpt_keep_period % self.ckpt_interval) != 0
    ):
      raise ValueError(
          f'{self.ckpt_keep_period=} must be a multiple of '
          f'{self.ckpt_interval=}. Otherwise, it does not preserve anything.'
      )
    options = ocp.CheckpointManagerOptions(
        save_decision_policy=self.ckpt_save_policy,
        preservation_policy=self.ckpt_preservation_policy,
        async_options=ocp.AsyncOptions(timeout_secs=360000),
    )
    return ocp.CheckpointManager(self.ckpt_dir, options=options)

  def __post_init__(self):
    if self.should_save_data:
      epath.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

  @property
  def metric_logdir(self) -> str:
    return (epath.Path(self.experiment_dir) / 'tb_log').as_posix()

  @functools.cached_property
  def metric_writer(self) -> MetricWriter | None:
    """Creates a metric writer based on metric_writer_type."""
    if not self.should_save_data:
      return None

    if self.metric_writer_type == 'wandb':
      return WandbMetricWriter(
          project=self.wandb_project,
          entity=self.wandb_entity,
          name=self.wandb_name,
          tags=self.wandb_tags,
          config=self.wandb_config,
          dir=self.experiment_dir,
      )
    else:
      # Default to tensorboard (clu metric_writers)
      metric_logdir = epath.Path(self.metric_logdir)
      metric_logdir.mkdir(parents=True, exist_ok=True)
      writer = metric_writers.create_default_writer(
          logdir=metric_logdir,
          just_logging=not self.should_save_data,
          asynchronous=True,
      )
      return writer

  @functools.cached_property
  def metrics_aggregator(self) -> 'MetricsAggregator':
    return MetricsAggregator(average_last_n_steps=self.metric_log_interval)

  def set_notes(self, notes: str) -> None:
    print(f'NOTES: {notes}')

  def write_record(self, record: Mapping[str, Any]):
    logging_record = True
    if logging_record:
      for k, v in record.items():
        logging.info('%s: %s', k, v)

  def save_config_info(self, config, sharding_config, model=None):
    """Save model and config information."""
    if model is not None:
      model_basic_jsons = json.dumps(
          pytree.dump(model, only_dump_basic=True), indent=2
      )
      model_full_jsons = json.dumps(
          pytree.dump(model, only_dump_basic=False), indent=2
      )
    else:
      model_basic_jsons = ''
      model_full_jsons = ''
    experiment_config_jsons = json.dumps(pytree.dump(config), indent=2)
    sharding_config_jsons = json.dumps(pytree.dump(sharding_config), indent=2)
    self.write_texts(
        step=0,
        texts={
            'experiment_config': f'```\n{experiment_config_jsons}\n```',
            'sharding_config': f'```\n{sharding_config_jsons}\n```',
            'model_full_jsons': f'```\n{model_full_jsons}\n```',
            'model_basic_jsons': f'```\n{model_basic_jsons}\n```',
        },
    )
    self.flush()

    if self.should_save_data:
      experiment_dir = epath.Path(self.experiment_dir)
      with (experiment_dir / 'experiment_config.json').open('w') as f:
        f.write(experiment_config_jsons)
      with (experiment_dir / 'model_basic.json').open('w') as f:
        f.write(model_basic_jsons)
      with (experiment_dir / 'model_full.json').open('w') as f:
        f.write(model_full_jsons)
      if sharding_config:
        with (experiment_dir / 'sharding_config.json').open('w') as f:
          f.write(sharding_config_jsons)

  def add_metric(self, metric_name: str, metric_value: np.typing.ArrayLike):
    self.metrics_aggregator.add(metric_name, metric_value)

  def get_aggregated_metrics(self):
    return self.metrics_aggregator.get_aggregated_metrics()

  def should_log_metrics(self, step):
    return step % self.metric_log_interval == 0 or step == (
        self.num_train_steps - 1
    )

  def should_log_additional_info(self, step):
    return self.log_additional_info and self.should_log_metrics(step)

  def write_scalars(self, step, scalars, filter_nonscalars=True):
    scalars = common.get_raw_arrays(scalars)
    if filter_nonscalars:
      filtered_scalars = {}
      for k, v in scalars.items():
        if (converted_v := convert_to_scalar(v)) is not None:
          filtered_scalars[k] = converted_v
        else:
          logging.warning('Skipping non-scalar metric: %s = %s', k, v)
      scalars = filtered_scalars
    if metric_writer := self.metric_writer:
      metric_writer.write_scalars(step, scalars)

  def write_texts(self, step, texts):
    if metric_writer := self.metric_writer:
      metric_writer.write_texts(step, texts)

  def flush(self):
    if metric_writer := self.metric_writer:
      metric_writer.flush()

  def save_state_info(self, state):
    """Save state information."""
    state = common.get_raw_arrays(state)
    params_shape = jax.tree_util.tree_map(
        lambda x: str(x.shape), state['params']
    )
    logging.info('params shape: %s', params_shape)
    params_sharding = jax.tree_util.tree_map(
        lambda x: str(x.sharding), state['params']
    )
    logging.info('params sharding: %s', params_sharding)
    num_params = sum(
        jax.tree.leaves(
            jax.tree_util.tree_map(lambda x: np.prod(x.shape), state['params'])
        )
    )
    logging.info('num_params: %s M', num_params / 1e6)

    param_info_map = jax.tree_util.tree_map(
        lambda x, y: f'{x} :: {y}', params_shape, params_sharding
    )
    param_info_text = yaml.dump(
        param_info_map, default_flow_style=False, sort_keys=False
    )
    self.write_texts(
        step=0,
        texts={
            'num_params': f'`{num_params}`',
            'param_info_text': f'```\n{param_info_text}\n```',
        },
    )
    self.flush()
    if self.should_save_data:
      experiment_dir = epath.Path(self.experiment_dir)
      with (experiment_dir / 'params_info.json').open('w') as f:
        f.write(
            json.dumps(
                {
                    'params_shape': params_shape,
                    'params_sharding': params_sharding,
                    'num_params': int(num_params),
                },
                indent=2,
            )
        )

  def save_ckpt(self, state, step, data=None):
    if self.ckpt_mngr:
      ckpt_lib.save_checkpoint(self.ckpt_mngr, state, step, data=data)
      logging.info('Saving checkpoint at step %s.', step)

  def close(self, final_result=None):
    """Closes the experiment helper and saves the final result."""
    # Ensure all the checkpoints are saved.
    if self.ckpt_mngr:
      self.ckpt_mngr.close()
    if self.metric_writer:
      self.metric_writer.close()
    if self.should_save_data and final_result:
      experiment_dir = epath.Path(self.experiment_dir)
      with (experiment_dir / 'final_result.json').open('w') as f:
        f.write(json.dumps(final_result, indent=2))


@dataclasses.dataclass(frozen=True)
class MetricsAggregator(object):
  """Metrics aggregator."""

  average_last_n_steps: int = 100

  def __post_init__(self):
    if self.average_last_n_steps <= 0:
      raise ValueError(f'{self.average_last_n_steps=} must be positive.')

  @functools.cached_property
  def metrics(self) -> Mapping[str, collections.deque[np.typing.ArrayLike]]:
    return collections.defaultdict(collections.deque[np.typing.ArrayLike])

  def add(self, name: str, value: np.typing.ArrayLike) -> None:
    """Adds a metric to the aggregator."""
    if np.size(value) > 1:
      # raise ValueError(f'Value {value} for metric {name} must be a scalar.')
      logging.warning(
          'Value %s for metric %s is not a scalar, ignored in metric'
          ' aggregation.', value, name
      )
      return
    if isinstance(value, np.ndarray):
      value = value.item()
    self.metrics[name].append(value)
    if len(self.metrics[name]) > self.average_last_n_steps:
      self.metrics[name].popleft()

  def reset(self) -> None:
    self.metrics = collections.defaultdict(collections.deque)

  def get_aggregated_metrics(self) -> Mapping[str, np.ndarray]:
    agg_metrics = {}
    for k, vlist in self.metrics.items():
      agg_metrics[k] = np.mean(vlist)
    return agg_metrics
