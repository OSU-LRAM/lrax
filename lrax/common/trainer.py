# Copyright 2026, Laboratory for Robotics and Applied Mechanics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, assert_never

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray, PyTree, ScalarLike
from lightning.pytorch import LightningDataModule
from tqdm import tqdm

from .._custom_types import Metrics, Optimizer
from .callbacks import Callback, StopTraining
from .logger import Logger

# takes either (model, batch, keys) or, when `filter_spec` is set, (diff_model,
# static_model, batch, keys); returns ((loss, metrics), grads)
type LossFn = Callable[..., tuple[tuple[ScalarLike, Metrics], PyTree]]


@dataclass
class Checkpoint:
    """Parameters used to checkpoint a model."""

    path: Path
    monitor: Optional[str] = None
    mode: Literal["min", "max", "latest"] = "latest"


@dataclass
class BaseTrainer:
    """Base class for trainers."""

    name: str
    logger: Optional[Logger] = None

    def log_hyperparams(self, params: dict[str, Any]):
        """Log a set of hyperparameters to the configured logger.

        Parameters
        ----------
        - `params`: A dictionary of hyperparameter names to values.

        Raises
        ------
        `ValueError` if no logger is configured.
        """
        if self.logger is None:
            raise ValueError("Cannot write parameters without a configured logger!")
        self.logger.log_hyperparams(params)


@dataclass
class ModelTrainer(BaseTrainer):
    """Trains an Equinox model with a standard supervised training loop."""

    def fit(
        self,
        key: PRNGKeyArray,
        model: eqx.Module,
        dm: LightningDataModule,
        loss_fn: LossFn,
        optim: Optimizer,
        *,
        epochs: int = 25,
        checkpoint: Optional[Checkpoint] = None,
        filter_spec: Optional[PyTree[bool]] = None,
        callbacks: Sequence[Callback] = (),
        is_multi_transform: bool = False,
    ):
        """Train `model` on the data provided by `dm`.

        Parameters
        ----------
        - `key`: A `jax.random.key` used to provide randomness for each training step.
        - `model`: The Equinox model to train.
        - `dm`: A `LightningDataModule` providing the training dataloader.
        - `loss_fn`: A callable that computes the loss and gradients for a batch. If
            `filter_spec` is set, it is called as `loss_fn(diff_model, static_model,
            batch, keys)`; otherwise as `loss_fn(model, batch, keys)`. In both cases it
            returns `((loss, metrics), grads)`.
        - `optim`: The optax optimizer used to update the model's parameters.
        - `epochs`: The number of epochs to train for. Defaults to `25`. (Keyword only
            argument.)
        - `checkpoint`: If set, configures periodic saving of the model to disk.
            Defaults to `None`. (Keyword only argument.)
        - `filter_spec`: If set, partitions the model into differentiable and static
            components, e.g., to freeze certain parameters during training. Defaults to
            `None`. (Keyword only argument.)
        - `callbacks`: A sequence of `Callback`s run after every training step, e.g., to
            trigger early stopping. Defaults to `()`. (Keyword only argument.)
        - `is_multi_transform`: Whether `optim` applies different optimizers to distinct
            parts of the model, requiring different `equinox` handling. Defaults to
            `False`. (Keyword only argument.)

        Returns
        -------
        The trained model.
        """
        # the multi-transform option allows us to apply different optimizers to
        # distinct parts of the model. equinox requires us to apply some different
        # steps depending on whether or not the multi-transform is being used
        if is_multi_transform:
            opt_state = optim.init(eqx.filter([model], eqx.is_inexact_array))
        else:
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def make_training_step(_model, _opt_state, _batch, _keys):
            # the filter spec is used to partition a model into differentiable and
            # static components. this is used when we want to freeze certain parameters
            # during training
            if filter_spec is not None:
                diff_model, static_model = eqx.partition(_model, filter_spec)
                result, grads = loss_fn(diff_model, static_model, _batch, _keys)
            else:
                result, grads = loss_fn(_model, _batch, _keys)

            loss, metrics = result

            if is_multi_transform:
                updates, _opt_state = optim.update(
                    eqx.filter([grads], eqx.is_inexact_array),
                    _opt_state,
                    eqx.filter([_model], eqx.is_inexact_array),
                )
                _model = eqx.apply_updates(_model, updates[0])  # type: ignore
            else:
                updates, _opt_state = optim.update(grads, _opt_state, params=_model)
                _model = eqx.apply_updates(_model, updates)

            return loss, metrics, _model, _opt_state

        if checkpoint is not None:
            checkpoint.path.parent.mkdir(parents=True, exist_ok=True)

        # setup the dataloader for training
        #
        # atm we only support training; we don't implement support for validation or
        # test datasets (I don't use them in my work)
        dm.setup("fit")
        loader = dm.train_dataloader()

        step, stop, best_metric = 0, False, None

        print(f"Training {self.name}...")
        for epoch in range(epochs):
            epoch_losses = []

            for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
                # split the key into one key per element of the batch; we need this
                # when training stochastic models that depend on a key for each rollout
                batch_size = jtu.tree_leaves(batch)[0].shape[0]
                keys = jr.split(jr.fold_in(key, step), batch_size)

                result = make_training_step(model, opt_state, batch, keys)
                loss, metrics, model, opt_state = result

                if self.logger is not None:
                    self.logger.log_metrics(metrics)

                # this gives users the option to watch metrics other than the loss
                # value during training. if you are training on a cluster, then this
                # probably isn't especially useful, but it's helpful for debugging.
                if checkpoint is not None and checkpoint.monitor is not None:
                    epoch_losses.append(metrics[checkpoint.monitor])
                else:
                    epoch_losses.append(loss)

                # finish up by running the callbacks. I used to use these for adapting
                # parameters in the model during training, but didn't find that to be
                # especially useful in practice. now they are mostly used for early
                # stopping
                try:
                    for callback in callbacks:
                        callback(loss, metrics, step)
                except StopTraining as e:
                    print(f"Early stop at step {step}: {e}")
                    stop = True

                # track the total *training steps* not the current step in the epoch
                step += 1

                if stop:
                    break

            epoch_loss = jnp.mean(jnp.array(epoch_losses))
            if checkpoint is not None:
                # epoch_loss already reflects the monitored quantity: it's the mean
                # of checkpoint.monitor if set, otherwise the mean training loss
                match checkpoint.mode:
                    case "latest":
                        should_save = True
                    case "min":
                        should_save = best_metric is None or epoch_loss < best_metric
                    case "max":
                        should_save = best_metric is None or epoch_loss > best_metric
                    case _:
                        assert_never(checkpoint.mode)

                if should_save:
                    best_metric = epoch_loss
                    eqx.tree_serialise_leaves(checkpoint.path, model)

            print(f"Epoch {epoch} | Training Loss: {epoch_loss}")

            # early stopping check
            if stop:
                break

        return model
