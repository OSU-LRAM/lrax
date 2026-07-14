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

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import jax.tree_util as jtu
import wandb

from .._custom_types import Metrics


class Logger(ABC):
    """Base class for loggers used to record training hyperparameters and metrics.

    A `Logger` is attached to a trainer (e.g., `ModelTrainer`) to persist run data
    somewhere durable, e.g., to disk or to a remote experiment tracker.
    """

    @abstractmethod
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Record a set of hyperparameters for the run.

        Parameters
        ----------
        - `params`: A dictionary of hyperparameter names to values.
        """
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: Metrics, step: Optional[int] = None) -> None:
        """Record a set of metrics for a single training step.

        Parameters
        ----------
        - `metrics`: A dictionary of metric names to scalar values.
        - `step`: The step the metrics correspond to. If `None`, the logger tracks the
            step count internally. Defaults to `None`.
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release any resources held by the logger.

        This is a no-op by default; subclasses that hold open resources, e.g., a file
        handle or a remote run, should override this to clean them up.
        """


def _next_version(root_dir: Path, name: str) -> int:
    if not root_dir.exists():
        return 0
    versions = [
        int(suffix)
        for p in root_dir.glob(f"{name}_*")
        if (suffix := p.name.removeprefix(f"{name}_")).isdigit()
    ]
    return max(versions, default=-1) + 1


class FileLogger(Logger):
    """Logs hyperparameters and metrics to CSV files in a versioned run directory."""

    def __init__(
        self,
        root_dir: str | Path = ".",
        name: str = "lrax",
        version: Optional[int] = None,
    ):
        """Create a new file logger.

        Parameters
        ----------
        - `root_dir`: The directory under which versioned run directories are created.
            Defaults to the current directory.
        - `name`: The prefix used for the run directory, e.g., `f"{name}_{version}"`.
            Defaults to `"lrax"`.
        - `version`: The version number of the run. If `None`, this is set to one more
            than the highest existing version under `root_dir`. Defaults to `None`.
        """
        root_dir = Path(root_dir)
        version = _next_version(root_dir, name) if version is None else version

        self.log_dir = root_dir / f"{name}_{version}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing logs to {self.log_dir}")

        self._metrics_file = open(self.log_dir / "metrics.csv", "w", newline="")
        self._metrics_writer: Optional[csv.DictWriter] = None
        self._step = 0

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        with open(self.log_dir / "hyperparams.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(params.keys()))
            writer.writeheader()
            writer.writerow(params)

    def log_metrics(self, metrics: Metrics, step: Optional[int] = None) -> None:
        row = {"step": self._step if step is None else step, **metrics}

        # the metric names logged on the first call fix the CSV's columns for the rest
        # of the run
        if self._metrics_writer is None:
            self._metrics_writer = csv.DictWriter(
                self._metrics_file, fieldnames=list(row.keys())
            )
            self._metrics_writer.writeheader()

        self._metrics_writer.writerow(row)
        self._metrics_file.flush()

        self._step += 1

    def cleanup(self) -> None:
        self._metrics_file.close()


class MultiLogger(Logger):
    """Wrapper for multiple loggers."""

    def __init__(self, loggers: Sequence[Logger]):
        """Create a new multi-logger.

        Parameters
        ----------
        - `loggers`: The loggers to wrap.
        """
        self.loggers = list(loggers)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        jtu.tree_map(lambda logger: logger.log_hyperparams(params), self.loggers)

    def log_metrics(self, metrics: Metrics, step: Optional[int] = None) -> None:
        jtu.tree_map(lambda logger: logger.log_metrics(metrics, step), self.loggers)

    def cleanup(self) -> None:
        jtu.tree_map(lambda logger: logger.cleanup(), self.loggers)


class WandbLogger(Logger):
    """Logs hyperparameters and metrics to a Weights & Biases run."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create a new Weights & Biases logger.

        Parameters
        ----------
        - `project`: The name of the W&B project to log to.
        - `name`: The name of the run. If `None`, W&B assigns a random name. Defaults
            to `None`.
        - `kwargs`: Additional keyword arguments forwarded to `wandb.init`.
        """
        self._run = wandb.init(project=project, name=name, **kwargs)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        self._run.config.update(params)

    def log_metrics(self, metrics: Metrics, step: Optional[int] = None) -> None:
        self._run.log(metrics, step=step)

    def cleanup(self) -> None:
        self._run.finish()
