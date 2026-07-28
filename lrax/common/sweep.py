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
from typing import Any, Literal

import wandb


@dataclass
class SweepConfig:
    """A Weights & Biases hyperparameter sweep search space.

    This only describes *what* to search over. Registering the sweep (getting a sweep
    ID) and launching agents to run it, e.g., as Slurm jobs, are separate steps — see
    `register_sweep` and `agent_command`.
    """

    method: Literal["grid", "random", "bayes"]
    metric_name: str
    parameters: dict[str, dict[str, Any]]
    metric_goal: Literal["minimize", "maximize"] = "minimize"
    name: str | None = None
    program: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render this configuration as the dictionary `wandb.sweep` expects."""
        config: dict[str, Any] = {
            "method": self.method,
            "metric": {"name": self.metric_name, "goal": self.metric_goal},
            "parameters": self.parameters,
        }
        if self.name is not None:
            config["name"] = self.name
        if self.program is not None:
            config["program"] = self.program
        return config


def register_sweep(config: SweepConfig, project: str, entity: str | None = None) -> str:
    """Register a sweep with Weights & Biases.

    Parameters
    ----------
    - `config`: The search space to sweep over.
    - `project`: The W&B project to register the sweep under.
    - `entity`: The W&B entity (user or team) that owns the project. Defaults to
        `None`, in which case W&B uses the logged-in user's default entity.

    Returns
    -------
    The ID of the registered sweep, to be passed to `agent_command`.
    """
    return wandb.sweep(config.to_dict(), project=project, entity=entity)


def agent_command(
    sweep_id: str,
    project: str | None = None,
    entity: str | None = None,
    count: int | None = None,
) -> list[str]:
    """Build the `wandb agent` command that runs trials for a registered sweep.

    Each invocation of this command polls the sweep for the next set of hyperparameter
    values and runs one trial; the trial's own `WandbLogger` picks those values up via
    `wandb.init` without any special handling. The returned command is meant to be used
    as a `lrax.launch.Job`'s `command`, so that Slurm can run one or more agents as
    independent jobs.

    Parameters
    ----------
    - `sweep_id`: The ID returned by `register_sweep`.
    - `project`: The W&B project the sweep belongs to. Only needed if `sweep_id` isn't
        already qualified as `entity/project/sweep_id`. Defaults to `None`.
    - `entity`: The W&B entity the sweep belongs to. Defaults to `None`.
    - `count`: The maximum number of trials this agent should run before exiting. If
        `None`, the agent runs until the sweep is exhausted or stopped. Defaults to
        `None`.

    Returns
    -------
    The `wandb agent` command as a list of arguments.
    """
    command = ["wandb", "agent"]
    if project is not None:
        command += ["--project", project]
    if entity is not None:
        command += ["--entity", entity]
    if count is not None:
        command += ["--count", str(count)]
    command.append(sweep_id)
    return command
