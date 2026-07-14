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

import abc

import equinox as eqx
from jaxtyping import PRNGKeyArray
from optax import OptState

from .._custom_types import Metrics, Optimizer
from .envs import AbstractEnv, EnvState
from .policies import ActorCritic


class AbstractAlgorithm(eqx.Module):
    """Abstract base class for an on-policy actor-critic training algorithm."""

    @abc.abstractmethod
    def step(
        self,
        model: ActorCritic,
        opt_state: OptState,
        optim: Optimizer,
        env: AbstractEnv,
        env_state: EnvState,
        key: PRNGKeyArray,
    ) -> tuple[ActorCritic, OptState, EnvState, Metrics]:
        """Run one training iteration.

        Parameters
        ----------
        - `model`: The current `ActorCritic` model.
        - `opt_state`: The optimizer state for `model`.
        - `optim`: The optax optimizer used to update `model`.
        - `env`: The (vectorized) environment to interact with.
        - `env_state`: The environment state to resume from.
        - `key`: A `jax.random.key` used to provide randomness for this iteration.

        Returns
        -------
        The updated model and optimizer state, the environment state to resume from on
        the next call, and a dictionary of scalar metrics for this iteration.
        """
        raise NotImplementedError
