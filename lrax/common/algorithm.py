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
from jaxtyping import PRNGKeyArray, PyTree
from optax import OptState

from .._custom_types import Metrics, Optimizers
from .env import AbstractEnv, EnvState

type _AlgState = PyTree


class AbstractAlgorithm(eqx.Module, abc.ABC):
    """Abstract base class for an actor-critic training algorithm."""

    @abc.abstractmethod
    def init(
        self,
        model: PyTree,
        env: AbstractEnv,
        *,
        key: PRNGKeyArray,
    ) -> _AlgState:
        """Build the initial algorithm state.

        Parameters
        ----------
        - `model`: The initial model, as passed to `PolicyTrainer.learn`.
        - `env`: The (vectorized) environment `step` will interact with.
        - `key`: A `jax.random.key` used to provide randomness for initialisation.
            (Keyword only argument.)

        Returns
        -------
        The initial algorithm state, which should be used the first time `step` is
        called.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        model: PyTree,
        alg_state: _AlgState,
        opt_state: PyTree[OptState],
        optim: Optimizers,
        env: AbstractEnv,
        env_state: EnvState,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, _AlgState, PyTree[OptState], EnvState, Metrics]:
        """Run one training iteration.

        Parameters
        ----------
        - `model`: The current model.
        - `alg_state`: The current algorithm state, as returned by `init` or by the
            previous call to `step`.
        - `opt_state`: The optimizer state for `model`. Most algorithms use a single
            optimizer, but algorithms that optimize distinct parts of the model
            separately (e.g. `SHAC`) instead take a matching PyTree of optimizer
            states, e.g. `{"actor": ..., "critic": ...}`.
        - `optim`: The optax optimizer(s) used to update `model`; see `opt_state`.
        - `env`: The (vectorized) environment to interact with.
        - `env_state`: The environment state to resume from.
        - `key`: A `jax.random.key` used to provide randomness for this iteration.

        Returns
        -------
        The updated model, algorithm state, and optimizer state, the environment state
        to resume from on the next call, and a dictionary of scalar metrics for this
        iteration.
        """
        raise NotImplementedError
