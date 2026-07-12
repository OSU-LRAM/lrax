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
from typing import Any

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


class EnvState(eqx.Module):
    """The state of a vectorized environment."""

    pipeline_state: Any
    obs: Array
    reward: Array
    done: Array
    info: dict[str, Array]


class AbstractEnv(eqx.Module):
    """Abstract base class for a simulation environment."""

    obs_size: eqx.AbstractVar[int]
    act_size: eqx.AbstractVar[int]
    num_envs: eqx.AbstractVar[int]

    @abc.abstractmethod
    def reset(self, key: PRNGKeyArray) -> EnvState:
        """Reset all `num_envs` environments to an initial state.

        Parameters
        ----------
        - `key`: A `jax.random.key` used to provide randomness for the reset.

        Returns
        -------
        The initial `EnvState`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: EnvState, action: Array) -> EnvState:
        """Advance all `num_envs` environments by one step.

        Parameters
        ----------
        - `state`: The current environment state.
        - `action`: A JAX array of shape `(num_envs, act_size)`.

        Returns
        -------
        The next `EnvState`. Any environment whose episode ends this step should be
        auto-reset, with `done` reporting the pre-reset termination/truncation flag.
        """
        raise NotImplementedError
