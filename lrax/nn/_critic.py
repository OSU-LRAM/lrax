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

from typing import Callable

import equinox as eqx
import jax.nn as jnn
from equinox.internal import doc_repr
from jaxtyping import Array, PRNGKeyArray

_identity = doc_repr(lambda x: x, "lambda x: x")
_elu = doc_repr(jnn.elu, "<function elu>")


class Critic(eqx.Module):
    """A state-value network.

    Estimates the expected return from a given observation, `V(obs)`. This is used as
    the advantage baseline in PPO and as the differentiable value target in the
    short-horizon actor-critic (SHAC) objective.
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        obs_size: int,
        width_size: int,
        depth: int,
        activation: Callable = _elu,
        final_activation: Callable = _identity,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a critic network.

        Parameters
        ----------
        - `obs_size`: The size of the observation vector. The input to the module
            should be a vector of shape `(obs_size,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
        - `activation`: The activation function after each hidden layer. Defaults to
            `jax.nn.elu`.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        self.mlp = eqx.nn.MLP(
            obs_size, "scalar", width_size, depth, activation, final_activation, key=key
        )

    def __call__(self, obs: Array) -> Array:
        """Forward pass of the critic network.

        Parameters
        ----------
        - `obs`: A JAX array with shape `(obs_size,)`.

        Returns
        -------
        A scalar JAX array, the estimated value of `obs`.
        """
        return self.mlp(obs)
