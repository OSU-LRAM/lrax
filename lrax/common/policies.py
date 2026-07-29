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

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.internal import doc_repr
from jaxtyping import Array, PRNGKeyArray

_identity = doc_repr(lambda x: x, "lambda x: x")
_elu = doc_repr(jnn.elu, "<function elu>")


class Actor(eqx.Module):
    """A diagonal Gaussian policy network."""

    mlp: eqx.nn.MLP
    log_std: Array

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        width_size: int,
        depth: int,
        activation: Callable = _elu,
        final_activation: Callable = _identity,
        log_std_init: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize an actor network.

        Parameters
        ----------
        - `obs_size`: The size of the observation vector. The input to the module
            should be a JAX array of shape `(obs_size,)`.
        - `act_size`: The size of the action vector.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
        - `activation`: The activation function after each hidden layer. Defaults to
            `jax.nn.elu`.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `log_std_init`: The initial value of the (state-independent) log standard
            deviation, broadcast to shape `(act_size,)`. Defaults to `0.0`.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        self.mlp = eqx.nn.MLP(
            obs_size, act_size, width_size, depth, activation, final_activation, key=key
        )
        self.log_std = log_std_init * jnp.ones(act_size)

    def __call__(self, obs: Array) -> Array:
        """Forward pass of the actor network.

        Parameters
        ----------
        - `obs`: A JAX array with shape `(obs_size,)`.

        Returns
        -------
        The mean action, a JAX array with shape `(act_size,)`.
        """
        return self.mlp(obs)

    def sample(self, obs: Array, *, key: PRNGKeyArray) -> tuple[Array, Array]:
        """Draw a reparameterized action sample and its log-probability.

        Parameters
        ----------
        - `obs`: A JAX array with shape `(obs_size,)`.
        - `key`: A `jax.random.key` used to provide randomness for sampling.
            (Keyword only argument.)

        Returns
        -------
        A `(action, log_prob)` tuple: `action` is a JAX array with shape
        `(act_size,)`, and `log_prob` is the scalar log-probability of `action`
        under the distribution conditioned on `obs`.
        """
        mean = self(obs)
        std = jnp.exp(self.log_std)
        action = mean + std * jr.normal(key, mean.shape)
        return action, self.log_prob(obs, action)

    def log_prob(self, obs: Array, value: Array) -> Array:
        """The log-probability of `value` summed over the action dimensions.

        Parameters
        ----------
        - `obs`: A JAX array with shape `(obs_size,)`.
        - `value`: A JAX array with shape `(act_size,)`, e.g., an action returned by
            `sample`.

        Returns
        -------
        A scalar JAX array, the log-probability of `value` under the distribution
        conditioned on `obs`.
        """
        mean = self(obs)
        var = jnp.exp(2.0 * self.log_std)
        log_probs = (
            -((value - mean) ** 2) / (2 * var)
            - self.log_std
            - jnp.log(jnp.sqrt(2 * jnp.pi))
        )
        return jnp.sum(log_probs, axis=-1)

    def entropy(self) -> Array:
        """The entropy of the distribution summed over the action dimensions.

        Since the standard deviation is state-independent, this is the same for
        every observation.

        Returns
        -------
        A scalar JAX array.
        """
        return jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + self.log_std, axis=-1)


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


class ActorCritic(eqx.Module):
    """Bundles an `Actor` and a `Critic` into a single pytree.

    Algorithms such as PPO update both networks from one loss function, and the
    training step machinery in `lrax.trainers` operates on a single model pytree;
    `ActorCritic` gives them that single pytree to differentiate through.
    """

    actor: Actor
    critic: Critic


class ContinuousCritic(eqx.Module):
    """Critic network(s) used for SAC

    This is inspired by the `stable_baselines3` implementation linked below:
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L912
    """

    num_critics: int = eqx.field(static=True)
    q_networks: Sequence[eqx.nn.MLP]

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        width_size: int,
        depth: int,
        activation: Callable = _elu,
        final_activation: Callable = _identity,
        num_critics: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a "continuous" critic network.

        Parameters
        ----------
        - `obs_size`: The size of the observation vector.
        - `act_size`: The size of the action vector.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
        - `activation`: The activation function after each hidden layer. Defaults to
            `jax.nn.elu`.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `num_critics`: The number of critic networks to create.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        self.num_critics = num_critics

        q_networks = []
        for q in range(num_critics):
            subkey = jr.fold_in(key, q)
            mlp = eqx.nn.MLP(
                obs_size + act_size,
                "scalar",
                width_size,
                depth,
                activation,
                final_activation,
                key=subkey,
            )
            q_networks.append(mlp)

        self.q_networks = q_networks

    def __call__(self, obs: Array, actions: Array) -> tuple[Array, ...]:
        """Forward pass of the critic network(s).

        Parameters
        ----------
        - `obs`: A JAX array with shape `(obs_size,)`.
        - `actions`: A JAX array with shape `(act_size,)`.

        Returns
        -------
        A tuple of JAX arrays, representing estimates from the action-state value
        functions.
        """
        qvalue_input = jnp.concatenate([obs, actions], axis=-1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1(self, obs: Array, actions: Array) -> Array:
        """Forward pass of the *first* critic network.

        This is useful when you don't need estimates from all networks.

        Parameters
        ----------
        - `obs`: A JAX array with shape `(obs_size,)`.
        - `actions`: A JAX array with shape `(act_size,)`.

        Returns
        -------
        A JAX array representing the estimated value from first network in `q_networks`.
        """
        return self.q_networks[0](jnp.concatenate([obs, actions], axis=-1))
