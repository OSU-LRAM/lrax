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

from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.internal import doc_repr
from jaxtyping import Array, PRNGKeyArray

from ..common.policies import ContinuousCritic

_identity = doc_repr(lambda x: x, "lambda x: x")
_elu = doc_repr(jnn.elu, "<function elu>")


class Actor(eqx.Module):
    """A squashed, diagonal Gaussian policy network used by SAC."""

    mlp: eqx.nn.MLP
    act_size: int = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        width_size: int,
        depth: int,
        activation: Callable = _elu,
        final_activation: Callable = _identity,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a SAC actor network.

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
        - `log_std_min`: The minimum value the predicted log standard deviation is
            clipped to. Defaults to `-20.0`.
        - `log_std_max`: The maximum value the predicted log standard deviation is
            clipped to. Defaults to `2.0`.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        self.mlp = eqx.nn.MLP(
            obs_size,
            2 * act_size,
            width_size,
            depth,
            activation,
            final_activation,
            key=key,
        )
        self.act_size = act_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def _dist_params(self, obs: Array) -> tuple[Array, Array]:
        mean, log_std = jnp.split(self.mlp(obs), 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def __call__(self, obs: Array) -> Array:
        """The deterministic (mode) action for `obs`, used at evaluation time.

        Parameters
        ----------
        - `obs`: A JAX array with shape `(obs_size,)`.

        Returns
        -------
        The squashed mean action, a JAX array with shape `(act_size,)`.
        """
        mean, _ = self._dist_params(obs)
        return jnn.tanh(mean)

    def sample(self, obs: Array, *, key: PRNGKeyArray) -> tuple[Array, Array]:
        """Draw a reparameterized, squashed action sample and its log-probability.

        Parameters
        ----------
        - `obs`: A JAX array with shape `(obs_size,)`.
        - `key`: A `jax.random.key` used to provide randomness for sampling.
            (Keyword only argument.)

        Returns
        -------
        A `(action, log_prob)` tuple: `action` is a JAX array with shape `(act_size,)`
        squashed to `[-1, 1]`, and `log_prob` is the scalar log-probability of `action`
        under the (squashed) distribution conditioned on `obs`.
        """
        mean, log_std = self._dist_params(obs)
        std = jnp.exp(log_std)
        pre_tanh = mean + std * jr.normal(key, mean.shape)
        action = jnn.tanh(pre_tanh)

        # log-probability of a diagonal Gaussian sample, corrected for the change of
        # variables induced by the tanh squashing (see Appendix C of the SAC paper)
        var = jnp.exp(2.0 * log_std)
        gaussian_log_prob = jnp.sum(
            -((pre_tanh - mean) ** 2) / (2 * var)
            - log_std
            - jnp.log(jnp.sqrt(2 * jnp.pi)),
            axis=-1,
        )

        # use an epsilon value larger than the project constant; without this SAC will
        # end up being numerically unstable
        correction = jnp.sum(jnp.log(1.0 - action**2 + 1e-6), axis=-1)
        log_prob = gaussian_log_prob - correction

        return action, log_prob


class ActorCritic(eqx.Module):
    """An SAC policy, including the actor, (twin) critic, and learned entropy
    temperature."""

    actor: Actor
    critic: ContinuousCritic
    log_alpha: Array

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        width_size: int,
        depth: int,
        num_critics: int = 2,
        activation: Callable = _elu,
        final_activation: Callable = _identity,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_log_alpha: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a SAC policy.

        Parameters
        ----------
        - `obs_size`: The size of the observation vector.
        - `act_size`: The size of the action vector.
        - `width_size`: The size of each hidden layer, for both the actor and critic
            networks.
        - `depth`: The number of hidden layers, including the output layer, for both
            the actor and critic networks.
        - `num_critics`: The number of critic networks in the (twin) critic ensemble.
            Defaults to `2`.
        - `activation`: The activation function after each hidden layer. Defaults to
            `jax.nn.elu`.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `log_std_min`: The minimum value the actor's log standard deviation is
            clipped to. Defaults to `-20.0`.
        - `log_std_max`: The maximum value the actor's log standard deviation is
            clipped to. Defaults to `2.0`.
        - `init_log_alpha`: The initial value of the log entropy temperature. Defaults
            to `0.0`.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        actor_key, critic_key = jr.split(key)
        self.actor = Actor(
            obs_size,
            act_size,
            width_size,
            depth,
            activation,
            final_activation,
            log_std_min,
            log_std_max,
            key=actor_key,
        )
        self.critic = ContinuousCritic(
            obs_size,
            act_size,
            width_size,
            depth,
            activation,
            final_activation,
            num_critics,
            key=critic_key,
        )
        self.log_alpha = jnp.asarray(init_log_alpha)
