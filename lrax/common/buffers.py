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

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class Transition(eqx.Module):
    """A batch of `(obs, action, reward, next_obs, done)` transitions."""

    obs: Array
    actions: Array
    rewards: Array
    next_obs: Array
    dones: Array


class ReplayBuffer(eqx.Module):
    """A fixed-capacity circular buffer of transitions, used for off-policy training."""

    obs: Array
    actions: Array
    rewards: Array
    next_obs: Array
    dones: Array
    timeouts: Array
    ptr: Array
    size: Array
    capacity: int = eqx.field(static=True)
    handle_timeout_termination: bool = eqx.field(static=True, default=True)

    @classmethod
    def empty(cls, capacity: int, obs_size: int, act_size: int) -> "ReplayBuffer":
        """Create an empty buffer with room for `capacity` transitions.

        Parameters
        ----------
        - `capacity`: The maximum number of transitions the buffer can hold. Once full,
            new transitions overwrite the oldest ones.
        - `obs_size`: The size of the observation vector.
        - `act_size`: The size of the action vector.

        Returns
        -------
        An empty `ReplayBuffer`.
        """
        return cls(
            obs=jnp.zeros((capacity, obs_size)),
            actions=jnp.zeros((capacity, act_size)),
            rewards=jnp.zeros((capacity,)),
            next_obs=jnp.zeros((capacity, obs_size)),
            dones=jnp.zeros((capacity,)),
            timeouts=jnp.zeros((capacity,)),
            ptr=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
        )

    def add(
        self,
        obs: Array,
        actions: Array,
        rewards: Array,
        next_obs: Array,
        dones: Array,
        timeouts: Array,
    ) -> "ReplayBuffer":
        """Insert a batch of transitions, overwriting the oldest entries if full.

        Parameters
        ----------
        - `obs`: A JAX array with shape `(n, obs_size)`.
        - `actions`: A JAX array with shape `(n, act_size)`.
        - `rewards`: A JAX array with shape `(n,)`.
        - `next_obs`: A JAX array with shape `(n, obs_size)`.
        - `dones`: A JAX array with shape `(n,)`. Whether the episode ended this step,
            by termination or truncation (i.e. an `EnvState.done`, not `.terminated`).
        - `timeouts`: A JAX array with shape `(n,)`. Whether the episode ended this step
            specifically due to a timeout/truncation.

        Returns
        -------
        The updated `ReplayBuffer`.
        """
        n = obs.shape[0]
        idx = (self.ptr + jnp.arange(n)) % self.capacity
        return ReplayBuffer(
            obs=self.obs.at[idx].set(obs),
            actions=self.actions.at[idx].set(actions),
            rewards=self.rewards.at[idx].set(rewards),
            next_obs=self.next_obs.at[idx].set(next_obs),
            dones=self.dones.at[idx].set(dones),
            timeouts=self.timeouts.at[idx].set(timeouts),
            ptr=(self.ptr + n) % self.capacity,
            size=jnp.minimum(self.size + n, self.capacity),
            capacity=self.capacity,
            handle_timeout_termination=self.handle_timeout_termination,
        )

    def sample(self, batch_size: int, *, key: PRNGKeyArray) -> Transition:
        """Draw a batch of transitions, sampled uniformly with replacement.

        Parameters
        ----------
        - `batch_size`: The number of transitions to sample.
        - `key`: A `jax.random.key` used to provide randomness for sampling. (Keyword
            only argument.)

        Returns
        -------
        A `Transition` whose fields have a leading `batch_size` dimension.
        """
        idx = jr.randint(key, (batch_size,), 0, self.size)
        dones = self.dones[idx]
        if self.handle_timeout_termination:
            dones = dones * (1.0 - self.timeouts[idx])
        return Transition(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            dones=dones,
        )
