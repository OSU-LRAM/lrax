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

from typing import Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import lax
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, ScalarLike
from optax import OptState

from .._custom_types import Metrics, Optimizer
from .._epsilon import EPSILON
from ..env import Env, EnvState
from ..nn._actor_critic import ActorCritic
from ._algorithm import AbstractAlgorithm


class Rollout(eqx.Module):
    """Container representing the transition state."""

    obs: PyTree[ArrayLike]
    actions: PyTree[ArrayLike]
    log_probs: Array
    values: Array
    rewards: Array
    dones: Array


class Batch(eqx.Module):
    """Container representing the batch state."""

    obs: PyTree[ArrayLike]
    actions: PyTree[ArrayLike]
    log_probs: Array
    values: Array
    advantages: Array
    returns: Array


def _compute_gae(
    rollouts: Rollout,
    last_values: Array,
    gamma: ScalarLike,
    gae_lambda: ScalarLike,
) -> tuple[Array, Array]:
    """Calculate the advantages and returns using generalized advantage estimation.

    "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
    John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan and Pieter Abbeel

    Parameters
    ----------
    - `rollouts`: The rollout to compute advantages and returns for.
    - `last_values`: The critic's value estimate for the `env_state` reached after
        the rollout, used to bootstrap the last step's advantage.
    - `gamma`: The discount factor.
    - `gae_lambda`: The GAE trace-decay parameter.

    Returns
    -------
    A `(advantages, returns)` tuple, each with the same shape as `rollouts.rewards`.
    """
    next_values = jnp.concatenate([rollouts.values[1:], last_values[None]], axis=0)
    non_terminal = 1.0 - rollouts.dones
    deltas = rollouts.rewards + gamma * next_values * non_terminal - rollouts.values

    def _step(gae_next: Array, xs: tuple[Array, Array]) -> tuple[Array, Array]:
        delta, mask = xs
        gae = delta + gamma * gae_lambda * mask * gae_next
        return gae, gae

    advantages = jnp.zeros_like(last_values)
    _, advantages = lax.scan(_step, advantages, (deltas, non_terminal), reverse=True)
    returns = advantages + rollouts.values

    return advantages, returns


def _adapt_lr(
    current_lr: float,
    kl: float,
    desired_kl: float,
    lr_scale_factor: float,
    min_lr: float,
    max_lr: float,
) -> Array:
    """Adapt a learning rate to keep `kl` near `desired_kl`.

    This is based on the implementation used in `rsl_rl`:
    https://github.com/leggedrobotics/rsl_rl/

    Parameters
    ----------
    - `current_lr`: The learning rate before this update.
    - `kl`: The measured KL divergence.
    - `desired_kl`: The target KL divergence.
    - `lr_scale_factor`: The multiplicative factor used to grow or shrink the
        learning rate.
    - `min_lr`: The minimum allowed learning rate.
    - `max_lr`: The maximum allowed learning rate.

    Returns
    -------
    The adapted learning rate.
    """
    b1 = jnp.maximum(min_lr, current_lr / lr_scale_factor)
    b2 = jnp.minimum(max_lr, current_lr * lr_scale_factor)
    return jnp.where(
        kl > desired_kl * 2.0,
        b1,
        jnp.where((kl < desired_kl / 2.0) & (kl > 0.0), b2, current_lr),
    )


class PPO(AbstractAlgorithm):
    """Proximal Policy Optimization.

    "Proximal Policy Optimization Algorithms", John Schulman, Filip Wolski, Prafulla
    Dhariwal, Alec Radford, Oleg Klimov.

    Notes
    -----
    - If `desired_kl` is set, `optim` must be built with `optax.inject_hyperparams` so
        its learning rate can be adapted at runtime, e.g.
        `optax.inject_hyperparams(optax.adam)(learning_rate=3e-4)`.
    """

    num_steps: int = 24
    num_epochs: int = 5
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: Optional[float] = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    normalize_advantages: bool = False
    desired_kl: Optional[float] = None
    lr_scale_factor: float = 1.5
    min_lr: float = 1e-5
    max_lr: float = 1e-2

    def _rollout(
        self, model: ActorCritic, env: Env, env_state: EnvState, *, key: PRNGKeyArray
    ) -> tuple[Rollout, EnvState]:
        """Collect `num_steps` transitions from `env`, starting at `env_state`.

        Parameters
        ----------
        - `model`: The `ActorCritic` model used to sample actions and estimate
            values.
        - `env`: The vectorized environment to interact with.
        - `env_state`: The environment state to resume from.
        - `key`: A `jax.random.key` used to provide randomness for action sampling.
            (Keyword only argument.)

        Returns
        -------
        A `(rollouts, env_state)` tuple: the collected `Rollout`, with each field of
        shape `(num_steps, env.num_envs, ...)`, and the environment state to resume
        from on the next call.
        """
        step_keys = jr.split(key, self.num_steps)

        def _step(state: EnvState, step_key: PRNGKeyArray) -> tuple[EnvState, Rollout]:
            env_keys = jr.split(step_key, env.num_envs)
            sample_fn = lambda o, k: model.actor.sample(o, key=k)  # noqa
            actions, log_probs = jax.vmap(sample_fn)(state.obs, env_keys)
            values = jax.vmap(model.critic)(state.obs)
            next_state = env.step(state, actions)
            rollout = Rollout(
                obs=state.obs,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=next_state.reward,
                dones=next_state.done,
            )
            return next_state, rollout

        final_state, rollouts = lax.scan(_step, env_state, step_keys)
        return rollouts, final_state

    def _loss(self, model: ActorCritic, batch: Batch) -> tuple[Array, Metrics]:
        """Compute the clipped PPO surrogate loss for a batch of transitions.

        Parameters
        ----------
        - `model`: The `ActorCritic` model being updated.
        - `batch`: A `Batch` of transitions to compute the loss over.

        Returns
        -------
        A `(loss, metrics)` tuple: the scalar loss to differentiate, and a
        dictionary of scalar metrics for logging.
        """
        log_probs = jax.vmap(model.actor.log_prob)(batch.obs, batch.actions)
        values = jax.vmap(model.critic)(batch.obs)

        advantages = batch.advantages
        if self.normalize_advantages:
            advantages = (advantages - jnp.mean(advantages)) / (
                jnp.std(advantages) + EPSILON
            )

        # calculate the surrogate loss
        ratio = jnp.exp(log_probs - batch.log_probs)
        surr1 = ratio * advantages
        surr2 = (
            jnp.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        )
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # calculate the value function loss
        if self.value_clip_ratio is not None:
            value_clipped = batch.values + jnp.clip(
                values - batch.values, -self.value_clip_ratio, self.value_clip_ratio
            )
            value_loss = 0.5 * jnp.mean(
                jnp.maximum(
                    (values - batch.returns) ** 2, (value_clipped - batch.returns) ** 2
                )
            )
        else:
            value_loss = 0.5 * jnp.mean((values - batch.returns) ** 2)

        entropy = model.actor.entropy()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        clip_fraction = jnp.mean(
            (jnp.abs(ratio - 1.0) > self.clip_ratio).astype(jnp.float32)
        )

        # return the metrics for logging during training
        metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean(batch.log_probs - log_probs),
            "clip_fraction": clip_fraction,
        }

        return loss, cast(Metrics, metrics)

    def _update(
        self,
        model: ActorCritic,
        opt_state: OptState,
        optim: Optimizer,
        batch: Batch,
        *,
        key: PRNGKeyArray,
    ) -> tuple[ActorCritic, OptState, Metrics]:
        """Update `model` for `num_epochs` passes over `batch`, `num_minibatches` each.

        Parameters
        ----------
        - `model`: The `ActorCritic` model to update.
        - `opt_state`: The optimizer state for `model`.
        - `optim`: The optax optimizer used to update `model`. If `self.desired_kl`
            is set, this must be built with `optax.inject_hyperparams`.
        - `batch`: The transitions to train on, reshuffled and split into
            `num_minibatches` minibatches for each epoch.
        - `key`: A `jax.random.key` used to provide randomness for shuffling `batch`
            each epoch. (Keyword only argument.)

        Returns
        -------
        The updated model and optimizer state, and a dictionary of scalar metrics
        averaged over every epoch and minibatch.

        Raises
        ------
        `ValueError` if `self.desired_kl` is set but `optim` was not built with
        `optax.inject_hyperparams`.
        """
        if self.desired_kl is not None and not hasattr(opt_state, "hyperparams"):
            raise ValueError(
                "PPO.desired_kl requires an optimizer whose learning rate can be "
                "adapted at runtime, e.g. "
                "`optax.inject_hyperparams(optax.adam)(learning_rate=...)`."
            )

        params, static = eqx.partition(model, eqx.is_inexact_array)

        batch_size = jtu.tree_leaves(batch)[0].shape[0]
        minibatch_size = batch_size // self.num_minibatches

        loss_and_grad = eqx.filter_value_and_grad(self._loss, has_aux=True)

        def _minibatch_step(
            carry: tuple[PyTree, OptState], minibatch: Batch
        ) -> tuple[tuple[PyTree, OptState], Metrics]:
            params, opt_state = carry
            model = eqx.combine(params, static)
            (_, metrics), grads = loss_and_grad(model, minibatch)

            if self.desired_kl is not None:
                updated_lr = _adapt_lr(
                    opt_state.hyperparams["learning_rate"],  # type: ignore
                    cast(float, metrics["approx_kl"]),
                    self.desired_kl,
                    self.lr_scale_factor,
                    self.min_lr,
                    self.max_lr,
                )
                opt_state.hyperparams["learning_rate"] = updated_lr  # type: ignore

            updates, opt_state = optim.update(grads, opt_state, params=params)
            model = eqx.apply_updates(model, updates)
            params = eqx.filter(model, eqx.is_inexact_array)
            return (params, opt_state), metrics

        def _epoch_step(
            carry: tuple[PyTree, OptState], epoch_key: PRNGKeyArray
        ) -> tuple[tuple[PyTree, OptState], Metrics]:
            params, opt_state = carry
            perm = jr.permutation(epoch_key, batch_size)
            shuffled = jtu.tree_map(lambda x: x[perm], batch)
            minibatches = jtu.tree_map(
                lambda x: x.reshape(self.num_minibatches, minibatch_size, *x.shape[1:]),
                shuffled,
            )
            (params, opt_state), metrics = lax.scan(
                _minibatch_step, (params, opt_state), minibatches
            )
            return (params, opt_state), metrics

        epoch_keys = jr.split(key, self.num_epochs)
        (params, opt_state), metrics = lax.scan(
            _epoch_step, (params, opt_state), epoch_keys
        )
        metrics = jtu.tree_map(jnp.mean, metrics)
        model = eqx.combine(params, static)
        return model, opt_state, metrics

    def step(
        self,
        model: ActorCritic,
        opt_state: OptState,
        optim: Optimizer,
        env: Env,
        env_state: EnvState,
        key: PRNGKeyArray,
    ) -> tuple[ActorCritic, OptState, EnvState, Metrics]:
        """Run one PPO iteration: rollout, GAE, then a clipped-surrogate update.

        See `AbstractAlgorithm.step` for the parameter and return descriptions.

        Raises
        ------
        `ValueError` if `num_steps * env.num_envs` is not divisible by
        `num_minibatches`, or if `self.desired_kl` is set but `optim` was not built
        with `optax.inject_hyperparams`.
        """
        total_size = self.num_steps * env.num_envs
        if total_size % self.num_minibatches != 0:
            raise ValueError(
                f"num_steps * env.num_envs ({total_size}) must be divisible by "
                f"num_minibatches ({self.num_minibatches})."
            )

        rollout_key, update_key = jr.split(key)
        rollouts, env_state = self._rollout(model, env, env_state, key=rollout_key)

        last_values = jax.vmap(model.critic)(env_state.obs)
        advantages, returns = _compute_gae(
            rollouts, last_values, self.gamma, self.gae_lambda
        )

        batch = Batch(
            obs=rollouts.obs,
            actions=rollouts.actions,
            log_probs=rollouts.log_probs,
            values=rollouts.values,
            advantages=advantages,
            returns=returns,
        )
        batch = jtu.tree_map(lambda x: x.reshape(total_size, *x.shape[2:]), batch)

        model, opt_state, metrics = self._update(
            model, opt_state, optim, batch, key=update_key
        )
        metrics["mean_reward"] = jnp.mean(rollouts.rewards)
        metrics["done_rate"] = jnp.mean(rollouts.dones)

        return model, opt_state, env_state, metrics
