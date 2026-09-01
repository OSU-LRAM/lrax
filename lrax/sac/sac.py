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

from typing import Literal, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import lax
from jaxtyping import Array, PRNGKeyArray, PyTree
from optax import OptState

from .._custom_types import Metrics, Optimizer
from ..common.algorithm import AbstractAlgorithm
from ..common.buffers import ReplayBuffer, Transition
from ..common.env import AbstractEnv, EnvState
from ..common.policies import ContinuousCritic
from .policies import ActorCritic

type _AlgState = tuple[ContinuousCritic, ReplayBuffer]


class Rollout(eqx.Module):
    """Container representing the transition state collected from the environment."""

    obs: Array
    actions: Array
    rewards: Array
    next_obs: Array
    dones: Array
    timeouts: Array
    metrics: Metrics


class SAC(AbstractAlgorithm):
    """Soft Actor-Critic.

    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
    Stochastic Actor", Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
    """

    buffer_size: int = 1_000_000
    num_steps: int = 1
    num_gradient_steps: int = 1
    batch_size: int = 256
    learning_starts: int = 1_000
    tau: float = 0.005
    gamma: float = 0.99
    entropy_coef: float | Literal["auto"] = "auto"
    target_entropy: float | None = None

    @override
    def init(
        self,
        model: ActorCritic,
        env: AbstractEnv,
        *,
        key: PRNGKeyArray,
    ) -> _AlgState:
        """Build the initial replay buffer and target critic.

        See `AbstractAlgorithm.init` for the parameter descriptions.
        """
        del key
        buffer = ReplayBuffer.empty(self.buffer_size, env.obs_size, env.act_size)
        return (model.critic, buffer)

    def _rollout(
        self,
        model: ActorCritic,
        env: AbstractEnv,
        env_state: EnvState,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Rollout, EnvState]:
        """Collect `num_steps` transitions from `env`, starting at `env_state`.

        Parameters
        ----------
        - `model`: The `ActorCritic` whose actor is used to sample actions.
        - `env`: The vectorized environment to interact with.
        - `env_state`: The environment state to resume from.
        - `key`: A `jax.random.key` used to provide randomness for action sampling.
            (Keyword only argument.)

        Returns
        -------
        A pytree of rollout transitions and the environment state to resume from on the
        next call.
        """
        step_keys = jr.split(key, self.num_steps)

        def _step(state: EnvState, step_key: PRNGKeyArray) -> tuple[EnvState, Rollout]:
            env_keys = jr.split(step_key, env.num_envs)
            sample_fn = lambda o, k: model.actor.sample(o, key=k)
            actions, _ = jax.vmap(sample_fn)(state.obs, env_keys)
            next_state = env.step(state, actions)
            rollout = Rollout(
                obs=state.obs,
                actions=actions,
                rewards=next_state.reward,
                next_obs=next_state.terminal_obs,
                dones=next_state.done,
                timeouts=next_state.done * (1.0 - next_state.terminated),
                metrics=next_state.aux,
            )
            return next_state, rollout

        final_state, rollouts = lax.scan(_step, env_state, step_keys)
        return rollouts, final_state

    def _loss(
        self,
        model: ActorCritic,
        critic_target: ContinuousCritic,
        batch: Transition,
        target_entropy: float,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, Metrics]:
        """Compute the combined actor, critic, and entropy-temperature loss.

        Parameters
        ----------
        - `model`: The `ActorCritic` being updated.
        - `critic_target`: The target critic used to build the regression target for
            the critic loss. Passed separately from `model` (rather than as part of
            it) so it never receives a gradient, on top of being Polyak-averaged
            rather than optimized.
        - `batch`: A `Transition` batch sampled from the replay buffer.
        - `target_entropy`: The target entropy used to adapt the entropy temperature,
            when `self.entropy_coef == "auto"`.
        - `key`: A `jax.random.key` used to provide randomness for resampling actions
            at `batch.obs` and `batch.next_obs`. (Keyword only argument.)

        Returns
        -------
        The scalar loss and a pytree of metrics used for logging/analytics.
        """
        actor_key, next_key = jr.split(key)
        batch_size = batch.obs.shape[0]

        if self.entropy_coef == "auto":
            alpha = jnp.exp(model.log_alpha)
        else:
            alpha = jnp.asarray(self.entropy_coef)
        alpha = cast(Array, alpha)
        alpha_detached = lax.stop_gradient(alpha)

        # calculate the critic loss
        next_keys = jr.split(next_key, batch_size)
        sample_fn = lambda o, k: model.actor.sample(o, key=k)
        next_actions, next_log_probs = jax.vmap(sample_fn)(batch.next_obs, next_keys)
        next_q_values = jax.vmap(critic_target)(batch.next_obs, next_actions)
        next_q_min = jnp.min(jnp.stack(next_q_values, axis=0), axis=0)
        target_q = lax.stop_gradient(
            batch.rewards
            + self.gamma
            * (1.0 - batch.dones)
            * (next_q_min - alpha_detached * next_log_probs)
        )
        current_q_values = jax.vmap(model.critic)(batch.obs, batch.actions)
        critic_loss = sum(jnp.mean((q - target_q) ** 2) for q in current_q_values)

        # calculate the actor loss
        actor_keys = jr.split(actor_key, batch_size)
        new_actions, log_probs = jax.vmap(sample_fn)(batch.obs, actor_keys)
        q_values_new = jax.vmap(model.critic)(batch.obs, new_actions)
        q_min_new = jnp.min(jnp.stack(q_values_new, axis=0), axis=0)
        actor_loss = jnp.mean(alpha_detached * log_probs - q_min_new)

        # calculate the entropy loss if using the adaptive entropy
        if self.entropy_coef == "auto":
            alpha_loss = -jnp.mean(
                model.log_alpha * lax.stop_gradient(log_probs + target_entropy)
            )
        else:
            alpha_loss = jnp.zeros(())

        loss = critic_loss + actor_loss + alpha_loss

        metrics = {
            "loss": loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha_detached,
            "entropy": -jnp.mean(log_probs),
        }

        return loss, cast(Metrics, metrics)

    def _update(
        self,
        model: ActorCritic,
        alg_state: _AlgState,
        opt_state: OptState,
        optim: Optimizer,
        target_entropy: float,
        *,
        key: PRNGKeyArray,
    ) -> tuple[ActorCritic, _AlgState, OptState, Metrics]:
        """Run `num_gradient_steps` off-policy updates, each on a fresh sampled batch.

        Parameters
        ----------
        - `model`: The `ActorCritic` to update.
        - `alg_state`: The `(critic_target, buffer)` algorithm state to update. Assumes
            `buffer` already holds at least `self.batch_size` transitions.
        - `opt_state`: The optimizer state for `model`.
        - `optim`: The optax optimizer used to update `model`.
        - `target_entropy`: The target entropy used to adapt the entropy temperature.
        - `key`: A `jax.random.key` used to provide randomness for sampling batches
            from the replay buffer and for action resampling. (Keyword only argument.)

        Returns
        -------
        The updated model, algorithm state, and optimizer state, and a dictionary of
        scalar metrics averaged over every gradient step. The target critic is
        Polyak-averaged towards the (online) critic after every gradient step.
        """
        model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
        state_params, state_static = eqx.partition(alg_state, eqx.is_array)
        loss_and_grad = eqx.filter_value_and_grad(self._loss, has_aux=True)

        def _grad_step(
            carry: tuple[PyTree, PyTree, OptState], step_key: PRNGKeyArray
        ) -> tuple[tuple[PyTree, PyTree, OptState], Metrics]:
            model_params, state_params, opt_state = carry
            model = eqx.combine(model_params, model_static)
            critic_target, buffer = eqx.combine(state_params, state_static)

            sample_key, loss_key = jr.split(step_key)
            batch = buffer.sample(self.batch_size, key=sample_key)
            (_, metrics), grads = loss_and_grad(
                model, critic_target, batch, target_entropy, key=loss_key
            )

            updates, opt_state = optim.update(grads, opt_state, params=model_params)
            model = eqx.apply_updates(model, updates)

            # Polyak-average the target critic towards the just-updated (online)
            # critic; only the array leaves are averaged, since `ContinuousCritic`
            # also carries non-array statics (e.g. the activation function)
            target_arrays, target_static = eqx.partition(
                critic_target, eqx.is_inexact_array
            )
            critic_arrays, _ = eqx.partition(model.critic, eqx.is_inexact_array)
            target_arrays = jtu.tree_map(
                lambda t, c: (1.0 - self.tau) * t + self.tau * c,
                target_arrays,
                critic_arrays,
            )
            critic_target = eqx.combine(target_arrays, target_static)

            model_params = eqx.filter(model, eqx.is_inexact_array)
            state_params = eqx.filter((critic_target, buffer), eqx.is_array)
            return (model_params, state_params, opt_state), metrics

        step_keys = jr.split(key, self.num_gradient_steps)
        (model_params, state_params, opt_state), metrics = lax.scan(
            _grad_step, (model_params, state_params, opt_state), step_keys
        )
        metrics = jtu.tree_map(jnp.mean, metrics)
        model = eqx.combine(model_params, model_static)
        alg_state = eqx.combine(state_params, state_static)
        return model, alg_state, opt_state, metrics

    @override
    def step(
        self,
        model: ActorCritic,
        alg_state: _AlgState,
        opt_state: OptState,
        optim: Optimizer,
        env: AbstractEnv,
        env_state: EnvState,
        key: PRNGKeyArray,
    ) -> tuple[ActorCritic, _AlgState, OptState, EnvState, Metrics]:
        """Run one SAC iteration.

        See `AbstractAlgorithm.step` for the parameter and return descriptions.
        """
        rollout_key, update_key = jr.split(key)
        rollouts, env_state = self._rollout(model, env, env_state, key=rollout_key)

        total_size = self.num_steps * env.num_envs
        flat = jtu.tree_map(lambda x: x.reshape(total_size, *x.shape[2:]), rollouts)
        critic_target, buffer = alg_state
        buffer = buffer.add(
            flat.obs,
            flat.actions,
            flat.rewards,
            flat.next_obs,
            flat.dones,
            flat.timeouts,
        )
        alg_state = (critic_target, buffer)

        target_entropy = (
            self.target_entropy
            if self.target_entropy is not None
            else -float(model.actor.act_size)
        )

        # trace the update function to extract the shape of the additional metrics
        metrics_shape = eqx.filter_eval_shape(
            self._update,
            model,
            alg_state,
            opt_state,
            optim,
            target_entropy,
            key=update_key,
        )[-1]

        model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
        state_params, state_static = eqx.partition(alg_state, eqx.is_array)

        def _do_update(
            model_params: PyTree,
            state_params: PyTree,
            opt_state: OptState,
            key: PRNGKeyArray,
        ) -> tuple[PyTree, PyTree, OptState, Metrics]:
            model = eqx.combine(model_params, model_static)
            state = eqx.combine(state_params, state_static)
            model, state, opt_state, metrics = self._update(
                model, state, opt_state, optim, target_entropy, key=key
            )
            return (
                eqx.filter(model, eqx.is_inexact_array),
                eqx.filter(state, eqx.is_array),
                opt_state,
                metrics,
            )

        def _skip_update(
            model_params: PyTree,
            state_params: PyTree,
            opt_state: OptState,
            key: PRNGKeyArray,
        ) -> tuple[PyTree, PyTree, OptState, Metrics]:
            del key
            zeros = jtu.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), metrics_shape)
            return model_params, state_params, opt_state, cast(Metrics, zeros)

        model_params, state_params, opt_state, metrics = lax.cond(
            buffer.size >= self.learning_starts,
            _do_update,
            _skip_update,
            model_params,
            state_params,
            opt_state,
            update_key,
        )
        model = eqx.combine(model_params, model_static)
        alg_state = eqx.combine(state_params, state_static)

        # merge the environment metrics with the loss metrics
        metrics = dict(metrics)
        metrics.update(jtu.tree_map(jnp.mean, flat.metrics))
        metrics["mean_reward"] = jnp.mean(flat.rewards)
        metrics["done_rate"] = jnp.mean(flat.dones)
        metrics["buffer_size"] = buffer.size

        return model, alg_state, opt_state, env_state, cast(Metrics, metrics)
