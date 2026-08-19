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
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from jax import lax
from jaxtyping import Array, PRNGKeyArray, PyTree, ScalarLike
from optax import OptState

from .._custom_types import Metrics, Optimizer
from .._epsilon import EPSILON
from ..common.algorithm import AbstractAlgorithm
from ..common.envs import AbstractEnv, DiffEnvState, EnvState
from ..common.normalizers import RunningMeanStd
from .policies import ActorCritic, Critic


class Rollout(eqx.Module):
    """Container representing the transition data collected for critic training."""

    obs: PyTree[Array]
    rewards: Array
    dones: Array
    next_values: Array
    metrics: Metrics


class CriticBatch(eqx.Module):
    """Container representing a minibatch of critic regression targets."""

    obs: PyTree[Array]
    target_values: Array


class _StepOutput(eqx.Module):
    """Output of an short-horizon rollout step."""

    obs: Array
    raw_obs: Array
    reward: Array
    raw_reward: Array
    done: Array
    raw_done: Array
    next_value: Array
    ret: Array
    metrics: Metrics


class _AlgState(eqx.Module):
    """The SHAC algorithm state."""

    critic_target: Critic
    obs_rms: RunningMeanStd | None
    ret_rms: RunningMeanStd | None
    ret: Array


def _compute_target_values(
    rollout: Rollout,
    gamma: ScalarLike,
    gae_lambda: ScalarLike,
    critic_method: Literal["one-step", "td-lambda"],
) -> Array:
    """Compute per-step regression targets for the critic.

    Parameters
    ----------
    - `rollout`: The `Rollout` to compute targets for. `rollout.dones` is expected to
        have its last step forced to `1.0`, cutting the `"td-lambda"` trace off at the
        end of the short horizon.
    - `gamma`: The discount factor.
    - `gae_lambda`: The TD(lambda) trace-decay parameter, used only when `critic_method`
        is `"td-lambda"`.
    - `critic_method`: `"one-step"` for a one-step TD target, or `"td-lambda"` for an
        exponentially-weighted average of n-step returns.

    Returns
    -------
    An array of target values, the same shape as `rollout.rewards`.
    """
    if critic_method == "one-step":
        return rollout.rewards + gamma * rollout.next_values

    def _step(carry: tuple[Array, Array, Array], xs: tuple[Array, Array, Array]):
        a, b, trace = carry
        done, next_value, reward = xs

        trace = trace * gae_lambda * (1.0 - done) + done
        a = (1.0 - done) * (
            gae_lambda * gamma * a
            + gamma * next_value
            + (1.0 - trace) / (1.0 - gae_lambda) * reward
        )
        b = gamma * (next_value * done + b * (1.0 - done)) + reward
        target = (1.0 - gae_lambda) * a + trace * b

        return (a, b, trace), target

    num_envs = rollout.rewards.shape[1]
    init = (jnp.zeros(num_envs), jnp.zeros(num_envs), jnp.ones(num_envs))
    _, targets = lax.scan(
        _step,
        init,
        (rollout.dones, rollout.next_values, rollout.rewards),
        reverse=True,
    )
    return targets


class SHAC(AbstractAlgorithm):
    """Short-Horizon Actor-Critic for training policies in differentiable simulators.

    "Accelerated Policy Learning with Parallel Differentiable Simulation", Jie Xu,
    Viktor Makoviychuk, Yashraj Narang, Fabio Ramos, Wojciech Matusik, Animesh Garg,
    Miles Macklin.
    """

    num_steps: int = 32
    num_critic_epochs: int = 16
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    target_alpha: float = 0.4
    critic_method: Literal["one-step", "td-lambda"] = "td-lambda"
    reward_scale: float = 1.0
    normalize_obs: bool = False
    normalize_returns: bool = False
    squash_actions: bool = False

    @override
    def init(
        self,
        model: ActorCritic,
        env: AbstractEnv,
        *,
        key: PRNGKeyArray,
    ) -> _AlgState:
        """Build the initial target critic and (optional) running statistics.

        See `AbstractAlgorithm.init` for the parameter descriptions.

        Raises
        ------
        `ValueError` if `num_steps * env.num_envs` is not divisible by
        `num_minibatches`. `TypeError` if `env.reset` does not return a `DiffEnvState`.
        """
        total_size = self.num_steps * env.num_envs
        if total_size % self.num_minibatches != 0:
            raise ValueError(
                f"num_steps * env.num_envs ({total_size}) must be divisible by "
                f"num_minibatches ({self.num_minibatches})."
            )

        if not isinstance(eqx.filter_eval_shape(env.reset, key), DiffEnvState):
            raise TypeError("Unsupported environment: SHAC requires a `DiffEnvState`.")

        obs_rms = RunningMeanStd.empty((env.obs_size,)) if self.normalize_obs else None
        ret_rms = RunningMeanStd.empty(()) if self.normalize_returns else None

        return _AlgState(
            critic_target=model.critic,
            obs_rms=obs_rms,
            ret_rms=ret_rms,
            ret=jnp.zeros(env.num_envs),
        )

    def _actor_loss(
        self,
        model: ActorCritic,
        alg_state: _AlgState,
        env: AbstractEnv,
        env_state: DiffEnvState,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, tuple[Rollout, DiffEnvState, _AlgState]]:
        """Compute the negated discounted short-horizon return.

        Parameters
        ----------
        - `model`: The `ActorCritic` being updated (only `actor` is used here).
        - `alg_state`: The current algorithm state.
        - `env`: The differentiable, vectorized environment to roll out in.
        - `env_state`: The environment state to resume from.
        - `key`: A `jax.random.key` used to provide randomness for action sampling.
            (Keyword only argument.)

        Returns
        -------
        The scalar actor loss, and an auxiliary tuple comprising the `Rollout` used for
        critic training, the environment state to resume from on the next call, and
        `alg_state` with `obs_rms`/`ret_rms` updated (gradient-free) from this rollout.
        """
        env_state = lax.stop_gradient(env_state)

        obs_rms = alg_state.obs_rms
        ret_var = alg_state.ret_rms.var if alg_state.ret_rms is not None else None

        def _normalize_obs(obs: Array) -> Array:
            if obs_rms is None:
                return obs
            return lax.stop_gradient(obs_rms).normalize(obs)

        def _step(
            carry: tuple[DiffEnvState, Array, Array, Array],
            xs: tuple[PRNGKeyArray, Array],
        ):
            state, rew_acc, discount, ret = carry
            step_key, i = xs
            env_keys = jr.split(step_key, env.num_envs)

            obs = _normalize_obs(state.obs)
            sample_fn = lambda o, k: model.actor.sample(o, key=k)
            actions, _ = jax.vmap(sample_fn)(obs, env_keys)
            if self.squash_actions:
                actions = jnn.tanh(actions)

            next_state = cast(DiffEnvState, env.step(state, actions))

            raw_reward = next_state.reward
            scaled_reward = raw_reward * self.reward_scale
            ret = ret * self.gamma + scaled_reward
            if ret_var is not None:
                reward = scaled_reward / jnp.sqrt(ret_var + EPSILON)
            else:
                reward = scaled_reward

            terminal_obs = _normalize_obs(next_state.terminal_obs)

            next_value = jax.vmap(alg_state.critic_target)(terminal_obs)
            next_value = next_value * (1.0 - next_state.terminated)

            done = next_state.done
            is_last = i == self.num_steps - 1
            rew_acc = rew_acc + discount * reward
            contribution = jnp.where(
                done | is_last,
                -(rew_acc + self.gamma * discount * next_value),
                0.0,
            )

            discount = jnp.where(done, 1.0, discount * self.gamma)
            rew_acc = jnp.where(done, 0.0, rew_acc)

            out = _StepOutput(
                obs=obs,
                raw_obs=next_state.obs,
                reward=reward,
                raw_reward=raw_reward,
                done=jnp.where(is_last, 1.0, done.astype(jnp.float32)),
                raw_done=done,
                next_value=next_value,
                ret=ret,
                metrics=next_state.aux,
            )
            return (next_state, rew_acc, discount, ret), (out, contribution)

        step_keys = jr.split(key, self.num_steps)
        init_carry = (
            env_state,
            jnp.zeros(env.num_envs),
            jnp.ones(env.num_envs),
            alg_state.ret,
        )
        (final_state, _, _, final_ret), (outputs, contributions) = lax.scan(
            _step, init_carry, (step_keys, jnp.arange(self.num_steps))
        )

        actor_loss = jnp.sum(contributions) / (self.num_steps * env.num_envs)
        if ret_var is not None:
            actor_loss = actor_loss * jnp.sqrt(ret_var + EPSILON)

        new_obs_rms = None
        if obs_rms is not None:
            raw_obs_stream = jnp.concatenate(
                [env_state.obs[None], outputs.raw_obs], axis=0
            )
            flat_obs = raw_obs_stream.reshape(-1, raw_obs_stream.shape[-1])
            new_obs_rms = obs_rms.update(lax.stop_gradient(flat_obs))

        new_ret_rms, new_ret = None, alg_state.ret
        if alg_state.ret_rms is not None:
            new_ret_rms = alg_state.ret_rms.update(
                lax.stop_gradient(outputs.ret.reshape(-1))
            )
            new_ret = lax.stop_gradient(final_ret)

        new_alg_state = _AlgState(
            critic_target=alg_state.critic_target,
            obs_rms=new_obs_rms,
            ret_rms=new_ret_rms,
            ret=new_ret,
        )

        rollout = Rollout(
            obs=outputs.obs,
            rewards=outputs.reward,
            dones=outputs.done,
            next_values=outputs.next_value,
            metrics={
                **outputs.metrics,
                "mean_reward": jnp.mean(outputs.raw_reward),
                "done_rate": jnp.mean(outputs.raw_done),
            },
        )

        return actor_loss, (rollout, final_state, new_alg_state)

    def _critic_loss(
        self, model: ActorCritic, batch: CriticBatch
    ) -> tuple[Array, Metrics]:
        """Compute the critic's regression loss for a batch of target values.

        Parameters
        ----------
        - `model`: The `ActorCritic` being updated. Only `model.critic` is used.
        - `batch`: A `CriticBatch` of observations and target values to regress onto.

        Returns
        -------
        The scalar loss and a pytree of metrics used for logging/analytics.
        """
        values = jax.vmap(model.critic)(batch.obs)
        value_loss = 0.5 * jnp.mean((values - batch.target_values) ** 2)
        return value_loss, cast(Metrics, {"value_loss": value_loss})

    def _update_critic(
        self,
        model: ActorCritic,
        opt_state: OptState,
        optim: Optimizer,
        batch: CriticBatch,
        *,
        key: PRNGKeyArray,
    ) -> tuple[ActorCritic, OptState, Metrics]:
        """Update the critic for `num_critic_epochs` passes over `batch`.

        Parameters
        ----------
        - `model`: The `ActorCritic` to update.
        - `opt_state`: The optimizer state for `model` (i.e. `opt_state["critic"]`, as
            built by `PolicyTrainer.learn`).
        - `optim`: The optax optimizer used to update `model` (i.e. `optim["critic"]`).
        - `batch`: The critic regression targets to train on, reshuffled and split into
            `num_minibatches` minibatches for each epoch.
        - `key`: A `jax.random.key` used to provide randomness for shuffling `batch`
            each epoch. (Keyword only argument.)

        Returns
        -------
        The updated model and optimizer state, and a dictionary of scalar metrics
        averaged over every epoch and minibatch.
        """
        params, static = eqx.partition(model, eqx.is_inexact_array)

        batch_size = jtu.tree_leaves(batch)[0].shape[0]
        minibatch_size = batch_size // self.num_minibatches

        loss_and_grad = eqx.filter_value_and_grad(self._critic_loss, has_aux=True)

        def _minibatch_step(
            carry: tuple[PyTree, OptState], minibatch: CriticBatch
        ) -> tuple[tuple[PyTree, OptState], Metrics]:
            params, opt_state = carry
            model = eqx.combine(params, static)
            (_, metrics), grads = loss_and_grad(model, minibatch)

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

        epoch_keys = jr.split(key, self.num_critic_epochs)
        (params, opt_state), metrics = lax.scan(
            _epoch_step, (params, opt_state), epoch_keys
        )
        metrics = jtu.tree_map(jnp.mean, metrics)
        model = eqx.combine(params, static)
        return model, opt_state, metrics

    @override
    def step(
        self,
        model: ActorCritic,
        alg_state: _AlgState,
        opt_state: dict[str, OptState],
        optim: dict[str, Optimizer],
        env: AbstractEnv,
        env_state: DiffEnvState,
        key: PRNGKeyArray,
    ) -> tuple[ActorCritic, _AlgState, dict[str, OptState], EnvState, Metrics]:
        """Run one SHAC iteration.

        See `AbstractAlgorithm.step` for the parameter and return descriptions.

        Raises
        ------
        `ValueError` if `optim` or `opt_state` is not a mapping with `"actor"` and
        `"critic"` keys.
        """
        actor_key, critic_key = jr.split(key)

        loss_and_grad = eqx.filter_value_and_grad(self._actor_loss, has_aux=True)
        (actor_loss, (rollout, env_state, alg_state)), grads = loss_and_grad(
            model, alg_state, env, env_state, key=actor_key
        )
        actor_params = eqx.filter(model, eqx.is_inexact_array)
        actor_updates, actor_opt_state = optim["actor"].update(
            grads, opt_state["actor"], params=actor_params
        )
        model = eqx.apply_updates(model, actor_updates)
        actor_grad_norm = optax.global_norm(grads)

        target_values = _compute_target_values(
            rollout, self.gamma, self.gae_lambda, self.critic_method
        )
        total_size = self.num_steps * env.num_envs
        critic_batch = CriticBatch(
            obs=rollout.obs.reshape(total_size, *rollout.obs.shape[2:]),
            target_values=target_values.reshape(total_size),
        )
        model, critic_opt_state, critic_metrics = self._update_critic(
            model, opt_state["critic"], optim["critic"], critic_batch, key=critic_key
        )
        opt_state = {"actor": actor_opt_state, "critic": critic_opt_state}

        target_arrays, target_static = eqx.partition(
            alg_state.critic_target, eqx.is_inexact_array
        )
        critic_arrays, _ = eqx.partition(model.critic, eqx.is_inexact_array)
        target_arrays = jtu.tree_map(
            lambda t, c: self.target_alpha * t + (1.0 - self.target_alpha) * c,
            target_arrays,
            critic_arrays,
        )
        alg_state = _AlgState(
            critic_target=eqx.combine(target_arrays, target_static),
            obs_rms=alg_state.obs_rms,
            ret_rms=alg_state.ret_rms,
            ret=alg_state.ret,
        )

        metrics: Metrics = {
            "loss": actor_loss,
            "actor_loss": actor_loss,
            "actor_grad_norm": actor_grad_norm,
        }
        metrics.update(critic_metrics)
        metrics.update(rollout.metrics)

        return model, alg_state, opt_state, env_state, cast(Metrics, metrics)
