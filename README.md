# lrax

lrax is a JAX library for robotics learning research. It provides JIT-compiled
building blocks and training infrastructure for fast training pipelines and
realtime hardware deployment.

## Main Features

- Standard robotics learning algorithms, such as `PPO`, `SAC`, and `SHAC`
- Vectorized environments for GPU-accelerated training
- Training interfaces for RL and supervised learning with built-in logging support

## Installation

lrax requires Python 3.12+, and can be installed using pip or uv

```bash
# pip installation
pip install git+https://github.com/OSU-LRAM/lrax.git

# uv installation
uv pip install git+https://github.com/OSU-LRAM/lrax.git

# or add to your uv project using
uv add git+ssh://git@github.com/OSU-LRAM/lrax.git
```

To install the MuJoCo helpers, include the optional `mjx` dependencies

```bash
uv add "lrax[mjx] @ git+ssh://git@github.com/OSU-LRAM/lrax.git"
```

## Usage

See the following example for training a policy with PPO against a custom
environment.

```python
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray
from lrax import PPO
from lrax.common.envs import AbstractEnv, EnvState
from lrax.common.policies import Actor, ActorCritic, Critic
from lrax.common.trainer import PolicyTrainer


class PointMassEnv(AbstractEnv):
    """A minimal environment: drive a 1D point mass to the origin."""

    obs_size: int = 2
    act_size: int = 1
    num_envs: int = 16

    def reset(self, key: PRNGKeyArray) -> EnvState:
        obs = jr.uniform(key, (self.num_envs, 2), minval=-1.0, maxval=1.0)
        return EnvState(
            pipeline_state=None,
            obs=obs,
            reward=jnp.zeros(self.num_envs),
            done=jnp.zeros(self.num_envs, dtype=bool),
            aux={},
        )

    def step(self, state: EnvState, action: Array) -> EnvState:
        pos, vel = state.obs[:, 0], state.obs[:, 1]
        vel = vel + action[:, 0] * 0.1
        pos = pos + vel * 0.1
        obs = jnp.stack([pos, vel], axis=-1)
        reward = -(pos**2)
        return EnvState(
            pipeline_state=None,
            obs=obs,
            reward=reward,
            done=jnp.zeros(self.num_envs, dtype=bool),
            aux={}, # include auxilliary data from the environment
        )

key = jr.key(0)
actor_key, critic_key, train_key = jr.split(key, 3)
env = PointMassEnv()

actor = Actor(env.obs_size, env.act_size, width_size=32, depth=2, key=actor_key)
critic = Critic(env.obs_size, width_size=32, depth=2, key=critic_key)

# uses the `PolicyTrainer` to train a policy using the custom environment
trained_model = PolicyTrainer(name="ppo").learn(
    train_key,
    ActorCritic(actor, critic),
    env,
    PPO(),
    optax.adam(3e-4),
    num_iterations=100,
)
```

### SHAC

`SHAC` (Short-Horizon Actor-Critic) trains by backpropagating through the environment's
dynamics, so it requires a *differentiable* simulator: `env.step` must support autodiff
from the actions it receives through to the next `EnvState`. Because it needs the
observation immediately before an episode auto-resets, both `env.reset` and `env.step`
must return `DiffEnvState` rather than plain `EnvState`. And because the actor is
updated by differentiating through the rollout while the critic is trained separately
by regression, `optim` must be a mapping with `"actor"` and `"critic"` keys rather than
a single optimizer.

```python
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray
from lrax import SHAC
from lrax.common.envs import AbstractEnv, DiffEnvState
from lrax.common.policies import Actor, ActorCritic, Critic
from lrax.common.trainer import PolicyTrainer


class DiffPointMassEnv(AbstractEnv):
    """A differentiable version of the point mass above."""

    obs_size: int = 2
    act_size: int = 1
    num_envs: int = 16

    def reset(self, key: PRNGKeyArray) -> DiffEnvState:
        obs = jr.uniform(key, (self.num_envs, 2), minval=-1.0, maxval=1.0)
        return DiffEnvState(
            pipeline_state=None,
            obs=obs,
            reward=jnp.zeros(self.num_envs),
            done=jnp.zeros(self.num_envs, dtype=bool),
            aux={},
            terminated=jnp.zeros(self.num_envs, dtype=bool),
            terminal_obs=obs,
        )

    def step(self, state: DiffEnvState, action: Array) -> DiffEnvState:
        pos, vel = state.obs[:, 0], state.obs[:, 1]
        vel = vel + action[:, 0] * 0.1
        pos = pos + vel * 0.1
        obs = jnp.stack([pos, vel], axis=-1)
        reward = -(pos**2)
        done = jnp.zeros(self.num_envs, dtype=bool)  # e.g. a time limit, in practice
        return DiffEnvState(
            pipeline_state=None,
            obs=obs,
            reward=reward,
            done=done,
            aux={},
            terminated=jnp.zeros(self.num_envs, dtype=bool),  # true (bad-state) termination
            terminal_obs=obs,  # obs *before* any auto-reset applied by `done`
        )


key = jr.key(0)
actor_key, critic_key, train_key = jr.split(key, 3)
env = DiffPointMassEnv()

actor = Actor(env.obs_size, env.act_size, width_size=32, depth=2, key=actor_key)
critic = Critic(env.obs_size, width_size=32, depth=2, key=critic_key)

# the actor and critic are optimized separately, so `optim` is a mapping rather than a
# single optimizer; the betas below match SHAC's reference implementation
optim = {
    "actor": optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.linear_schedule(2e-3, 1e-5, 100), b1=0.7, b2=0.95),
    ),
    "critic": optax.chain(optax.zero_nans(), optax.adam(2e-3, b1=0.7, b2=0.95)),
}

trained_model = PolicyTrainer(name="shac").learn(
    train_key,
    ActorCritic(actor, critic),
    env,
    SHAC(),
    optim,
    num_iterations=100,
)
```

Because each iteration keeps the full `num_steps`-step computation graph in memory for
the backward pass, activation memory scales with `num_steps * env.num_envs`; reduce one
or the other if you run out of memory.

## Citation

If you use lrax in your research, please cite the project:

```bibtex
@misc{lrax2026github,
  author  = {Palmer, Evan F. and Hatton, Ross L.},
  title   = {{LRAX}: A {JAX} library for robotics learning research},
  url     = {http://github.com/OSU-LRAM/lrax},
  year    = {2026},
}
```

## License

lrax is released under the MIT license.
