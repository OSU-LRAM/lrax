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

The base install pulls in CPU-only JAX. For CUDA-accelerated JAX on Linux,
include the optional `cuda` dependencies

```bash
uv add "lrax[cuda] @ git+ssh://git@github.com/OSU-LRAM/lrax.git"
```

To install the MuJoCo helpers, include the optional `mjx` dependencies

```bash
uv add "lrax[mjx] @ git+ssh://git@github.com/OSU-LRAM/lrax.git"
```

Extras combine, e.g. `lrax[cuda,mjx]`.

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
            terminated=jnp.zeros(self.num_envs, dtype=bool),
            terminal_obs=obs,
            aux={},
        )

    def step(self, state: EnvState, action: Array) -> EnvState:
        pos, vel = state.obs[:, 0], state.obs[:, 1]
        vel = vel + action[:, 0] * 0.1
        pos = pos + vel * 0.1
        obs = jnp.stack([pos, vel], axis=-1)
        reward = -(pos**2)
        done = jnp.zeros(self.num_envs, dtype=bool)
        return EnvState(
            pipeline_state=None,
            obs=obs,
            reward=reward,
            done=done,
            terminated=jnp.zeros(self.num_envs, dtype=bool),
            terminal_obs=obs,
            aux={},  # include auxiliary data from the environment
        )


key = jr.key(0)
actor_key, critic_key, train_key = jr.split(key, 3)
env = PointMassEnv()

actor = Actor(env.obs_size, env.act_size, width_size=32, depth=2, key=actor_key)
critic = Critic(env.obs_size, width_size=32, depth=2, key=critic_key)

trained_model = PolicyTrainer(name="ppo").learn(
    train_key,
    ActorCritic(actor, critic),
    env,
    PPO(),
    optax.adam(3e-4),
    num_iterations=100,
)
```

## Citation

If you use lrax in your research, please cite the project:

```bibtex
@misc{lrax2026github,
  author  = {Palmer, Evan F. and Hatton, Ross L.},
  title   = {lrax: A {JAX} library for robotics learning research},
  url     = {http://github.com/OSU-LRAM/lrax},
  year    = {2026},
}
```

## License

lrax is released under the MIT license.
