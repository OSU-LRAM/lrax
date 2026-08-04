# lrax

lrax is a collection of JAX modules used by LRAM for robotics learning. The
main features of lrax include:

- `PPO` and `SAC`, JIT-compiled actor-critic reinforcement learning algorithms
  built on JAX and Equinox
- `Actor`, `Critic`, `ActorCritic`, and `ContinuousCritic`, reusable policy and
  value network building blocks
- `PolicyTrainer` and `ModelTrainer`, training loops for RL and supervised
  learning respectively
- `AbstractEnv`, a vectorized environment interface, with an optional MJX
  backend for MuJoCo-based environments

## Installation

lrax can be installed using pip or uv,

```bash
# pip installation
pip install git+https://github.com/OSU-LRAM/lrax.git

# uv installation
uv pip install git+https://github.com/OSU-LRAM/lrax.git

# or add to your uv project using
uv add git+ssh://git@github.com/OSU-LRAM/lrax.git
```

MuJoCo-based environments require the optional `mjx` extra,

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
            pipeline_state=None, obs=obs, reward=reward,
            done=jnp.zeros(self.num_envs, dtype=bool), aux={},
        )

key = jr.key(0)
actor_key, critic_key, train_key = jr.split(key, 3)
env = PointMassEnv()

model = ActorCritic(
    actor=Actor(env.obs_size, env.act_size, width_size=32, depth=2, key=actor_key),
    critic=Critic(env.obs_size, width_size=32, depth=2, key=critic_key),
)

trained_model = PolicyTrainer(name="ppo").learn(
    train_key, model, env, PPO(), optax.adam(3e-4), num_iterations=100,
)
```

## License

lrax is released under the MIT license.
