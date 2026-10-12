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

import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike


def discount(
    rewards: Float[Array, "time *batch"],
    gamma: ScalarLike = 0.99,
    *,
    decay: Callable[[Array], Array] | None = None,
) -> Float[Array, "*batch"]:
    """Calculate the discounted return of a rollout.

    Parameters
    ----------
    - `rewards`: The rewards collected over the rollout, with time along the leading
        axis.
    - `gamma`: The discount factor in `[0, 1]`. Ignored when `decay` is given.
    - `decay`: A function mapping the step indices `0, ..., T - 1` to the weight
        applied to the reward at each step, or `None` to weight step `t` by
        `gamma**t`. Defaults to `None`.

    Returns
    -------
    The weighted sum of the rewards over the time axis, with shape
    `rewards.shape[1:]`.
    """
    t = jnp.arange(rewards.shape[0])
    weights = jnp.power(gamma, t) if decay is None else decay(t)
    return jnp.tensordot(weights, rewards, axes=1)
