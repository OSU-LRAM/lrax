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
from jaxtyping import Array, ArrayLike, Num, PyTree, ScalarLike, Shaped

_S = PyTree[Shaped[ArrayLike, "?*s"], "S"]


class ZeroOrderSampler(eqx.Module):
    """Select the most recent element at or before the sampling time `t`."""

    ts: Num[Array, "horizon"]
    ys: Num[_S, "horizon"]

    def __check_init__(self):
        if self.ys.shape[0] != self.ts.shape[0]:
            raise ValueError(
                "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                "number of entries along the timelike dimension."
            )

    @eqx.filter_jit
    def evaluate(self, t0: ScalarLike) -> _S:
        sample = jnp.searchsorted(self.ts, t0, side="right") - 1
        idx = jnp.clip(sample, 0, self.ys.shape[0] - 1)
        return self.ys[idx]
