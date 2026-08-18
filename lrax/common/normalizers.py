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
from jaxtyping import Array

from .._epsilon import EPSILON


class RunningMeanStd(eqx.Module):
    """Tracks the running mean and variance of a stream of batches."""

    mean: Array
    var: Array
    count: Array

    @classmethod
    def empty(cls, shape: tuple[int, ...] = ()) -> "RunningMeanStd":
        """Create a `RunningMeanStd` with no observations.

        Parameters
        ----------
        - `shape`: The shape of a single observation, i.e. excluding any leading batch
            dimension. Defaults to `()`, a running scalar statistic.

        Returns
        -------
        A `RunningMeanStd` with zero mean, unit variance, and zero count.
        """
        return cls(mean=jnp.zeros(shape), var=jnp.ones(shape), count=jnp.zeros(()))

    def update(self, x: Array) -> "RunningMeanStd":
        """Fold a batch of new observations into the running statistics.

        Parameters
        ----------
        - `x`: A JAX array of shape `(batch, *shape)` representing a batch of
        observations.

        Returns
        -------
        The updated `RunningMeanStd`.
        """
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = jnp.asarray(x.shape[0], dtype=self.count.dtype)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = new_m2 / total_count

        return RunningMeanStd(mean=new_mean, var=new_var, count=total_count)

    def normalize(self, x: Array) -> Array:
        """Normalize `x` to zero mean and unit variance under the running statistics.

        Parameters
        ----------
        - `x`: A JAX array broadcastable against `mean`/`var`.

        Returns
        -------
        `(x - mean) / sqrt(var + EPSILON)`.
        """
        return (x - self.mean) / jnp.sqrt(self.var + EPSILON)
