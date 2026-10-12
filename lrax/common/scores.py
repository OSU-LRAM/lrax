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

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, Float, Scalar, ScalarLike, Shaped

from .._custom_types import Kernel, P
from .kernels import gaussian


def _get_sample_size(samples: Shaped[P, "samples"]) -> int:
    def check(path: jtu.KeyPath, leaf: ArrayLike) -> int:
        shape = jnp.shape(leaf)

        if not shape:
            raise ValueError(
                "Every leaf must carry a leading sample axis. Got a scalar leaf at "
                f"`samples{jtu.keystr(path)}`."
            )

        return shape[0]

    leaves = jtu.tree_leaves_with_path(samples)

    if not leaves:
        raise ValueError("`samples` must have at least one array leaf.")

    sizes = {check(path, leaf) for path, leaf in leaves}

    if len(sizes) != 1:
        raise ValueError(
            "Every leaf must agree on the number of samples. Got the sample counts "
            f"{sorted(sizes)}."
        )

    return sizes.pop()


def _map[X, Y](f: Callable[[X], Y], xs: X, batch_size: int | None) -> Y:
    if batch_size is None:
        return jax.vmap(f)(xs)

    if batch_size < 1:
        raise ValueError(f"`batch_size` must be positive. Got {batch_size}.")

    return jax.lax.map(jax.checkpoint(f), xs, batch_size=batch_size)


def _total(
    kernel: Kernel,
    xs: Shaped[P, "n"],
    ys: Shaped[P, "m"],
    batch_size: int | None,
) -> Scalar:
    def row(x: P) -> Scalar:
        return jnp.sum(jax.vmap(kernel, in_axes=(None, 0))(x, ys))

    return jnp.sum(_map(row, xs, batch_size))


def _within(
    kernel: Kernel,
    xs: Shaped[P, "samples"],
    unbiased: bool,
    batch_size: int | None,
) -> Scalar:
    n = _get_sample_size(xs)

    if unbiased and n < 2:
        raise ValueError(
            "The unbiased estimator drops the pairs of a member with itself, so it "
            f"needs at least two samples. Got {n}. Pass `unbiased=False` to average "
            "over those pairs as well."
        )

    total = _total(kernel, xs, xs, batch_size)

    if not unbiased:
        return total / n**2

    def alone(x: P) -> Scalar:
        return kernel(x, x)

    return (total - jnp.sum(_map(alone, xs, batch_size))) / (n * (n - 1))


def kernel(
    observation: P,
    forecasts: Shaped[P, "samples"],
    kernel: Kernel = gaussian,
    *,
    unbiased: bool = True,
    batch_size: int | None = None,
) -> Scalar:
    """Kernel scoring rule for a forecast.

    Parameters
    ----------
    - `observation`: The observed (data) point.
    - `forecasts`: A set of forecasted samples, with each sample sharing the tree
        structure of `observation`, with a leading axis on each leaf.
    - `kernel`: The kernel applied to a single pair of points.
    - `unbiased`: Whether to estimate the within-set term with the U-statistic, which
        is unbiased for the squared discrepancy at the cost of admitting slightly
        negative values.
    - `batch_size`: The number of rows of pairs to form at a time or `None` to form them
        all at once (i.e., use `jax.vmap` or `jax.lax.map` with a batch reduction for
        each row).

    Returns
    -------
    The scalar squared discrepancy. If the kernel is characteristic, the score is
    strictly proper.
    """

    def against(x: P) -> Scalar:
        return kernel(x, observation)

    spread = _within(kernel, forecasts, unbiased, batch_size)
    accuracy = jnp.mean(_map(against, forecasts, batch_size))

    return 0.5 * spread - accuracy + 0.5 * kernel(observation, observation)


def mmd(
    observations: Shaped[P, "n"],
    forecasts: Shaped[P, "m"],
    kernel: Kernel = gaussian,
    *,
    unbiased: bool = True,
    within_observations: bool = False,
    batch_size: int | None = None,
) -> Scalar:
    """Squared maximum mean discrepancy between two sets of samples.

    Parameters
    ----------
    - `observations`: A set of observed (data) samples, with a leading axis on each
        leaf.
    - `forecasts`: A set of forecasted samples, with each sample sharing the tree
        structure of the observations, with a leading axis on each leaf. The two sets
        need not hold the same number of samples.
    - `kernel`: The kernel applied to a single pair of points.
    - `unbiased`: Whether to estimate the within-set terms with the U-statistic, which
        is unbiased for the squared discrepancy at the cost of admitting slightly
        negative values.
    - `within_observations`: Whether to include the within-set term over the
        observations. It does not depend on the forecasts, so leaving it out shifts the
        discrepancy by a constant and leaves its gradient untouched.
    - `batch_size`: The number of rows of pairs to form at a time or `None` to form them
        all at once (i.e., use `jax.vmap` or `jax.lax.map` with a batch reduction for
        each row).

    Returns
    -------
    The scalar squared discrepancy. If the kernel is characteristic, the score is
    strictly proper.
    """
    n = _get_sample_size(observations)
    m = _get_sample_size(forecasts)

    predicted = _within(kernel, forecasts, unbiased, batch_size)
    cross = _total(kernel, observations, forecasts, batch_size) / (n * m)
    discrepancy = predicted - 2.0 * cross

    if not within_observations:
        return discrepancy

    return discrepancy + _within(kernel, observations, unbiased, batch_size)


def variogram(
    observation: P,
    forecasts: Shaped[P, "samples"],
    order: ScalarLike = 0.5,
    weights: Float[Array, "d d"] | None = None,
) -> Scalar:
    """Variogram scoring rule for a multivariate forecast.

    Parameters
    ----------
    - `observation`: The observed (data) point.
    - `forecasts`: A set of forecasted samples, with each sample sharing the tree
        structure of `observation`, with a leading axis on each leaf.
    - `order`: The exponent applied to the pairwise differences. Orders in `(0, 1]` damp
        the influence of the tails, with one half the usual choice.
    - `weights`: The weight of each pair of coordinates, with shape `(d, d)` for a
        `d`-dimensional observation, or `None` to weight every pair equally. Weights
        that decay with the separation between coordinates are common when the
        coordinates are laid out in space or time.

    Returns
    -------
    The scalar score. The score is proper but not strictly proper: it reads only the
    differences between coordinates, so it is blind to a shift applied to every
    coordinate at once.
    """

    def powered(values: Array) -> Array:
        # Mask the zeros so a fractional `order` is never differentiated at zero, which
        # happens on the diagonal, where it would give an infinite gradient.
        positive = values > 0.0
        safe = jnp.where(positive, values, 1.0)
        return jnp.where(positive, safe**order, 0.0)

    _get_sample_size(forecasts)  # validates the tree, the count itself is unused

    y, _ = ravel_pytree(observation)
    xs = jax.vmap(lambda sample: ravel_pytree(sample)[0])(forecasts)

    observed = powered(jnp.abs(y[:, None] - y[None, :]))
    predicted = powered(jnp.abs(xs[:, :, None] - xs[:, None, :]))
    residual = (observed - jnp.mean(predicted, axis=0)) ** 2

    if weights is None:
        return jnp.sum(residual)

    if jnp.shape(weights) != jnp.shape(residual):
        raise ValueError(
            "`weights` must hold one weight for each pair of coordinates. Got weights "
            f"with shape {jnp.shape(weights)} and an observation with "
            f"{jnp.shape(residual)[0]} coordinates."
        )

    return jnp.sum(weights * residual)


def log(
    observation: Float[ArrayLike, "d"],
    mean: Float[ArrayLike, "d"],
    scale: Float[ArrayLike, "d d"] | Float[ArrayLike, "d"] | ScalarLike,
) -> Scalar:
    """Logarithmic scoring rule for a normal forecast.

    This is equivalent to the negative log likelihood of the normal distribution.

    Parameters
    ----------
    - `observation`: The observed (data) point.
    - `mean`: The mean of the forecast.
    - `scale`: The scale of the forecast. A scalar or a vector is read as the standard
        deviations of an isotropic or diagonal covariance, and a matrix is read as the
        lower triangular Cholesky factor of a dense covariance.

    Returns
    -------
    The scalar score. The score is strictly proper.
    """
    error = jnp.asarray(observation) - jnp.asarray(mean)
    scale = jnp.asarray(scale)
    dimension = error.shape[-1]

    match scale.ndim:
        case 0 | 1:
            whitened = error / scale
            volume = jnp.sum(jnp.broadcast_to(jnp.log(scale), error.shape))
        case 2:
            whitened = jsl.solve_triangular(scale, error, lower=True)
            volume = jnp.sum(jnp.log(jnp.abs(jnp.diagonal(scale))))
        case _:
            raise ValueError(
                "`scale` must either be a scalar, a vector of standard deviations, or "
                f"the Cholesky factor of a covariance. Got shape {scale.shape}."
            )

    normalization = 0.5 * dimension * math.log(2.0 * math.pi)

    return 0.5 * jnp.sum(whitened**2) + volume + normalization


def brier(
    observation: Float[ArrayLike, " k"],
    probabilities: Float[ArrayLike, " k"],
) -> Scalar:
    """Brier scoring rule for a categorical forecast.

    Parameters
    ----------
    - `observation`: The observed (data) indicator over the categories.
    - `probabilities`: The forecasted probability of each category, summing to one over
        the categories.

    Returns
    -------
    The scalar score, in `[0, 2]` over two or more categories and in `[0, 1]` in the
    binary form. The score is strictly proper.
    """
    error = jnp.asarray(probabilities) - jnp.asarray(observation)

    return jnp.sum(error**2)


def interval(
    observation: Float[ArrayLike, " d"],
    lower: Float[ArrayLike, " d"],
    upper: Float[ArrayLike, " d"],
    alpha: ScalarLike = 0.05,
) -> Scalar:
    """Interval scoring rule (Winkler score) for a central prediction interval.

    Parameters
    ----------
    - `observation`: The observed (data) point.
    - `lower`: The lower endpoint of the interval, at the `alpha / 2` quantile.
    - `upper`: The upper endpoint of the interval, at the `1 - alpha / 2` quantile.
    - `alpha`: The miscoverage rate in `(0, 1)`, so the interval is a central
        `1 - alpha` one. Defaults to a 95% interval.

    Returns
    -------
    The scalar score, summed over the coordinates. The score is proper for the central
    `1 - alpha` interval.
    """
    observation = jnp.asarray(observation)
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)

    below = jnp.maximum(lower - observation, 0.0)
    above = jnp.maximum(observation - upper, 0.0)

    return jnp.sum((upper - lower) + 2.0 / alpha * (below + above))


def quantile(
    observation: Float[ArrayLike, " d"],
    quantiles: Float[ArrayLike, "levels d"],
    levels: Float[ArrayLike, " levels"],
) -> Scalar:
    """Quantile scoring rule for a set of predicted quantiles.

    Parameters
    ----------
    - `observation`: The observed (data) point.
    - `quantiles`: The forecasted quantiles, one row for each level.
    - `levels`: The quantile levels in `(0, 1)`, one for each row of `quantiles`.

    Returns
    -------
    The scalar score, summed over the levels and the coordinates. The score is strictly
    proper for the quantile at each level.
    """
    error = jnp.asarray(observation) - jnp.asarray(quantiles)
    levels = jnp.asarray(levels)

    if jnp.shape(levels) != jnp.shape(error)[:1]:
        raise ValueError(
            "`levels` must hold one level for each row of `quantiles`. Got levels with "
            f"shape {jnp.shape(levels)} and quantiles with shape {jnp.shape(error)}."
        )

    def pinball(level: Scalar, row: Array) -> Scalar:
        return jnp.sum(jnp.maximum(level * row, (level - 1.0) * row))

    return jnp.sum(jax.vmap(pinball)(levels, error))
