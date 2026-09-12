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
from typing import Literal

import jax
import jax.numpy as jnp
from equinox.internal import doc_repr
from jaxtyping import Array, ArrayLike, Float, PyTree, Scalar, ScalarLike, Shaped

_euclidean = doc_repr(lambda x, y: x - y, "<euclidean distance>")

_P = PyTree[Shaped[ArrayLike, "?*p"], "P"]
_Kernel = Callable[[_P, _P], Scalar]


def _squared_radius[P](
    x: P, y: P, sigma: ArrayLike, distance: Callable[[P, P], Array]
) -> Scalar:
    return jnp.sum((distance(x, y) / sigma) ** 2)


def _radius[P](
    x: P,
    y: P,
    sigma: ArrayLike,
    distance: Callable[[P, P], Array],
    order: ScalarLike = 1.0,
) -> Scalar:
    squared = _squared_radius(x, y, sigma, distance)
    positive = squared > 0.0
    safe = jnp.where(positive, squared, 1.0)
    return jnp.where(positive, safe ** (order / 2.0), 0.0)


def exponential_kernel[P](
    x: P,
    y: P,
    sigma: ArrayLike = 1.0,
    order: ScalarLike = 1.0,
    *,
    distance: Callable[[P, P], Array] = _euclidean,
) -> Scalar:
    r"""Exponential kernel.

    .. math::

        k(x, y) = \exp\left(-r^{p}\right),
        \qquad
        r = \left\|\frac{d(x, y)}{\sigma}\right\|_{2}

    where :math:`d` is the distance function, :math:`\sigma` are the bandwidths, and
    :math:`p` is the order. An order of one gives the Laplacian kernel, matching
    `matern_kernel` at :math:`\nu = 1/2`, and an order of two gives the squared
    exponential kernel, matching `gaussian_kernel` with the bandwidths divided by
    :math:`\sqrt{2}`.

    Parameters
    ----------
    - `x`: The first point.
    - `y`: The second point, matching the structure of `x`.
    - `sigma`: The bandwidths, in the units of the distance. Either a scalar or an
        array matching the output of `distance`, giving each coordinate its own
        bandwidth.
    - `order`: The exponent applied to the radius. The kernel is positive definite for
        orders in `(0, 2]`.
    - `distance`: The distance between `x` and `y`, returning either a scalar or an
        array of coordinates. `distance` is generic as to avoid interpreting `x` and `y`
        as elements of a vector space. Defaults to the Euclidean distance.

    Returns
    -------
    The scalar kernel value, in `(0, 1]`.
    """
    return jnp.exp(-_radius(x, y, sigma, distance, order))


def energy_kernel[P](
    x: P,
    y: P,
    sigma: ArrayLike = 1.0,
    order: ScalarLike = 1.0,
    *,
    distance: Callable[[P, P], Array] = _euclidean,
) -> Scalar:
    r"""Energy kernel.

    .. math::

        k(x, y) = -r^{p},
        \qquad
        r = \left\|\frac{d(x, y)}{\sigma}\right\|_{2}

    where :math:`d` is the distance function, :math:`\sigma` are the bandwidths, and
    :math:`p` is the order. This is the kernel for which the kernel score reduces to
    the energy score, and it is the logarithm of `exponential_kernel` at the same
    bandwidths and order.

    Parameters
    ----------
    - `x`: The first point.
    - `y`: The second point, matching the structure of `x`.
    - `sigma`: The bandwidths, in the units of the distance. Either a scalar or an
        array matching the output of `distance`, giving each coordinate its own
        bandwidth.
    - `order`: The exponent applied to the radius. The kernel is conditionally
        negative definite for orders in `(0, 2]`, and the score it induces is
        degenerate at exactly two, where it compares only the means.
    - `distance`: The distance between `x` and `y`, returning either a scalar or an
        array of coordinates. `distance` is generic as to avoid interpreting `x` and `y`
        as elements of a vector space. Defaults to the Euclidean distance.

    Returns
    -------
    The scalar kernel value, in `(-inf, 0]`.
    """
    return -_radius(x, y, sigma, distance, order)


def gaussian_kernel[P](
    x: P,
    y: P,
    sigma: ArrayLike = 1.0,
    *,
    distance: Callable[[P, P], Array] = _euclidean,
) -> Scalar:
    r"""Gaussian kernel, also known as the radial basis function (RBF) kernel or
    automatic-relevance determination kernel (ARD) kernel for non-scalar bandwidths.

    .. math::

        k(x, y) = \exp\left(-\frac{r^{2}}{2}\right),
        \qquad
        r = \left\|\frac{d(x, y)}{\sigma}\right\|_{2}

    where :math:`d` is the distance function and :math:`\sigma` are the bandwidths. The
    kernel is positive definite for Euclidean distances.

    Parameters
    ----------
    - `x`: The first point.
    - `y`: The second point, matching the structure of `x`.
    - `sigma`: The bandwidths, in the units of the distance. Either a scalar or an
        array matching the output of `distance`, giving each coordinate its own
        bandwidth.
    - `distance`: The distance between `x` and `y`, returning either a scalar or an
        array of coordinates. `distance` is generic as to avoid interpreting `x` and `y`
        as elements of a vector space. Defaults to the Euclidean distance.

    Returns
    -------
    The scalar kernel value, in `(0, 1]`.
    """
    return jnp.exp(-0.5 * _squared_radius(x, y, sigma, distance))


def rational_quadratic_kernel[P](
    x: P,
    y: P,
    sigma: ArrayLike = 1.0,
    alpha: ScalarLike = 1.0,
    *,
    distance: Callable[[P, P], Array] = _euclidean,
) -> Scalar:
    r"""Rational quadratic kernel.

    .. math::

        k(x, y) = \left(1 + \frac{r^{2}}{2\alpha}\right)^{-\alpha},
        \qquad
        r = \left\|\frac{d(x, y)}{\sigma}\right\|_{2}

    where :math:`d` is the distance function, :math:`\sigma` are the bandwidths, and
    :math:`\alpha` is the shape. This is a scale mixture of `gaussian_kernel` over the
    bandwidths, so it scores several scales at once, and it converges to
    `gaussian_kernel` as :math:`\alpha` grows.

    Parameters
    ----------
    - `x`: The first point.
    - `y`: The second point, matching the structure of `x`.
    - `sigma`: The bandwidths, in the units of the distance. Either a scalar or an
        array matching the output of `distance`, giving each coordinate its own
        bandwidth.
    - `alpha`: The shape of the mixture over bandwidths. Smaller values spread the
        mixture over a wider range of scales.
    - `distance`: The distance between `x` and `y`, returning either a scalar or an
        array of coordinates. `distance` is generic as to avoid interpreting `x` and `y`
        as elements of a vector space. Defaults to the Euclidean distance.

    Returns
    -------
    The scalar kernel value, in `(0, 1]`.
    """
    squared = _squared_radius(x, y, sigma, distance)
    return jnp.exp(-alpha * jnp.log1p(squared / (2.0 * alpha)))  # type: ignore


def inverse_multiquadric_kernel[P](
    x: P,
    y: P,
    sigma: ArrayLike = 1.0,
    beta: ScalarLike = 0.5,
    *,
    distance: Callable[[P, P], Array] = _euclidean,
) -> Scalar:
    r"""Inverse multiquadric kernel.

    .. math::

        k(x, y) = \left(1 + r^{2}\right)^{-\beta},
        \qquad
        r = \left\|\frac{d(x, y)}{\sigma}\right\|_{2}

    where :math:`d` is the distance function, :math:`\sigma` are the bandwidths, and
    :math:`\beta` is the decay. The tails decay polynomially rather than
    exponentially, so the gradient does not vanish for distant points the way it does
    under `gaussian_kernel`.

    Parameters
    ----------
    - `x`: The first point.
    - `y`: The second point, matching the structure of `x`.
    - `sigma`: The bandwidths, in the units of the distance. Either a scalar or an
        array matching the output of `distance`, giving each coordinate its own
        bandwidth.
    - `beta`: The decay of the tails. Values in `(0, 1)` are characteristic.
    - `distance`: The distance between `x` and `y`, returning either a scalar or an
        array of coordinates. `distance` is generic as to avoid interpreting `x` and `y`
        as elements of a vector space. Defaults to the Euclidean distance.

    Returns
    -------
    The scalar kernel value, in `(0, 1]`.
    """
    return jnp.exp(-beta * jnp.log1p(_squared_radius(x, y, sigma, distance)))  # type: ignore


def matern_kernel[P](
    x: P,
    y: P,
    sigma: ArrayLike = 1.0,
    nu: Literal["1/2", "3/2", "5/2"] = "3/2",
    *,
    distance: Callable[[P, P], Array] = _euclidean,
) -> Scalar:
    r"""Matern kernel.

    .. math::

        k(x, y) = \begin{cases}
            \exp(-r) & \nu = 1/2 \\
            \left(1 + \sqrt{3} r\right) \exp(-\sqrt{3} r) & \nu = 3/2 \\
            \left(1 + \sqrt{5} r + \frac{5 r^{2}}{3}\right)
                \exp(-\sqrt{5} r) & \nu = 5/2
        \end{cases}
        \qquad
        r = \left\|\frac{d(x, y)}{\sigma}\right\|_{2}

    where :math:`d` is the distance function, :math:`\sigma` are the bandwidths, and
    :math:`\nu` is the smoothness. The sample paths are :math:`\lceil \nu \rceil - 1`
    times differentiable, so the smoothness sits between `exponential_kernel` at
    :math:`\nu = 1/2` and `gaussian_kernel` in the limit.

    Parameters
    ----------
    - `x`: The first point.
    - `y`: The second point, matching the structure of `x`.
    - `sigma`: The bandwidths, in the units of the distance. Either a scalar or an
        array matching the output of `distance`, giving each coordinate its own
        bandwidth.
    - `nu`: The smoothness, one of `"1/2"`, `"3/2"`, or `"5/2"`. The general kernel
        needs a modified Bessel function, so only the half-integer forms are
        implemented.
    - `distance`: The distance between `x` and `y`, returning either a scalar or an
        array of coordinates. `distance` is generic as to avoid interpreting `x` and `y`
        as elements of a vector space. Defaults to the Euclidean distance.

    Returns
    -------
    The scalar kernel value, in `(0, 1]`.

    Raises
    ------
    A `ValueError` if `nu` is not one of the supported half-integers.
    """
    radius = _radius(x, y, sigma, distance)

    match nu:
        case "1/2":
            return jnp.exp(-radius)
        case "3/2":
            scaled = math.sqrt(3.0) * radius
            return (1.0 + scaled) * jnp.exp(-scaled)
        case "5/2":
            scaled = math.sqrt(5.0) * radius
            return (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)
        case _:
            raise ValueError(f'`nu` should either be "1/2", "3/2", or "5/2". Got {nu}.')


def mixture[P](
    kernel: Callable[..., Scalar],
    sigma: Float[ArrayLike, "scales ?*sigma"],
    weights: Float[ArrayLike, " scales"] | None = None,
) -> Callable[[P, P], Scalar]:
    r"""Mix a kernel over a ladder of bandwidths.

    .. math::

        k(x, y) = \sum_{l} w_{l}\, k(x, y; \sigma_{l})

    where :math:`\sigma_{l}` is the :math:`l`th rung of the ladder and :math:`w_{l}`
    is its weight.

    Parameters
    ----------
    - `kernel`: The kernel to mix.
    - `sigma`: The ladder of bandwidths.
    - `weights`: The weight of each rung. Defaults to weighting the rungs equally.

    Returns
    -------
    The mixed kernel, taking two points and returning a scalar.

    Raises
    ------
    A `ValueError` if `weights` does not hold one weight for each rung of the ladder.
    """
    rungs = jnp.shape(sigma)[0]

    if weights is not None and jnp.shape(weights) != (rungs,):
        raise ValueError(
            "`weights` must hold one weight for each rung of the ladder. Got weights "
            f"with shape {jnp.shape(weights)} and a ladder with {rungs} rungs."
        )

    def mixed(x: P, y: P) -> Scalar:
        def rung(bandwidths: Float[ArrayLike, "?*sigma"]) -> Scalar:
            return kernel(x, y, bandwidths)

        values = jax.vmap(rung)(sigma)

        if weights is None:
            return jnp.mean(values)

        return jnp.sum(weights * values)

    return mixed


def gram(
    kernel: _Kernel,
    xs: Shaped[_P, "n"],
    ys: Shaped[_P, "m"],
    *,
    batch_size: int | None = None,
) -> Float[Array, "n m"]:
    """Evaluate a kernel on every pair drawn from two sets of samples.

    Parameters
    ----------
    - `kernel`: The kernel applied to a single pair of points.
    - `xs`: The first set of samples, with a leading sample axis on each leaf.
    - `ys`: The second set of samples, matching the structure of `xs`.
    - `batch_size`: How many rows to form at a time, or `None` to form them all at
        once. This generally should be left as `None`, but you can play with this if you
        are running into memory issues. Keyword-only argument.

    Returns
    -------
    The Gram matrix, whose `(i, j)` entry is the kernel at the `i`th member of `xs` and
    the `j`th member of `ys`.

    Raises
    ------
    A `ValueError` if `batch_size` is not positive.
    """

    def row(x: _P) -> Float[Array, " m"]:
        return jax.vmap(kernel, in_axes=(None, 0))(x, ys)

    if batch_size is None:
        return jax.vmap(row)(xs)

    if batch_size < 1:
        raise ValueError(f"`batch_size` must be positive. Got {batch_size}.")

    return jax.lax.map(jax.checkpoint(row), xs, batch_size=batch_size)
