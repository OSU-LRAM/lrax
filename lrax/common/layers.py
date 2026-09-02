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
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox._misc import default_floating_dtype
from equinox.nn._misc import default_init
from jaxtyping import Array, PRNGKeyArray


class BayesianLinear(eqx.Module):
    """A linear layer with a mean-field Gaussian posterior over its weights and bias."""

    mean_weight: Array
    logstd_weight: Array
    mean_bias: Array | None
    logstd_bias: Array | None
    in_features: int | Literal["scalar"] = eqx.field(static=True)
    out_features: int | Literal["scalar"] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    prior_scale: float = eqx.field(static=True)
    inference: bool

    def __init__(
        self,
        in_features: int | Literal["scalar"],
        out_features: int | Literal["scalar"],
        use_bias: bool = True,
        prior_scale: float = 1.0,
        init_logstd: float = -5.0,
        inference: bool = False,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the layer.

        Parameters
        ----------
        - `in_features`: The input size, or `"scalar"` for a scalar input.
        - `out_features`: The output size, or `"scalar"` for a scalar output.
        - `use_bias`: Whether to add a bias. Defaults to `True`.
        - `prior_scale`: Variance of the zero-mean Gaussian weight prior. Defaults
            to `1`.
        - `init_logstd`: Initial posterior log-std of every parameter; small values
            start the layer close to a deterministic `Linear`. Defaults to `-5`.
        - `inference`: If `True`, `__call__` uses the posterior mean by default. Toggle
            the whole model with `equinox.nn.inference_mode`. Defaults to `False`.
        - `dtype`: The parameter dtype. Defaults to the JAX default float.
        - `key`: A `jax.random.key` for parameter initialisation. (Keyword only
            argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jr.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1.0 if in_features_ == 0 else 1 / math.sqrt(in_features_)

        wshape = (out_features_, in_features_)
        bshape = (out_features_,)
        self.mean_weight = default_init(wkey, wshape, dtype, lim)
        self.logstd_weight = jnp.full(wshape, init_logstd, dtype=dtype)
        self.mean_bias = default_init(bkey, bshape, dtype, lim) if use_bias else None
        self.logstd_bias = (
            jnp.full(bshape, init_logstd, dtype=dtype) if use_bias else None
        )

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.prior_scale = prior_scale
        self.inference = inference

    def _sample_parameters(self, key: PRNGKeyArray) -> tuple[Array, Array | None]:
        """Draw a weight (and bias) from the posterior via the reparameterisation
        trick."""
        wkey, bkey = jr.split(key)
        eps_w = jr.normal(wkey, self.mean_weight.shape)
        weight = self.mean_weight + jnp.exp(self.logstd_weight) * eps_w
        if self.mean_bias is None or self.logstd_bias is None:
            return weight, None

        eps_b = jr.normal(bkey, self.mean_bias.shape)
        bias = self.mean_bias + jnp.exp(self.logstd_bias) * eps_b
        return weight, bias

    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Array:
        """Apply the layer to `x`, sampling the weights unless in inference mode.

        A `key` is required when not in inference mode; `inference` overrides
        `self.inference` for this call.
        """
        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))

        if inference is None:
            inference = self.inference
        if inference:
            weight, bias = self.mean_weight, self.mean_bias
        elif key is None:
            raise RuntimeError(
                "BayesianLinear requires a key when not in inference mode."
            )
        else:
            weight, bias = self._sample_parameters(key)

        x = weight @ x
        if bias is not None:
            x = x + bias

        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)

        return x

    def kl_divergence(self) -> Array:
        """The KL divergence from the weight and bias posterior to the zero-mean prior."""

        def kl(mean: Array, logstd: Array) -> Array:
            var = jnp.exp(2 * logstd)
            return 0.5 * jnp.sum(
                (mean**2 + var) / self.prior_scale
                - 1.0
                + math.log(self.prior_scale)
                - 2 * logstd
            )

        total = kl(self.mean_weight, self.logstd_weight)
        if self.mean_bias is not None and self.logstd_bias is not None:
            total = total + kl(self.mean_bias, self.logstd_bias)
        return total


class VBLL(eqx.Module):
    """A variational Bayesian last layer for regression.

    A Gaussian posterior over the output weights and a Wishart-regularised diagonal
    noise model give closed-form predictive uncertainty and a closed-form ELBO.

    "Variational Bayesian Last Layers", James Harrison, John Willes, and Jasper Snoek.
    """

    mean_weight: Array
    logdiag_weight: Array
    offdiag_weight: Array | None
    logdiag_noise: Array
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    parameterization: Literal["dense", "diagonal"] = eqx.field(static=True)
    prior_scale: float = eqx.field(static=True)
    wishart_scale: float = eqx.field(static=True)
    wishart_dof: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parameterization: Literal["dense", "diagonal"] = "dense",
        prior_scale: float = 1.0,
        wishart_scale: float = 1e-2,
        wishart_dof: float = 1.0,
        noise_scale: float = 1e-2,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the layer.

        Parameters
        ----------
        - `in_features`: The input (feature) size.
        - `out_features`: The output (target) size.
        - `parameterization`: `"dense"` for a full per-row weight covariance,
            `"diagonal"` for a factorised one. Defaults to `"dense"`.
        - `prior_scale`: Scale of the zero-mean weight prior; its variance is
            `prior_scale / in_features`. Defaults to `1`.
        - `wishart_scale`: Scale of the Wishart prior on the noise precision. Defaults
            to `1e-2`.
        - `wishart_dof`: Degrees-of-freedom slack of that prior; larger values pull the
            noise variance lower and harder. Defaults to `1`.
        - `noise_scale`: Initial observation-noise standard deviation. Defaults to
            `1e-2`.
        - `dtype`: The parameter dtype. Defaults to the JAX default float.
        - `key`: A `jax.random.key` for parameter initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, lkey, okey = jr.split(key, 3)
        shape = (out_features, in_features)

        mean_scale = math.sqrt(2 / in_features)
        logdiag_shift = 0.5 * math.log(in_features)
        self.mean_weight = mean_scale * jr.normal(wkey, shape, dtype=dtype)
        self.logdiag_weight = jr.normal(lkey, shape, dtype=dtype) - logdiag_shift
        if parameterization == "dense":
            offdiag = jr.normal(okey, (*shape, in_features), dtype=dtype)
            self.offdiag_weight = offdiag / in_features
        else:
            self.offdiag_weight = None
        self.logdiag_noise = jnp.full((out_features,), math.log(noise_scale), dtype)

        self.in_features = in_features
        self.out_features = out_features
        self.parameterization = parameterization
        self.prior_scale = prior_scale
        self.wishart_scale = wishart_scale
        self.wishart_dof = wishart_dof

    def _weight_cholesky(self) -> Array:
        """The Cholesky factor of each output row's weight covariance."""
        scale = jnp.exp(self.logdiag_weight)
        diag = scale[..., None] * jnp.eye(self.in_features, dtype=scale.dtype)
        if self.offdiag_weight is None:
            return diag
        return jnp.tril(self.offdiag_weight, -1) + diag

    @property
    def noise_variance(self) -> Array:
        """The aleatoric variance per output."""
        return jnp.exp(2 * self.logdiag_noise)

    def epistemic_variance(self, x: Array) -> Array:
        """The weight-posterior variance per output."""
        if self.offdiag_weight is None:
            return jnp.exp(2 * self.logdiag_weight) @ (x**2)
        projected = jnp.swapaxes(self._weight_cholesky(), -1, -2) @ x
        return jnp.sum(projected**2, axis=-1)

    def _trace_covariance(self) -> Array:
        """Calculate the covariance trace."""
        if self.offdiag_weight is None:
            return jnp.sum(jnp.exp(2 * self.logdiag_weight), axis=-1)
        return jnp.sum(self._weight_cholesky() ** 2, axis=(-2, -1))

    def _logdet_covariance(self) -> Array:
        """Calculate the logdet of the covariance."""
        return 2 * jnp.sum(self.logdiag_weight, axis=-1)

    def __call__(self, x: Array) -> tuple[Array, Array]:
        """The predictive mean and total variance for features `x`."""
        mean = self.mean_weight @ x
        variance = self.epistemic_variance(x) + self.noise_variance
        return mean, variance

    def log_likelihood(self, x: Array, y: Array) -> Array:
        """The expected log-likelihood of `y` under the weight posterior (the ELBO data
        term)."""
        residual = y - self.mean_weight @ x
        expected_square_error = residual**2 + self.epistemic_variance(x)
        return -0.5 * jnp.sum(
            math.log(2 * math.pi)
            + 2 * self.logdiag_noise
            + expected_square_error / self.noise_variance
        )

    def regularizer(self) -> Array:
        """The ELBO's prior penalty: the weight KL plus the Wishart noise-prior term.

        The training loss is usually something like:
        `-mean(log_likelihood) + regularization_weight * regularizer()`.
        """
        prior_variance = self.prior_scale / self.in_features
        effective_dof = 0.5 * (self.wishart_dof + self.out_features + 1)

        weight_kl = 0.5 * (
            jnp.sum(self.mean_weight**2) / prior_variance
            + jnp.sum(self._trace_covariance()) / prior_variance
            + self.out_features * self.in_features * math.log(prior_variance)
            - jnp.sum(self._logdet_covariance())
        )

        # Expected log-density of the Wishart prior on the noise precision.
        noise_logdet_precision = -2 * jnp.sum(self.logdiag_noise)
        noise_trace_precision = jnp.sum(jnp.exp(-2 * self.logdiag_noise))
        wishart = (
            effective_dof * noise_logdet_precision
            - 0.5 * self.wishart_scale * noise_trace_precision
        )

        return weight_kl - wishart
