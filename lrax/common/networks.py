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

from typing import Callable

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.internal import doc_repr
from jaxtyping import Array, PRNGKeyArray

from .activations import lipswish

_tanh = doc_repr(jnn.tanh, "<function tanh>")
_identity = doc_repr(lambda x: x, "lambda x: x")
_lipswish = doc_repr(lipswish, "<function lipswish>")
_softplus = doc_repr(jnn.softplus, "<function softplus>")


class SPD(eqx.Module):
    """Symmetric positive-definite network.

    This uses an MLP to predict a general (n, n) matrix and applies a positive function
    to its eigenvalues. This enables, e.g., geometrically-consistent SPD predictions.
    """

    size: int = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    metric: Callable
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        diag_size: int,
        width_size: int,
        depth: int,
        activation: Callable = _lipswish,
        final_activation: Callable = _identity,
        metric: Callable = _softplus,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize an SPD network.

        Parameters
        ----------
        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `diag_size`: The diagonal size of the predicted matrix. The output from the
            module will be a matrix of shape `(diag_size, diag_size)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
        - `activation`: The activation function after each hidden layer. Defaults to
            `jax.nn.tanh`.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `metric`: The positive function to apply to the eigenvalues. Defaults to
            `jax.nn.softplus`. To implement a Log-Euclidean Riemannian metric, this can
            be set to, `jnp.exp`.
        - `scales`: A JAX array representing the pre-initialized scales to apply to the
            predicted SPD matrix. If `None`, the scales are randomly initialized.
            Defaults to `None`.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        self.size = int(diag_size * (diag_size + 1) / 2)
        self.shape = (diag_size, diag_size)
        self.metric = metric
        self.mlp = eqx.nn.MLP(
            in_size, self.size, width_size, depth, activation, final_activation, key=key
        )

    def __call__(self, x: Array) -> Array:
        """Forward pass of the SPD network.

        Parameters
        ----------
        - `x`: A JAX array with shape `(in_size,)`.

        Returns
        -------
        A JAX array with shape `(diag_size, diag_size)`.
        """
        x = self.mlp(x)
        diag, off_diag = jnp.split(x, (self.shape[0],), axis=-1)
        i_lower, j_lower = jnp.tril_indices(self.shape[0], -1)
        U = jnp.diag(diag)
        U = U.at[i_lower, j_lower].set(off_diag)
        U = U.at[j_lower, i_lower].set(off_diag)
        V, w = jax.lax.linalg.eigh(U, sort_eigenvalues=False)
        M = V @ jnp.diag(self.metric(w)) @ V.T
        return M


class TamedMLP(eqx.Module):
    """Implements a tamed MLP, which helps prevent model blow-up during training.

    For additional information, see,

    "On neural differential equations"
    (https://ora.ox.ac.uk/objects/uuid:af32d844-df84-4fdc-824d-44bebc3d7aa9)
    """

    mlp: eqx.nn.MLP
    out_scale: Array

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = _lipswish,
        final_activation: Callable = _tanh,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize an SPD network.

        Parameters
        ----------
        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
        - `activation`: The activation function after each hidden layer. Defaults to
            `lipswish`.
        - `final_activation`: The activation function after the output layer. Defaults
            to `jax.nn.tanh`.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        keys = jr.split(key)
        self.out_scale = jr.normal(keys[0])
        self.mlp = eqx.nn.MLP(
            in_size,
            out_size,
            width_size,
            depth,
            activation,
            final_activation,
            key=keys[1],
        )

    def __call__(self, x: Array) -> Array:
        """Forward pass of the network.

        Parameters
        ----------
        - `x`: A JAX array with shape `(in_size,)`.

        Returns
        -------
        A JAX array with shape `(out_size,)`.
        """
        return self.out_scale * self.mlp(x)
