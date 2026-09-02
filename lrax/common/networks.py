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
from typing import Literal

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
        in_size: int | Literal["scalar"],
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

    "On Neural Differential Equations", Patrick Kidger.
    """

    mlp: eqx.nn.MLP
    out_scale: Array

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
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


class GRU(eqx.Module):
    """A multi-layer gated-recurrent unit (GRU)."""

    layers: tuple[eqx.nn.GRUCell, ...]
    in_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        num_layers: int = 1,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the GRU.

        Parameters
        ----------
        - `in_size`: The input size. The input to the module should be a vector of
            shape `(seq_len, in_size)`.
        - `hidden_size`: The size of each layer's hidden state.
        - `num_layers`: The total number of stacked `GRUCell` units. For example:
            - `num_layers=1` is just a single `GRUCell`
                `[GRUCell(in_size, hidden_size)]`
            - `num_layers=2` results in a network with layers
                `[GRUCell(in_size, hidden_size), GRUCell(hidden_size, hidden_size)]`
            Defaults to 1.
        - `use_bias`: Whether to add on a bias to each cell after each update. Defaults
            to `True`.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        keys = jr.split(key, num_layers)

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        init_cell = eqx.nn.GRUCell(in_size, hidden_size, use_bias, key=keys[0])
        layers = [init_cell]

        make_hidden_cell = lambda k: eqx.nn.GRUCell(
            hidden_size, hidden_size, use_bias, key=k
        )
        layers.extend(make_hidden_cell(keys[i]) for i in range(1, num_layers))

        self.layers = tuple(layers)

    def init_state(self) -> Array:
        """Create a new zeroed hidden state with shape `(num_layers, hidden_size)`."""
        return jnp.zeros((len(self.layers), self.hidden_size))

    def step(self, x: Array, h_x: Array) -> tuple[Array, Array]:
        """Advance each cell by one step.

        Parameters
        ----------
        - `x`: A JAX array of shape `(in_size,)` to provide as input to the cells.
        - `h_x`: A JAX array of shape `(num_layers, hidden_size)` holding hidden state
            of each layer.

        Returns
        -------
        The top layer's hidden state and the updated network hidden state.
        """
        new_hidden = []

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, h_x[i])
            new_hidden.append(h)

        return h, jnp.stack(new_hidden)

    def __call__(self, x: Array, h_0: Array) -> tuple[Array, Array]:
        """Apply the network over an input sequence.

        Parameters
        ----------
        - `x`: A JAX array of shape `(seq_len, in_size)` to scan the network over.
        - `h_0`: A JAX array of shape `(num_layers, hidden_size)` holding the initial
            hidden state of each layer.

        Returns
        -------
        The top layer's hidden state at each element in the sequence and the final
        hidden state of every layer.
        """

        def scan_fn(hidden: Array, x_t: Array) -> tuple[Array, Array]:
            top, hidden = self.step(x_t, hidden)
            return hidden, top

        h_n, output = jax.lax.scan(scan_fn, h_0, x)
        return output, h_n


class LSTM(eqx.Module):
    """A multi-layer long short-term memory (LSTM) network."""

    layers: tuple[eqx.nn.LSTMCell, ...]
    in_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        num_layers: int = 1,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the LSTM.

        Parameters
        ----------
        - `in_size`: The input size. The input to the module should be a vector of
            shape `(seq_len, in_size)`.
        - `hidden_size`: The size of each layer's hidden and cell state.
        - `num_layers`: The total number of stacked `LSTMCell` units. For example:
            - `num_layers=1` is just a single `LSTMCell`
                `[LSTMCell(in_size, hidden_size)]`
            - `num_layers=2` results in a network with layers
                `[LSTMCell(in_size, hidden_size), LSTMCell(hidden_size, hidden_size)]`
            Defaults to 1.
        - `use_bias`: Whether to add on a bias to each cell after each update. Defaults
            to `True`.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        keys = jr.split(key, num_layers)

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        init_cell = eqx.nn.LSTMCell(in_size, hidden_size, use_bias, key=keys[0])
        layers = [init_cell]

        make_hidden_cell = lambda k: eqx.nn.LSTMCell(
            hidden_size, hidden_size, use_bias, key=k
        )
        layers.extend(make_hidden_cell(keys[i]) for i in range(1, num_layers))

        self.layers = tuple(layers)

    def init_state(self) -> tuple[Array, Array]:
        """Create a new zeroed `(hidden, cell)` state, each with shape
        `(num_layers, hidden_size)`."""
        zeros = jnp.zeros((len(self.layers), self.hidden_size))
        return zeros, zeros

    def step(
        self, x: Array, h_x: tuple[Array, Array]
    ) -> tuple[Array, tuple[Array, Array]]:
        """Advance each cell by one step.

        Parameters
        ----------
        - `x`: A JAX array of shape `(in_size,)` to provide as input to the cells.
        - `h_x`: A `(hidden, cell)` 2-tuple of JAX arrays, each of shape
            `(num_layers, hidden_size)`, holding the state of each layer.

        Returns
        -------
        The top layer's hidden state and the updated network `(hidden, cell)` state.
        """
        hidden, cell = h_x
        new_hidden, new_cell = [], []

        h = x
        for i, layer in enumerate(self.layers):
            h, c = layer(h, (hidden[i], cell[i]))
            new_hidden.append(h)
            new_cell.append(c)

        return h, (jnp.stack(new_hidden), jnp.stack(new_cell))

    def __call__(
        self, x: Array, h_0: tuple[Array, Array]
    ) -> tuple[Array, tuple[Array, Array]]:
        """Apply the network over an input sequence.

        Parameters
        ----------
        - `x`: A JAX array of shape `(seq_len, in_size)` to scan the network over.
        - `h_0`: A `(hidden, cell)` 2-tuple of JAX arrays, each of shape
            `(num_layers, hidden_size)`, holding the initial state of each layer.

        Returns
        -------
        The top layer's hidden state at each element in the sequence and the final
        `(hidden, cell)` state of every layer.
        """

        def scan_fn(
            hidden: tuple[Array, Array], x_t: Array
        ) -> tuple[tuple[Array, Array], Array]:
            top, hidden = self.step(x_t, hidden)
            return hidden, top

        h_n, output = jax.lax.scan(scan_fn, h_0, x)
        return output, h_n
