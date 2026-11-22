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

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from equinox.internal import doc_repr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

from ..typing import Activation
from .activations import lipswish

_tanh = doc_repr(jnn.tanh, "<function tanh>")
_identity = doc_repr(lambda x: x, "lambda x: x")
_lipswish = doc_repr(lipswish, "<function lipswish>")
_softplus = doc_repr(jnn.softplus, "<function softplus>")


def _vec_to_sym(x: Array, n: int) -> Array:
    """Unpack a vector into a symmetric `(n, n)` matrix."""
    diag, off_diag = jnp.split(x, (n,), axis=-1)
    i_lower, j_lower = jnp.tril_indices(n, -1)
    U = jnp.diag(diag)
    U = U.at[i_lower, j_lower].set(off_diag)
    return U.at[j_lower, i_lower].set(off_diag)


def _vec_to_tril(x: Array, n: int) -> Array:
    """Unpack a vector into a lower-triangular `(n, n)` matrix."""
    i_lower, j_lower = jnp.tril_indices(n)
    return jnp.zeros((n, n), x.dtype).at[i_lower, j_lower].set(x)


def _tril_size(n: int) -> int:
    """The number of entries in the lower triangle of an `(n, n)` matrix."""
    return n * (n + 1) // 2


def _transform_diagonal(L: Array, f: Callable[[Array], Array]) -> Array:
    """Apply a function to the diagonal of `L`."""
    i_diag = jnp.diag_indices(L.shape[-1])
    return L.at[i_diag].set(f(L[i_diag]))


def _transform_eigenvalues(U: Array, f: Callable[[Array], Array]) -> Array:
    """Apply a function to the eigenvalues of `U`."""
    V, w = jax.lax.linalg.eigh(U, sort_eigenvalues=False)
    return V @ jnp.diag(f(w)) @ V.T


class SpectralOptions(TypedDict, total=False):
    """Options for `PSD` with `method="spectral"`.

    - `eigenvalue_fn`: The positive function applied to the eigenvalues of `U`.
        Defaults to `jax.nn.softplus`. Setting this to `jnp.exp` gives a
        Log-Euclidean parameterization.
    """

    eigenvalue_fn: Callable[[Array], Array]


class CholeskyOptions(TypedDict, total=False):
    """Options for `PSD` with `method="cholesky"`.

    - `diagonal_fn`: The positive function applied to the diagonal of `L`. Defaults to
        `jax.nn.softplus`. Setting this to `jnp.exp` gives a log-Cholesky
        parameterization.
    - `eps`: A non-negative constant added to the diagonal of the output, which bounds
        its smallest eigenvalue from below. Defaults to `0.0`.
    """

    diagonal_fn: Callable[[Array], Array]
    eps: float


class GramOptions(TypedDict, total=False):
    """Options for `PSD` with `method="gram"`.

    - `eps`: A non-negative constant added to the diagonal of the output, which bounds
        its smallest eigenvalue from below. The output is only SPD if `eps > 0`.
        Defaults to `0.0`.
    """

    eps: float


class PSD(eqx.Module):
    """Symmetric positive-semidefinite matrix network.

    This uses an MLP to predict the parameters of an `(n, n)` matrix, or of a batch of
    matrices that share features, and maps them to a PSD matrix with a chosen method.
    """

    shape: tuple[int, ...] = eqx.field(static=True)
    method: Literal["spectral", "cholesky", "gram"] = eqx.field(static=True)
    diagonal: tuple[bool, ...] = eqx.field(static=True)
    _fn: Callable[[Array], Array]
    _eps: float = eqx.field(static=True)
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        diag_size: int,
        width_size: int,
        depth: int,
        activation: Activation = _lipswish,
        final_activation: Activation = _identity,
        method: Literal["spectral", "cholesky", "gram"] = "spectral",
        options: SpectralOptions | CholeskyOptions | GramOptions | None = None,
        batch_shape: int | Sequence[int] = (),
        diagonal: bool | Sequence[bool] | ArrayLike = False,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a PSD network.

        Parameters
        ----------
        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `diag_size`: The diagonal size `n` of each predicted matrix.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
        - `activation`: The activation function after each hidden layer. Defaults to
            `lipswish`.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `method`: The method used to map the parameters of each matrix to the
            matrix. Defaults to `"spectral"`. This should be one of
            - `"spectral"`: The parameters form a symmetric matrix `U` (its diagonal,
                followed by its strictly-lower triangle in `jnp.tril_indices(n, -1)`
                order), and a positive function is applied to its eigenvalues. The
                output is SPD. This enables, e.g., geometrically-consistent SPD
                predictions.
            - `"cholesky"`: The parameters form a lower-triangular factor `L` (in
                `jnp.tril_indices(n)` order), a positive function is applied to its
                diagonal, and the output is `L @ L.T + eps * I`. Since the diagonal of
                `L` is positive, `L` is the unique Cholesky factor of `L @ L.T`, and
                the output is SPD.
            - `"gram"`: The parameters form an unconstrained lower-triangular factor
                `L` (in `jnp.tril_indices(n)` order), and the output is the Gram matrix
                `L @ L.T + eps * I`. The output is PSD, and is only SPD if `eps > 0`.
        - `options`: A dictionary of options for `method`. See `SpectralOptions`,
            `CholeskyOptions`, and `GramOptions`. Options that are not given take their
            default values. Defaults to `None`.
        - `batch_shape`: The shape of the batch of predicted matrices. The output from
            the module will be an array of shape `(*batch_shape, n, n)`. Defaults to
            `()`, i.e., a single matrix.
        - `diagonal`: Whether each predicted matrix is diagonal, i.e., has its
            off-diagonal parameters fixed at zero, so that only its `n` diagonal
            parameters are predicted. This is broadcast against `batch_shape`, so a
            `bool` applies to every matrix and a boolean array marks individual
            matrices. Defaults to `False`.
        - `key`: A `jax.random.key` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Raises
        ------
        `ValueError` if `method` is unknown, if `options` contains an option that
        `method` does not accept, if `eps` is negative, or if `diagonal` cannot be
        broadcast against `batch_shape`.

        Notes
        -----
        The MLP output holds the parameters of every full matrix, in row-major order
        over `batch_shape`, followed by those of every diagonal matrix. This layout
        can be used to initialize the network at a chosen matrix. For example, a
        single `"gram"` matrix can be started at the identity with
        ```python
        >>> identity = jnp.eye(n)[jnp.tril_indices(n)]
        >>> psd = affine_initialization(
        ...     psd, 0.0, identity, lambda m: (m.mlp.layers[-1].weight, m.mlp.layers[-1].bias)
        ... )
        ```
        This is recommended for `"gram"`, since a diagonal entry of `L` at zero is a
        saddle point of `L @ L.T`, and the sign of a diagonal entry cannot change
        without passing through it.

        The gradients of the eigendecomposition used by `"spectral"` are undefined
        when the predicted matrix has repeated eigenvalues, which can occur at
        initialization (e.g., when starting at the identity). `"cholesky"` and
        `"gram"` do not decompose the predicted matrix.
        """
        given: dict[str, Any] = dict(options or {})
        match method:
            case "spectral":
                valid = SpectralOptions.__optional_keys__
                fn, eps = given.get("eigenvalue_fn", _softplus), 0.0
            case "cholesky":
                valid = CholeskyOptions.__optional_keys__
                fn, eps = given.get("diagonal_fn", _softplus), given.get("eps", 0.0)
            case "gram":
                valid = GramOptions.__optional_keys__
                fn, eps = _identity, given.get("eps", 0.0)
            case _:
                raise ValueError(
                    f"Unknown method `{method!r}`. Expected one of `'spectral'`, "
                    "`'cholesky'`, or `'gram'`."
                )
        unknown = sorted(given.keys() - valid)
        if unknown:
            raise ValueError(
                f"Unknown options `{unknown}` for method `{method!r}`. The options for "
                f"this method are `{sorted(valid)}`."
            )
        if eps < 0:
            raise ValueError(f"`eps` must be non-negative. Got `{eps}`.")

        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        batch_shape = tuple(int(b) for b in batch_shape)
        try:
            mask = np.broadcast_to(np.asarray(diagonal, dtype=bool), batch_shape)
        except ValueError as e:
            raise ValueError(
                f"`diagonal` with shape `{np.shape(diagonal)}` cannot be broadcast "
                f"against `batch_shape` `{batch_shape}`."
            ) from e

        self.shape = (*batch_shape, diag_size, diag_size)
        self.method = method
        self.diagonal = tuple(bool(d) for d in mask.ravel())
        self._fn = fn
        self._eps = float(eps)

        num_diagonal = sum(self.diagonal)
        num_full = len(self.diagonal) - num_diagonal
        out_size = num_full * _tril_size(diag_size) + num_diagonal * diag_size
        self.mlp = eqx.nn.MLP(
            in_size, out_size, width_size, depth, activation, final_activation, key=key
        )

    def __call__(self, x: Array) -> Array:
        """Forward pass of the PSD network.

        Parameters
        ----------
        - `x`: A JAX array with shape `(in_size,)`.

        Returns
        -------
        A JAX array with shape `(*batch_shape, diag_size, diag_size)`.
        """
        n = self.shape[-1]
        if self.method == "spectral":

            def to_full(y: Array) -> Array:
                return _transform_eigenvalues(_vec_to_sym(y, n), self._fn)

            def to_diag(d: Array) -> Array:
                return jnp.diag(self._fn(d))

        else:

            def to_full(y: Array) -> Array:
                L = _transform_diagonal(_vec_to_tril(y, n), self._fn)
                return L @ L.T + self._eps * jnp.eye(n, dtype=L.dtype)

            def to_diag(d: Array) -> Array:
                return jnp.diag(jnp.square(self._fn(d)) + self._eps)

        full_idx = [i for i, diag in enumerate(self.diagonal) if not diag]
        diag_idx = [i for i, diag in enumerate(self.diagonal) if diag]
        y_full, y_diag = jnp.split(self.mlp(x), (len(full_idx) * _tril_size(n),))

        matrices = []
        if full_idx:
            matrices.append(jax.vmap(to_full)(y_full.reshape(len(full_idx), -1)))
        if diag_idx:
            matrices.append(jax.vmap(to_diag)(y_diag.reshape(len(diag_idx), n)))
        order = np.argsort(full_idx + diag_idx)
        return jnp.concatenate(matrices)[order].reshape(self.shape)


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
        activation: Activation = _lipswish,
        final_activation: Activation = _tanh,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a tamed MLP.

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


class WrapEnsemble[M: eqx.Module](eqx.Module):
    """Wraps a sequence of models and evaluates them together as an ensemble."""

    models: Sequence[M]

    def __post_init__(self):
        assert len(self.models) > 0, "`models` must contain at least one model."
        model_type = type(self.models[0])
        all_same = all(type(model) is model_type for model in self.models)
        assert all_same, "All models in `models` must be of the same type."

    def __call__(self, *args, **kwargs) -> PyTree[Array]:
        """Evaluate every member on the same input and stack the results.

        Parameters
        ----------
        - `*args`, `**kwargs`: Forwarded unchanged to each member's ``__call__``.

        Returns
        -------
        The members' outputs stacked along a leading axis of size `len(models)`.

        Notes
        -----
        If you need any statistics on the model predictions, it's cheaper to perform
        these statistics on the model outputs explicitly (which is why they aren't
        implemented internally). For example,
        ```python
        >>> outputs = ensemble(x)
        >>> WrapEnsemble.mean(outputs)
        ```
        """
        outputs = [model(*args, **kwargs) for model in self.models]  # type: ignore
        return jax.tree.map(lambda *out: jnp.stack(out), *outputs)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self):
        yield from self.models

    def __getitem__(self, index: int | slice) -> "M | WrapEnsemble[M]":
        if isinstance(index, slice):
            return WrapEnsemble(self.models[index])
        return self.models[index]

    @staticmethod
    def mean(outputs: PyTree[Array]) -> PyTree[Array]:
        """The mean of `outputs` over the ensemble axis.

        Parameters
        ----------
        - `outputs`: A stacked output from `__call__`, i.e. a pytree whose leaves have
            a leading axis of size `len(models)`.

        Returns
        -------
        A pytree matching a single member's output, each leaf averaged over the members.
        """
        return jax.tree.map(lambda out: jnp.mean(out, axis=0), outputs)

    @staticmethod
    def std(outputs: PyTree[Array]) -> PyTree[Array]:
        """The standard deviation of `outputs` over the ensemble axis.

        This is the usual ensemble-disagreement measure of epistemic uncertainty.

        Parameters
        ----------
        - `outputs`: A stacked output from `__call__`, i.e. a pytree whose leaves have
            a leading axis of size `len(models)`.

        Returns
        -------
        A pytree matching a single member's output, each leaf's spread over the members.
        """
        return jax.tree.map(lambda out: jnp.std(out, axis=0), outputs)

    @staticmethod
    def min(outputs: PyTree[Array]) -> PyTree[Array]:
        """The elementwise minimum of `outputs` over the ensemble axis.

        Parameters
        ----------
        - `outputs`: A stacked output from `__call__`, i.e. a pytree whose leaves have
            a leading axis of size `len(models)`.

        Returns
        -------
        A pytree matching a single member's output, each leaf reduced over the members.
        """
        return jax.tree.map(lambda out: jnp.min(out, axis=0), outputs)

    @staticmethod
    def max(outputs: PyTree[Array]) -> PyTree[Array]:
        """The elementwise maximum of `outputs` over the ensemble axis.

        Parameters
        ----------
        - `outputs`: A stacked output from `__call__`, i.e. a pytree whose leaves have
            a leading axis of size `len(models)`.

        Returns
        -------
        A pytree matching a single member's output, each leaf reduced over the members.
        """
        return jax.tree.map(lambda out: jnp.max(out, axis=0), outputs)

    def select(self, indices: Sequence[int]) -> "WrapEnsemble[M]":
        return WrapEnsemble([self.models[i] for i in indices])

    def subsample(
        self,
        outputs: PyTree[Array],
        num_members: int,
        *,
        key: PRNGKeyArray,
        replace: bool = False,
    ) -> PyTree[Array]:
        """Draw a random subset of the ensemble axis from an already-computed output.

        Parameters
        ----------
        - `outputs`: A stacked output from `__call__`, i.e. a pytree whose leaves have a
            leading axis of size `num_models`.
        - `num_members`: The number of members to keep.
        - `key`: A `jax.random.key` used to provide randomness for the subset draw.
            (Keyword only argument.)
        - `replace`: Whether to sample the members with replacement. Defaults to
            `False`.

        Returns
        -------
        A pytree matching `outputs` with each leaf's leading axis reduced to
        `num_members`.
        """
        indices = jr.choice(key, len(self.models), (num_members,), replace=replace)
        return jax.tree.map(lambda out: out[indices], outputs)


class WrapResidual(eqx.Module):
    """Composes a nominal value with a learned additive correction.

    The output is `nominal + correction(x)`, where `nominal` is either a fixed value
    (e.g., a standard matrix) or a callable evaluated on the input (e.g., a physics
    model). This is the usual residual-learning structure used for grey-box models,
    where the correction only has to capture what the nominal model gets wrong.

    Both the nominal value and the correction may be pytrees of arrays, in which case
    they are summed leaf-by-leaf.
    """

    nominal: PyTree[Array] | Callable[[Array], PyTree[Array]]
    correction: Callable[[Array], PyTree[Array]]
    freeze_nominal: bool = eqx.field(static=True)

    def __init__(
        self,
        nominal: PyTree[ArrayLike] | Callable[[Array], PyTree[Array]],
        correction: Callable[[Array], PyTree[Array]],
        freeze_nominal: bool = True,
    ):
        """Initialize a residual network.

        Parameters
        ----------
        - `nominal`: The nominal value. This can be either a pytree of arrays, which is
            returned as-is for every input, or a callable (e.g., an `eqx.Module`) that
            maps the input to the nominal value. Note that Python lists and tuples are
            treated as pytrees, so pass `jnp.array([...])` for an array nominal.
        - `correction`: The callable (e.g., an `eqx.nn.MLP`) that maps the input to an
            additive correction. Its output must have the same pytree structure and
            leaf shapes as the nominal value.
        - `freeze_nominal`: Whether to block gradients through the nominal value. Set
            this to `False` to jointly learn any parameters in `nominal` (e.g., for
            system identification). Defaults to `True`.

        Notes
        -----
        `freeze_nominal` only stops gradients. Optimizers that update parameters
        independently of their gradients (e.g., weight decay in `optax.adamw`) will
        still modify a frozen nominal unless it is also excluded with a `filter_spec`.

        It is often helpful for the model to start at the nominal value. For an
        `eqx.nn.MLP` correction, this can be done by zeroing its final layer:
        ```python
        >>> correction = affine_initialization(
        ...     mlp, 0.0, 0.0, lambda m: (m.layers[-1].weight, m.layers[-1].bias)
        ... )
        ```
        """
        if callable(nominal):
            self.nominal = nominal
        else:
            self.nominal = jax.tree.map(jnp.asarray, nominal)
        self.correction = correction
        self.freeze_nominal = freeze_nominal

    def __call__(self, x: Array) -> PyTree[Array]:
        """Forward pass of the residual network.

        Parameters
        ----------
        - `x`: A JAX array to provide as input to `nominal` (if callable) and
            `correction`.

        Returns
        -------
        The leaf-wise sum of the nominal value and the correction.

        Raises
        ------
        `ValueError` if the correction does not have exactly the same pytree structure
        and leaf shapes as the nominal value. Similar to diffrax, broadcasting is
        intentionally disallowed.
        """
        nominal = self.nominal(x) if callable(self.nominal) else self.nominal
        if self.freeze_nominal:
            nominal = jax.lax.stop_gradient(nominal)
        correction = self.correction(x)

        same_structure = jax.tree.structure(nominal) == jax.tree.structure(correction)
        if not same_structure or any(
            jnp.shape(n) != jnp.shape(c)
            for n, c in zip(jax.tree.leaves(nominal), jax.tree.leaves(correction))
        ):
            raise ValueError(
                "The output of `correction` must have the same pytree structure and "
                "leaf shapes as `nominal`. Got a nominal value of "
                f"`{eqx.tree_pformat(nominal)}` and a correction of "
                f"`{eqx.tree_pformat(correction)}`."
            )

        return jax.tree.map(jnp.add, nominal, correction)


class WrapPSDResidual(eqx.Module):
    """Composes a nominal PSD matrix with a learned PSD-preserving correction.

    The correction is composed with the nominal matrix `M₀` with a chosen method. With
    every method, the output is PSD for any correction, the correction can both
    increase and decrease the nominal matrix, and a zero correction returns the nominal
    matrix exactly.
    """

    nominal: Array | Callable[[Array], Array]
    correction: Callable[[Array], Array]
    method: Literal["cholesky", "congruence"] = eqx.field(static=True)
    freeze_nominal: bool = eqx.field(static=True)

    def __init__(
        self,
        nominal: ArrayLike | Callable[[Array], Array],
        correction: Callable[[Array], Array],
        method: Literal["cholesky", "congruence"] = "congruence",
        freeze_nominal: bool = True,
    ):
        """Initialize a PSD residual network.

        Parameters
        ----------
        - `nominal`: The nominal PSD matrix, with shape `(*batch, n, n)`. This can be
            either an array, which is used for every input, or a callable (e.g., an
            `eqx.Module`) that maps the input to the nominal matrix. A leading batch
            shape can be used to compose a stack of matrices (e.g., the outputs of a
            `PSD` with a `batch_shape`).
        - `correction`: The callable (e.g., an `eqx.nn.MLP`) that maps the input to the
            correction, with the shape required by `method`.
        - `method`: The method used to compose the correction with the nominal matrix.
            Defaults to `"congruence"`, which accepts any PSD nominal matrix. This
            should be one of
            - `"cholesky"`: The correction is added to the log-Cholesky parameters of
                `M₀`. That is, given the Cholesky factor `L₀` of `M₀`, the correction
                is added to the lower triangle of `L₀` (in `jnp.tril_indices(n)` order)
                with its diagonal replaced by its logarithm, and the output is
                `L @ L.T`, where `L` is the corrected factor with its diagonal mapped
                back through `exp`. The correction has shape
                `(*batch, n * (n + 1) / 2)`. `M₀` must be SPD, and the output is SPD.
                This corresponds to the log-Cholesky metric on SPD matrices
                ("Riemannian Geometry of Symmetric Positive Definite Matrices via
                Cholesky Decomposition", Zhenhua Lin), and matches `PSD` with
                `method="cholesky"` and a `jnp.exp` `diagonal_fn` when `M₀ = I`.
            - `"congruence"`: The output is `A @ M₀ @ A.T + D @ D.T`, where
                `A = I + ΔA` and `D` are lower-triangular matrices. The correction has
                shape `(*batch, n * (n + 1))`, where the first half of the entries are
                `ΔA` and the second half are `D`, each in `jnp.tril_indices(n)` order.
                `A` can reshape or shrink `M₀`, while `D @ D.T` can add to it in any
                direction (including the null space of `M₀`). Since `M₀` is never
                factorized, it may be singular (e.g., a drag matrix that is zero on
                some degrees of freedom), and the output is PSD.
        - `freeze_nominal`: Whether to block gradients through the nominal matrix. Set
            this to `False` to jointly learn any parameters in `nominal` (e.g., for
            system identification). Defaults to `True`.

        Raises
        ------
        `ValueError` if `method` is unknown.

        Notes
        -----
        For `"cholesky"`, the nominal matrix must be SPD. Its Cholesky factorization is
        not checked, and will produce NaNs otherwise. If the nominal matrix is learned,
        parameterize it so that it remains SPD (e.g., with a callable), rather than
        learning the raw matrix entries. Unlike a log-Euclidean residual, this does not
        require an eigendecomposition, whose gradients are undefined when the composed
        matrix has repeated eigenvalues (e.g., a diagonal nominal with repeated entries
        and a zero-initialized correction).

        For `"congruence"`, do not initialize the correction to exactly zero. The
        gradient of `D @ D.T` with respect to `D` vanishes at `D = 0`, so a
        zero-initialized `D` never learns, and the output can then never leave the
        range of the nominal matrix. Instead, scale the final layer down so that the
        model starts close to the nominal matrix:
        ```python
        >>> correction = affine_initialization(
        ...     mlp, 1e-2, 0.0, lambda m: (m.layers[-1].weight, m.layers[-1].bias)
        ... )
        ```

        See `WrapResidual` for notes on freezing the nominal value and on initializing
        the correction to zero.
        """
        if method not in ("cholesky", "congruence"):
            raise ValueError(
                f"Unknown method `{method!r}`. Expected one of `'cholesky'` or "
                "`'congruence'`."
            )
        self.nominal = nominal if callable(nominal) else jnp.asarray(nominal)
        self.correction = correction
        self.method = method
        self.freeze_nominal = freeze_nominal

    def __call__(self, x: Array) -> Array:
        """Forward pass of the PSD residual network.

        Parameters
        ----------
        - `x`: A JAX array to provide as input to `nominal` (if callable) and
            `correction`.

        Returns
        -------
        A JAX array with the same shape as the nominal matrix, `(*batch, n, n)`.

        Raises
        ------
        `ValueError` if the nominal value is not a (stack of) square matrices, or if
        the correction does not have the shape required by `method`.
        """
        nominal = self.nominal(x) if callable(self.nominal) else self.nominal
        if self.freeze_nominal:
            nominal = jax.lax.stop_gradient(nominal)
        correction = self.correction(x)

        shape = jnp.shape(nominal)
        if len(shape) < 2 or shape[-1] != shape[-2]:
            raise ValueError(
                "`nominal` must be a square matrix or a stack of square matrices with "
                f"shape `(*batch, n, n)`. Got a nominal value with shape `{shape}`."
            )
        n = shape[-1]

        if self.method == "cholesky":
            size = _tril_size(n)

            def compose(M0: Array, delta: Array) -> Array:
                L0 = jnp.linalg.cholesky(M0)
                L = _transform_diagonal(L0, jnp.log) + _vec_to_tril(delta, n)
                L = _transform_diagonal(L, jnp.exp)
                return L @ L.T

        else:
            size = 2 * _tril_size(n)

            def compose(M0: Array, delta: Array) -> Array:
                delta_A, delta_D = jnp.split(delta, 2)
                A = jnp.eye(n, dtype=delta.dtype) + _vec_to_tril(delta_A, n)
                D = _vec_to_tril(delta_D, n)
                return A @ M0 @ A.T + D @ D.T

        expected = (*shape[:-2], size)
        if jnp.shape(correction) != expected:
            raise ValueError(
                f"The output of `correction` must have shape `{expected}` for a "
                f"nominal matrix with shape `{shape}` and method `{self.method!r}`. Got "
                f"a correction with shape `{jnp.shape(correction)}`."
            )

        compose = jnp.vectorize(compose, signature="(n,n),(m)->(n,n)")
        return compose(nominal, correction)
