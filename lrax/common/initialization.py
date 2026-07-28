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

import equinox as eqx
import jax.numpy as jnp


def affine_initialization(
    model: eqx.Module, scale: float, shift: float, where: Callable
):
    """Apply an affine transformation to the chosen `Linear` layer(s) in a model.

    Parameters
    ----------
    - `model`: The model to initialize.
    - `scale`: Rescaling applied to the selected weights in `model`.
    - `shift`: Translation applied to the selected biases in `model`.
    - `where`: Function used to get the leaves that should be modified.

    Returns
    -------
    The transformed model.
    """

    def _init_fn(_wb):
        w, b = _wb
        return (scale * w, jnp.full(b.shape, shift))

    new_leaves = _init_fn(where(model))
    new_model = eqx.tree_at(where, model, new_leaves)
    return new_model
