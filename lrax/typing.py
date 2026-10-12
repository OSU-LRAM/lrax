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

import optax
from jaxtyping import Array, ArrayLike, PyTree, Scalar, ScalarLike, Shaped

type Metrics = dict[str, ScalarLike]
type Optimizer = optax.GradientTransformation | optax.GradientTransformationExtraArgs
type Optimizers = PyTree[Optimizer]

# takes either (model, batch, keys) or, when `filter_spec` is set, (diff_model,
# static_model, batch, keys); returns ((loss, metrics), grads)
type LossFn = Callable[..., tuple[tuple[ScalarLike, Metrics], PyTree]]

type Activation = Callable[[Array], Array]

type Point = PyTree[Shaped[ArrayLike, "?*s"], "S"]  # type: ignore
type DistanceFn = Callable[[Point, Point], Array]
type Kernel = Callable[[Point, Point], Scalar]
