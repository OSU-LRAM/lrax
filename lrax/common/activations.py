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

import jax.nn as jnn
from jaxtyping import Array


def lipswish(x: Array) -> Array:
    """LipSwish activation function.

    LipSwish has been shown to perform well when applied to neural SDEs. Refer to the
    following papers for additional information:

    - "Residual Flows for Invertible Generative Modeling" (https://arxiv.org/abs/1906.02735)
    - "Efficient and Accurate Gradients for Neural SDEs" (https://arxiv.org/pdf/2105.13493)

    Parameters
    ----------
        - `x`: The input tensor.

    Returns
    -------
        The LipSwish activation.
    """
    return 0.909 * jnn.silu(x)
