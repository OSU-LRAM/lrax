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

import abc
from typing import override

import jax.numpy as jnp
from jaxtyping import ScalarLike

from .._custom_types import Metrics


class StopTraining(Exception):
    """Raised by a callback to end training."""


class Callback(abc.ABC):
    """Base class for a callback invoked after every training step.

    Subclasses implement `__call__`, inspecting the step's loss and metrics to decide
    whether to act, e.g., raising `StopTraining` to end training.
    """

    @abc.abstractmethod
    def __call__(self, loss: ScalarLike, metrics: Metrics, step: ScalarLike):
        """Run the callback for a single training step.

        Parameters
        ----------
        - `loss`: The scalar loss value for this step.
        - `metrics`: A dictionary of scalar metrics logged for this step.
        - `step`: The current training step.
        """
        raise NotImplementedError


class StopOnNaN(Callback):
    """Stops training as soon as a NaN loss value is encountered."""

    @override
    def __call__(self, loss: ScalarLike, metrics: Metrics, step: ScalarLike):
        """Check the step's loss for a NaN value.

        Parameters
        ----------
        - `loss`: The scalar loss value for this step.
        - `metrics`: Unused.
        - `step`: Unused.

        Raises
        ------
        `StopTraining` if `loss` is NaN.
        """
        del metrics, step
        if jnp.isnan(jnp.asarray(loss)):
            raise StopTraining("NaN loss value detected.")
