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

from pathlib import Path
from typing import Optional

import mujoco
from jaxtyping import Array
from mujoco import mjx
from mujoco.mjx import Data, Model


def load_model(path: Path) -> Model:
    """Load an `mjx.Model` from a configuration file.

    Parameters
    ----------
    - `path`: The full path to the `xml` configuration file.

    Returns
    -------
    The loaded `mjx.Model`.
    """
    model = mujoco.MjModel.from_xml_path(path)  # type: ignore
    return mjx.put_model(model)


def reset(model: Model, qpos: Optional[Array] = None) -> Data:
    """Create an initial simulation state.

    Parameters
    ----------
    - `model`: The MJX model to create the state for.
    - `qpos`: The initial configuration.

    Returns
    -------
    The initial simulation state.
    """
    data = mjx.make_data(model)

    if qpos is not None:
        data = data.replace(qpos=qpos)

    return mjx.forward(model, data)


def step(model: Model, data: Data, control: Array) -> Data:
    """Apply a control to the system and advance the simulation by one timestep.

    Parameters
    ----------
    - `model`: The MJX model being simulated.
    - `data`: The current simulation state.
    - `control`: A JAX array representing the control input to apply to the system.

    Returns
    -------
    The next simulation state.
    """
    data = data.replace(ctrl=control)
    return mjx.step(model, data)
