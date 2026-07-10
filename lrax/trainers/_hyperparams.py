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

from dataclasses import make_dataclass
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def load_params_from_yaml(fp: str) -> "DataclassInstance":
    """Load training parameters from a YAML file.

    Parameters
    ----------
    - `fp`: The file path to the YAML file.

    Returns
    -------
    A dataclass instance whose members are the declared hyperparameters.
    """
    with open(fp) as f:
        params = yaml.safe_load(f)["hyperparameters"]
    return make_dataclass("Params", ((k, type(v)) for k, v in params.items()))(**params)
