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

from glob import glob

import matplotlib.font_manager


def load_msttcorefonts(font_dir: str = "/usr/share/fonts/truetype/msttcorefonts/"):
    """Load the Microsoft fonts to matplotlib.

    The Microsoft fonts can be used in your matplotlib figures after calling this
    function. For example,
    ```python
    >>> load_msttcorefonts()
    >>> plt.rcParams["font.family"] = "sans-serif"
    >>> plt.rcParams["font.sans-serif"] = "Times New Roman"
    >>> plt.rcParams["pdf.fonttype"] = 42
    ```
    """
    for font in glob(font_dir + "*.ttf"):
        matplotlib.font_manager.fontManager.addfont(font)
