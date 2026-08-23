"""Backward-compat shim: `import pysmc` -> `import pyvsmc`.

This package is deprecated. Please use `pyvsmc`:

    pip install pyvsmc
    import pyvsmc as smc

This shim will be removed in 1.0.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "pysmc is deprecated, use pyvsmc instead: pip install pyvsmc; import pyvsmc",
    DeprecationWarning,
    stacklevel=2,
)

from pyvsmc import *  # noqa: F401,F403
from pyvsmc import (
    __version__,  # noqa: F401
    add_smc_columns,  # noqa: F401
)
