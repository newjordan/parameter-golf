"""NumPy casts that stay valid on NumPy 2.

``np.array(values, dtype=new_dtype, copy=False)`` raises ``ValueError`` when
the cast needs a copy. ``ndarray.astype(..., copy=False)`` still returns a new
array in that case. :func:`cast_array` returns a C-contiguous array and copies
only when the dtype changes.
"""

from __future__ import annotations

import numpy as np


def cast_array(values, dtype) -> np.ndarray:
    array = np.asarray(values)
    target = np.dtype(dtype)
    if array.dtype == target:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array.astype(target, copy=True))
