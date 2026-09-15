#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NumPy 2.x compatibility shims for the read-only GSFC-GRASP-Python-Interface repo.

Two independent NumPy-2 breakages are patched here (see apply()):
  1. np.trapz was removed  -> restore as an alias of np.trapezoid
  2. miscFunctions.loguniform passes a 1-element ndarray to math.log

NumPy 2.0 REMOVED ``np.trapz`` (it was renamed ``np.trapezoid``).  The interface
repo still calls ``np.trapz`` in 9 places -- runGRASP.py (parseOutAerosol,
parsePhaseMatrix, rsltDictTools), miscFunctions.py and mieFunctions.py -- so on
NumPy >= 2 a retrieval dies while PARSING GRASP's output, after the forward
calculation has already succeeded:

    AttributeError: module 'numpy' has no attribute 'trapz'

GSFC-GRASP-Python-Interface is a read-only dependency (see
../ALTERING_THE_MEASUREMENT_ERROR_MODEL.md), so instead of patching it we restore
the alias here.  ``np.trapezoid`` is the same function under the new name and is
signature-compatible for every call site above (all use the ``f(y, x)`` form).

Import this BEFORE anything that pulls in runGRASP::

    import err_sim.np_compat   # noqa: F401  -- must precede runGRASP import

Alternative if you would rather not carry a shim: pin ``numpy<2`` in the conda
env.  That was not done here because the env's pandas expects NumPy 2.
"""

import numpy as np


def _patch_loguniform():
    """Make ``miscFunctions.loguniform`` NumPy-2 safe.

    NumPy 2.0 removed the implicit conversion of size-1 (but ndim>0) arrays to
    Python scalars.  ``loguniform`` does::

        return lo ** ((((log(hi) / log(lo)) - 1) * random()) + 1)

    with ``from math import log``.  ``scrambleInitialGuess`` reaches it with lo/hi
    as 1-element ndarrays (a single-wavelength ``initial_guess`` min/max), which on
    NumPy 1.x converted silently and now raises::

        TypeError: only 0-dimensional arrays can be converted to Python scalars

    The replacement below reproduces the original semantics exactly -- including
    the "spectrally flat" behaviour for len>1 (one draw, reused at every
    wavelength, returned as a list) and the return TYPE of each branch -- but
    takes logs of Python floats.  ``random()`` is still called exactly once per
    invocation, so RNG consumption is unchanged.
    """
    import math
    import random as _random
    import numpy as _np
    try:
        import miscFunctions as mf
    except ImportError:
        return False   # interface repo not on sys.path yet; nothing to patch

    if getattr(mf.loguniform, '_errsim_patched', False):
        return True

    def loguniform(lo, hi):
        if isinstance(lo, (list, _np.ndarray)):
            if len(lo) > 1:
                expo = (((math.log(float(hi[0])) / math.log(float(lo[0]))) - 1)
                        * _random.random()) + 1
                return [float(lo[0]) ** expo] * len(lo)      # spectrally flat, as before
            expo = (((math.log(float(_np.ravel(hi)[0])) / math.log(float(_np.ravel(lo)[0]))) - 1)
                    * _random.random()) + 1
            return lo ** expo                                # keeps ndarray/list-pow type
        expo = (((math.log(hi) / math.log(lo)) - 1) * _random.random()) + 1
        return lo ** expo

    loguniform.__doc__ = mf.loguniform.__doc__
    loguniform._errsim_patched = True
    mf.loguniform = loguniform
    return True


def apply():
    """Apply all NumPy-2 shims. Idempotent; no-ops on NumPy 1.x."""
    if not hasattr(np, 'trapz'):
        if not hasattr(np, 'trapezoid'):
            raise ImportError('numpy has neither trapz nor trapezoid (version %s)'
                              % np.__version__)
        np.trapz = np.trapezoid
    _patch_loguniform()
    return np.trapz


apply()
