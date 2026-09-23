#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NumPy version-compatibility shims, so the same checkout runs on NumPy 1.x and 2.x.

Patched here (see apply()):
  1. the np.trapz / np.trapezoid split -- aliased in WHICHEVER direction is missing
  2. miscFunctions.loguniform passes a 1-element ndarray to math.log

THE TRAPEZOID SPLIT.  NumPy 2.0 renamed ``np.trapz`` to ``np.trapezoid`` and removed
the old name.  The two repos in play were written against different NumPy majors:

    GSFC-Retrieval-Simulators    uses np.trapezoid  (simulateRetrieval.py,
                                                     MADCAP_functions.py,
                                                     readOSSEnetCDF.py, ...)
    GSFC-GRASP-Python-Interface  used np.trapz until its py3.12 update (bcb2d1d,
                                 merged 2026-09), which migrated all 9 call sites in
                                 runGRASP.py / miscFunctions.py / mieFunctions.py /
                                 polarVectorPlot.py to np.trapezoid.

  * On NumPy >= 2 a PRE-bcb2d1d interface repo dies while PARSING GRASP's output,
    after the forward calculation has already succeeded:
        AttributeError: module 'numpy' has no attribute 'trapz'
  * On NumPy < 2 the simulator repo dies in simulateRetrieval._addReffMode:
        AttributeError: module 'numpy' has no attribute 'trapezoid'

Both were observed for real -- the first on a NumPy 2.5 laptop, the second on a
cluster whose GEOSpyD stack ships NumPy 1.x.

Now that BOTH repos spell it ``np.trapezoid``, the NumPy 2 direction of this shim is
dead weight on an up-to-date checkout, but the NumPy 1.x direction became MORE
load-bearing: on the cluster there is no longer any ``trapz`` call site to fall back
on, so every integration in both repos depends on the alias.  Keep both directions --
the interface repo is a read-only dependency (see
../ALTERING_THE_MEASUREMENT_ERROR_MODEL.md) that may be checked out at any commit.  They are the same function and are
signature-compatible for every call site here (all use the ``f(y, x)`` form).

Import this BEFORE anything that pulls in runGRASP or simulateRetrieval::

    import err_sim.np_compat   # noqa: F401  -- must precede those imports

Pinning a NumPy version instead would not fix this: no single major satisfies both
repos.
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
    """Apply the NumPy compatibility shims. Idempotent; safe on 1.x and 2.x.

    Aliases whichever of trapz/trapezoid the installed NumPy is missing, so both
    repos' spellings resolve regardless of major version.
    """
    hasTrapz = hasattr(np, 'trapz')
    hasTrapezoid = hasattr(np, 'trapezoid')
    if not hasTrapz and not hasTrapezoid:
        raise ImportError('numpy %s has neither trapz nor trapezoid' % np.__version__)
    if not hasTrapz:                    # NumPy >= 2: restore the old name
        np.trapz = np.trapezoid
    if not hasTrapezoid:                # NumPy < 2: provide the new name
        np.trapezoid = np.trapz
    _patch_loguniform()
    return np.trapz


apply()
