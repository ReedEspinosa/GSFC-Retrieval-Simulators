#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom measurement error model(s) for the err_sim experiment.

All new error-model code for this experiment lives in the ``err_sim`` directory.
The intent is to go beyond simply scaling the 1-sigma uncertainty on a Stokes
vector: here we fold in polarimeter *calibration* uncertainty derived from the
nsienkie-cal-uncertainty Monte Carlo model.

Dispatch path (see architectureMap.py):
    returnPixel()  -- 'harperrsim'   sets errStr='errsim01' (Path 1)
                      'harperrsimmc' sets errStr='errsim02' (Path 2)
    addError()     -- the 'errsim*' branch calls customErrModel() here

customErrModel() routes on the numeric id of the errStr:
    errsim01  -> _errsim_analytic()     (Path 1: analytical error propagation)
    errsim02  -> _errsim_montecarlo()   (Path 2: Monte Carlo sensor-space noise)

Shared call signature / return convention (drop-in for addError models):
    args:  measNm(str errsimNN), l(int wl-index), rsltFwd(dict truth), concase,
           orbit, lidErrDir, verbose
    returns: np.r_[I, Q, U]  (noised, ascending msTyp order [41,42,43])

------------------------------------------------------------------------------
CONFIGURE THE CALIBRATION-SIM INPUTS HERE (module header)
------------------------------------------------------------------------------
Set the three paths just below. They are read at load time (lazy on the first
pixel, or immediately at import if EAGER_LOAD=True). Because the store is a
module-level singleton, importing this module and pointing these at real files is
all that is needed -- no changes to the main retrieval simulator.

    CAL_MATRIX_H5_PATH -- run .h5 with characteristic + calibration matrices
    COV_MATRIX_PATH    -- SEPARATE file (CSV) holding the 9x9 covariance of vec(C).
                          Loaded explicitly, NOT recomputed from the MC data -- the
                          cal sim uses a custom covariance solver (centered on the
                          true mean mu = vec(inv(A))), and you may want to swap in
                          alternative covariances for test cases.

Any path left None uses a DUMMY stand-in until real files are configured.

------------------------------------------------------------------------------
PER-WAVELENGTH INSTRUMENT SELECTION
------------------------------------------------------------------------------
Each wavelength channel of a polarimeter is treated as an INDEPENDENT instrument
with its own characteristic matrix A. Rationale: the per-channel optics (polarizer
transmission/extinction, compensator retardance, absorber) are spectrally distinct,
so the cal sim's instrument-to-instrument parameter spread is a reasonable proxy
for one instrument's channel-to-channel variation. (Picture a spectral beam
splitter feeding a separate polarizer/sensor per wavelength.) The monotonic-with-
wavelength trend of real optics is neglected here.

Mechanics: customErrModel() is invoked once per (pixel, wavelength). EVERY call
draws a fresh instrument index (WITHOUT replacement) -> a new A per wavelength.
All view-angles of that wavelength share it. Path 2 additionally draws one random
calibration matrix from that same instrument.

------------------------------------------------------------------------------
IN-MEMORY MATRIX STORE (load once per process, not per pixel)
------------------------------------------------------------------------------
The calibration-sim products are loaded ONCE into a module-level singleton
(_MatrixStore) and held in RAM; every pixel then just indexes numpy arrays.
The store persists for the process lifetime, so it is never reloaded per pixel.
"""

import numpy as np
import os
import re
import warnings

# =============================================================================
# >>> CONFIGURE INPUT PATHS HERE <<<
# =============================================================================
# Default to the sibling nsienkie-cal-uncertainty repo under npp_projects.
_ERRSIM_DIR = os.path.dirname(os.path.abspath(__file__))              # .../GSFC-Retrieval-Simulators/err_sim
_NPP_DIR    = os.path.dirname(os.path.dirname(_ERRSIM_DIR))           # .../npp_projects
_CALSIM_DIR = os.path.join(_NPP_DIR, 'nsienkie-cal-uncertainty')

CAL_MATRIX_H5_PATH = os.path.join(_CALSIM_DIR, 'stor_data/eval_test/2026-07-27T19:53:18.h5')   # None -> dummy pool
COV_MATRIX_PATH    = os.path.join(_CALSIM_DIR, 'eval_results/output/csv/covariance_matrix_radcal.csv')  # None -> dummy cov
EAGER_LOAD = False          # True -> build the store at import; False -> on first pixel

# --- sensor-intensity noise model (Path 2): see ported instr_err() below ---
SENSOR_BIN_FACTOR = 10      # detector binning factor (cal sim's bin_size); radcal eval used 10

# 9x9 covariance is indexed k = 3*i + j over vec(C), where C[i,j] maps
# sensor intensity j -> Stokes component i ([I,Q,U] = C @ [s1,s2,s3]).
N_STOKES = 3   # [I, Q, U]  (V/S3 dropped, matching the cal sim)
N_SENSOR = 3   # 3-sensor division-of-amplitude polarimeter

# --- dummy-pool controls (used only while a path above is None) ---
_DUMMY_N_INSTR = 4096      # size of the synthetic instrument pool
_DUMMY_N_CAL = 500         # calibration trials per instrument (Path 2)
_DUMMY_SEED = 0            # deterministic dummy pool + deck shuffle

# module-level per-call state and singleton store
_CURRENT_INSTRUMENT_IDX = None   # characteristic-matrix index, drawn fresh each wavelength
_STORE = None

# side channel for Path 1 sigmas (until we decide how they feed downstream)
LAST_ANALYTIC_SIGMAS = None


# =============================================================================
# In-memory matrix store  (load once; index per pixel)
# =============================================================================
class _MatrixStore:
    """Holds the calibration-sim matrices in RAM and hands out instruments
    WITHOUT replacement via a shuffled deck.

    char_mats : (N_instr, 3, 3)          true characteristic matrices A (Stokes -> sensor)
    cal_mats  : (N_instr, 3, 3, N_cal)   fitted calibration matrices C (sensor -> Stokes)
    cov_C     : (9, 9)                    covariance of vec(C), index k = 3i+j (loaded, not computed)
    """

    def __init__(self, h5_path=None, cov_path=None, seed=_DUMMY_SEED, reshuffle_on_exhaust=True):
        self.h5_path = h5_path
        self.cov_path = cov_path
        self._rng = np.random.default_rng(seed)   # independent of global np.random / fixRndmSeed
        self.reshuffle_on_exhaust = reshuffle_on_exhaust
        self._warned_exhaust = False

        # characteristic + calibration matrices
        if h5_path is not None:
            self._load_matrices_from_hdf5(h5_path)
        else:
            self._load_dummy_matrices()

        # covariance loaded EXPLICITLY from its own path (never recomputed here)
        self.cov_C = _load_covariance_matrix(cov_path)

        self.n_instr = self.char_mats.shape[0]
        self._deck = list(self._rng.permutation(self.n_instr))   # draw order, no repeats

    # ---- matrix loaders ----
    def _load_matrices_from_hdf5(self, h5_path):
        """Load the cal-sim run once into RAM:
            results/characteristic_matrices (N_instr,3,4) -> drop V -> char_mats (N_instr,3,3)
            results/calibration_matrices    (N_instr,3,3,N_cal) -> cal_mats
        Covariance is NOT read here -- it comes from cov_path via _load_covariance_matrix()."""
        import h5py
        if not os.path.isfile(h5_path):
            raise FileNotFoundError("CAL_MATRIX_H5_PATH does not exist: %s" % h5_path)
        with h5py.File(h5_path, 'r') as f:
            # drop the V column (last of the 4 Stokes weights) -> (N_instr, 3, 3)
            self.char_mats = np.array(f['results/characteristic_matrices'][:, :, :N_STOKES])
            self.cal_mats  = np.array(f['results/calibration_matrices'][:])   # (N_instr,3,3,N_cal)

    def _load_dummy_matrices(self):
        """Synthetic pool: perturb the ideal 0/45/90 characteristic matrix per
        instrument (~2% spread), and a plausible calibration ensemble. DUMMY."""
        A0 = _ideal_characteristic_matrix()
        rng = np.random.default_rng(_DUMMY_SEED + 1)
        pert = rng.normal(scale=0.02, size=(_DUMMY_N_INSTR, 3, 3)) * np.abs(A0)
        self.char_mats = A0[None, :, :] + pert                       # (N_instr,3,3)
        invA = np.linalg.inv(self.char_mats)
        calscatter = rng.normal(scale=0.01, size=(_DUMMY_N_INSTR, 3, 3, _DUMMY_N_CAL))
        self.cal_mats = invA[..., None] + calscatter                 # (N_instr,3,3,N_cal)

    # ---- per-pixel draw (without replacement) ----
    def draw_index(self):
        """Pop the next instrument index; no repeat until the pool is exhausted."""
        if not self._deck:
            if not self.reshuffle_on_exhaust:
                raise RuntimeError(
                    "err_sim matrix pool exhausted (%d instruments); increase N_instr in the "
                    "calibration sim so it exceeds the number of pixels." % self.n_instr)
            if not self._warned_exhaust:
                warnings.warn("err_sim: instrument pool (%d) smaller than #pixels; reshuffling "
                              "-- matrices will now be reused." % self.n_instr)
                self._warned_exhaust = True
            self._deck = list(self._rng.permutation(self.n_instr))
        return self._deck.pop()


_UNSET = object()   # sentinel: distinguish "arg omitted" from an explicit None


def init_store(h5_path=_UNSET, cov_path=_UNSET, seed=_DUMMY_SEED, reshuffle_on_exhaust=True):
    """(Re)load the matrix store and reset the per-pixel draw deck. Normally you do
    NOT need to call this -- setting the header paths + lazy load is enough. Provided
    for explicit control / test-case swaps.

    Omitting an argument uses the header path (CAL_MATRIX_H5_PATH / COV_MATRIX_PATH);
    pass an explicit None to force the corresponding DUMMY instead.
    """
    global _STORE, _CURRENT_INSTRUMENT_IDX
    if h5_path is _UNSET:
        h5_path = CAL_MATRIX_H5_PATH
    if cov_path is _UNSET:
        cov_path = COV_MATRIX_PATH
    _STORE = _MatrixStore(h5_path=h5_path, cov_path=cov_path, seed=seed,
                          reshuffle_on_exhaust=reshuffle_on_exhaust)
    _CURRENT_INSTRUMENT_IDX = None
    return _STORE


def get_store():
    """Return the singleton store, lazy-initializing from the header paths if needed."""
    global _STORE
    if _STORE is None:
        _STORE = _MatrixStore(h5_path=CAL_MATRIX_H5_PATH, cov_path=COV_MATRIX_PATH)
    return _STORE


# =============================================================================
# Covariance loading (explicit; swappable) -- NOT computed from the MC data
# =============================================================================
def _load_covariance_matrix(cov_path):
    """Load the 9x9 covariance of vec(C) from cov_path (CSV), or return a dummy.

    The real covariance comes from the calibration sim's custom covariance solver
    (centered on the true mean mu = vec(inv(A))). To try alternative covariances,
    just point COV_MATRIX_PATH at a different CSV -- no code change.
    """
    if cov_path is None:
        return _dummy_covariance()
    if not os.path.isfile(cov_path):
        raise FileNotFoundError("COV_MATRIX_PATH does not exist: %s" % cov_path)
    return _read_cov_csv(cov_path)


def _read_cov_csv(path):
    """Read a 9x9 covariance CSV into a numpy array.

    Handles the exact format written by the cal sim's covariance_matrix.py -- a
    csv.writer grid with a header row ['' , C[0,0], ...] and a leading row-label
    column (labels quoted because 'C[i,j]' contains a comma) -- as well as a plain
    numeric 9x9 CSV. Uses the stdlib csv module so quoted commas parse correctly.
    """
    import csv
    with open(path, newline='') as f:
        rows = [r for r in csv.reader(f) if r]

    def _as_float_grid(grid):
        try:
            return np.array(grid, dtype=float)
        except (ValueError, TypeError):
            return None

    arr = _as_float_grid(rows)                       # plain numeric 9x9
    if arr is None:                                  # labeled: drop header row + label column
        arr = _as_float_grid([r[1:] for r in rows[1:]])
    if arr is None or arr.shape != (N_STOKES * N_SENSOR, N_STOKES * N_SENSOR):
        raise ValueError("Expected a 9x9 covariance in %s, got %s"
                         % (path, None if arr is None else arr.shape))
    return arr


def _dummy_covariance():
    """DUMMY 9x9 covariance of vec(C) until real CSVs are configured.

    Deterministic, diagonal (independent elements), ~ (0.02)^2 variance on each of
    the 9 calibration-matrix elements -> gives Path 1 a nonzero, plausible sigma.
    Replace by pointing COV_MATRIX_PATH at a real file.
    """
    return np.eye(N_STOKES * N_SENSOR) * (0.02 ** 2)


# =============================================================================
# Dispatcher
# =============================================================================
def customErrModel(measNm, l, rsltFwd, concase=None, orbit=None, lidErrDir=None, verbose=False):
    """Route an 'errsim<NN>' call. Draws a fresh instrument EVERY call, i.e. each
    wavelength channel is an independent instrument (unique characteristic matrix)."""
    global _CURRENT_INSTRUMENT_IDX
    _CURRENT_INSTRUMENT_IDX = get_store().draw_index()   # per-wavelength instrument
    if verbose:
        print('[err_sim] wl-index %d -> instrument index %d' % (l, _CURRENT_INSTRUMENT_IDX))

    mtch = re.match(r'^([A-Za-z]+)([0-9]+)$', measNm)
    errId = int(mtch.group(2)) if mtch else -1
    if errId == 1:
        return _errsim_analytic(measNm, l, rsltFwd, concase=concase, orbit=orbit,
                                lidErrDir=lidErrDir, verbose=verbose)
    elif errId == 2:
        return _errsim_montecarlo(measNm, l, rsltFwd, concase=concase, orbit=orbit,
                                  lidErrDir=lidErrDir, verbose=verbose)
    else:
        raise ValueError("Unknown err_sim id in measNm=%r (expected errsim01 or errsim02)" % measNm)


# =============================================================================
# Shared geometry / truth extraction
# =============================================================================
def _extract_truth_and_geometry(l, rsltFwd):
    """Pull truth Stokes (3,Nang) and per-view geometry out of rsltFwd at wl l."""
    stokes = np.vstack([rsltFwd['fit_I'][:, l], rsltFwd['fit_Q'][:, l], rsltFwd['fit_U'][:, l]])
    viewZen = rsltFwd['vis'][:, l]
    relAzim = rsltFwd['fis'][:, l]
    solZen  = rsltFwd['sza'][0, l]
    bandWvl = rsltFwd['lambda'][l]
    scatAng = np.degrees(np.arccos(
        -np.cos(np.radians(solZen)) * np.cos(np.radians(np.abs(viewZen)))
        + np.sin(np.radians(solZen)) * np.sin(np.radians(np.abs(viewZen))) * np.cos(np.radians(relAzim))))
    geom = dict(viewZen=viewZen, relAzim=relAzim, solZen=solZen, bandWvl=bandWvl, scatAng=scatAng)
    return stokes, geom


# =============================================================================
# Path 1 -- analytical error propagation
# =============================================================================
def _errsim_analytic(measNm, l, rsltFwd, concase=None, orbit=None, lidErrDir=None, verbose=False):
    """Path 1: propagate BOTH error sources into a Stokes covariance, take its 1-sigma,
    draw a Gaussian realization, return noised [I,Q,U].

        Cov(S) = C diag(sigma_I^2) C^T           (measurement noise; C = inv(A))
               + sum_ab I_a I_b Sigma[3i+a,3j+b] (calibration-matrix uncertainty)

    sigma_I is the per-sensor detector noise (instr_err) at each channel's intensity;
    Sigma is the loaded (swappable) 9x9 vec(C) covariance. This pixel's characteristic
    matrix A is used at all angles & wavelengths."""
    stokes, geom = _extract_truth_and_geometry(l, rsltFwd)
    Nang = stokes.shape[1]

    charMat = _load_characteristic_matrix(geom, l, verbose=verbose)   # (3,3) A for THIS pixel
    covC    = _load_calibration_covariance(geom, l, verbose=verbose)  # (9,9) loaded cov of vec(C)
    covC4   = covC.reshape(N_STOKES, N_SENSOR, N_STOKES, N_SENSOR)     # [i,a,j,b] = Sigma[3i+a,3j+b]
    invA    = np.linalg.inv(charMat)                                  # C = inv(A): sensor -> Stokes

    sigmaStokes = np.zeros((N_STOKES, Nang))
    sigmaDoLP   = np.zeros(Nang)
    noisyStokes = np.zeros((N_STOKES, Nang))
    for n in range(Nang):
        s = stokes[:, n]
        sensorInt = charMat @ s                                       # (3,) sensor intensities
        # (1) measurement-noise term: diagonal sensor covariance propagated through C=inv(A)
        sigI = instr_err(sensorInt, SENSOR_BIN_FACTOR)               # (3,) per-sensor 1-sigma (intensity-dependent)
        covFromI = invA @ np.diag(sigI**2) @ invA.T                  # (3,3)  C diag(sigma_I^2) C^T
        # (2) calibration-matrix uncertainty term: sum_ab I_a I_b Sigma[3i+a,3j+b]
        covFromC = np.einsum('a,b,iajb->ij', sensorInt, sensorInt, covC4)   # (3,3)
        covTot = covFromI + covFromC                                 # (3,3) total Stokes covariance
        sig = np.sqrt(np.clip(np.diag(covTot), 0, None))
        sigmaStokes[:, n] = sig
        sigmaDoLP[n] = _dolp_sigma(s, covTot)
        noisyStokes[:, n] = s + np.random.normal(size=N_STOKES) * sig

    global LAST_ANALYTIC_SIGMAS
    LAST_ANALYTIC_SIGMAS = dict(sigma_stokes=sigmaStokes, sigma_dolp=sigmaDoLP,
                                wavelength=geom['bandWvl'], instrument_idx=_CURRENT_INSTRUMENT_IDX)
    if verbose:
        print('[err_sim] analytic: l=%d wvl=%.3f Nang=%d instr=%s | mean sigma_I=%.3g sigma_DoLP=%.3g'
              % (l, geom['bandWvl'], Nang, _CURRENT_INSTRUMENT_IDX, sigmaStokes[0].mean(), sigmaDoLP.mean()))
    return np.r_[noisyStokes[0], noisyStokes[1], noisyStokes[2]]


def _dolp_sigma(stokes_vec, covStokes):
    """Delta-method 1-sigma of DoLP=sqrt(Q^2+U^2)/I given a 3x3 Stokes covariance."""
    I, Q, U = stokes_vec
    P = np.sqrt(Q**2 + U**2)
    if P <= 0 or I <= 0:
        return np.nan
    grad = np.array([-P / I**2, Q / (I * P), U / (I * P)])
    return np.sqrt(max(grad @ covStokes @ grad, 0.0))


# =============================================================================
# Path 2 -- Monte Carlo sensor-space noise + imperfect calibration
# =============================================================================
def _errsim_montecarlo(measNm, l, rsltFwd, concase=None, orbit=None, lidErrDir=None, verbose=False):
    """Path 2: truth Stokes -> true sensor intensities -> view-angle-dependent
    sensor noise -> recombine with a randomly-drawn calibration matrix (this pixel's
    instrument). Returns reconstructed (biased+noisy) [I,Q,U]. Wavelength dependence
    intentionally deferred here."""
    stokes, geom = _extract_truth_and_geometry(l, rsltFwd)

    charMatTrue = _load_characteristic_matrix(geom, l, verbose=verbose)   # (3,3) A_true (this pixel)
    calMat      = _load_calibration_matrix(geom, l, verbose=verbose)      # (3,3) C draw (same instrument)

    sensorInt   = charMatTrue @ stokes                                    # (3, Nang) true sensor intensities
    sigSensor   = _sensor_noise_sigma(sensorInt)                          # (3, Nang) counts-based 1-sigma
    noisySensor = sensorInt + np.random.normal(size=sensorInt.shape) * sigSensor
    outStokes   = calMat @ noisySensor                                    # (3, Nang) recombine w/ imperfect C

    if verbose:
        print('[err_sim] montecarlo: l=%d wvl=%.3f Nang=%d instr=%s | mean sigma_sensor=%.3g'
              % (l, geom['bandWvl'], stokes.shape[1], _CURRENT_INSTRUMENT_IDX, sigSensor.mean()))
    return np.r_[outStokes[0], outStokes[1], outStokes[2]]


# =============================================================================
# Matrix loaders -- backed by the in-memory store + per-pixel instrument index
# =============================================================================
def _ideal_characteristic_matrix():
    """Ideal 3-sensor (0/45/90 deg analyzer) characteristic matrix, [I,Q,U] cols.
    Row = first Mueller row of an ideal polarizer at phi = 0.5*[1,cos2phi,sin2phi]."""
    return 0.5 * np.array([[1.0,  1.0, 0.0],
                           [1.0,  0.0, 1.0],
                           [1.0, -1.0, 0.0]])


def _load_characteristic_matrix(geom, l, verbose=False):
    """THIS pixel's (3,3) characteristic matrix A from the store (fixed on l==0)."""
    return get_store().char_mats[_CURRENT_INSTRUMENT_IDX]


def _load_calibration_covariance(geom, l, verbose=False):
    """The loaded (swappable) pooled (9,9) covariance of vec(C) (index k=3i+j)."""
    return get_store().cov_C


def _load_calibration_matrix(geom, l, verbose=False):
    """Path 2: draw ONE random (3,3) calibration matrix C from THIS wavelength's
    instrument's N_cal ensemble (one calibration event for this channel)."""
    store = get_store()
    cals = store.cal_mats[_CURRENT_INSTRUMENT_IDX]
    k = int(store._rng.integers(cals.shape[-1]))
    return cals[:, :, k]


def instr_err(norm_intens, bin_size,
              counts_max=0.75 * 2**14,
              noise_floor=10.0 / (0.75 * 2**14)):
    """Instrument noise model: shot noise + additive floor for a binned detector.

    *** Ported verbatim from nsienkie-cal-uncertainty/lib/cal_uncertainty/
        calibration.py::instr_err (copied, not imported, to avoid cross-repo
        import coupling). ***

    Returns the 1-sigma noise in normalized intensity units.
    1. Shot noise:  sqrt(counts / bin_size)          (Poisson, reduced by binning)
    2. Noise floor: floor_counts / sqrt(bin_size)    (read noise/dark, reduced by binning)
    Both in count units, normalized back to [0,1] by dividing by counts_max.

    counts_max  : full-well in ADC counts (0.75*2^14 = 12288; 14-bit @ 75% saturation).
    noise_floor : normalized floor (10/12288 = 10-count read noise).
    """
    counts = norm_intens * counts_max
    floor_err = noise_floor * counts_max
    counts_err = np.sqrt(np.abs(counts) / bin_size)
    counts_err += floor_err / np.sqrt(bin_size)
    return counts_err / counts_max


def _sensor_noise_sigma(sensorInt):
    """Per-sensor 1-sigma noise on the sensor intensities, via the ported instr_err
    (shot + read-noise floor) at the configured SENSOR_BIN_FACTOR. Shape == sensorInt."""
    return instr_err(sensorInt, SENSOR_BIN_FACTOR)


# =============================================================================
# Optional eager load at import (set EAGER_LOAD=True in the header above)
# =============================================================================
if EAGER_LOAD:
    try:
        init_store(h5_path=CAL_MATRIX_H5_PATH, cov_path=COV_MATRIX_PATH)
    except Exception as _e:   # don't let import fail; fall back to lazy load
        warnings.warn("err_sim eager load failed (%s); will lazy-load on first pixel." % _e)
