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
    errsim03  -> _errsim_grasp_assumed()(Path 3: the BCK YAML's own assumed noise,
                                         i.e. the self-consistent control case)

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
# Location of the calibration-simulation repo (nsienkie-cal-uncertainty).
# Set CAL_UNCERTAINTY_DIR in the environment to override per machine; otherwise
# the paths below are used.
_ERRSIM_DIR = os.path.dirname(os.path.abspath(__file__))              # .../GSFC-Retrieval-Simulators/err_sim
_CALSIM_DIR = os.environ.get(
    'CAL_UNCERTAINTY_DIR',
    '/Users/nsienkie/working/uncertainty/nsienkie-cal-uncertainty')   # this machine

CAL_MATRIX_H5_PATH = os.path.join(_CALSIM_DIR, 'stor_data/eval_test/2026-09-15T16:46:06.h5')   # None -> dummy pool
COV_MATRIX_PATH    = os.path.join(_CALSIM_DIR, 'eval_results/output/csv/covariance_matrix_radcal.csv')  # None -> dummy cov
EAGER_LOAD = False          # True -> build the store at import; False -> on first pixel

# --- radiometric calibration ---
# The fitted radiometric gain is stored SEPARATELY from the polarimetric matrix
# in the cal-sim HDF5.  Radiometric calibration is part of the instrument
# characterization, so it must be folded into C before use -- otherwise the
# Path 2 matrices carry ~16x more scatter (sigma) than the radcal covariance
# COV_MATRIX_PATH points at, and the two paths model different instruments.
# Mirrors eval_results/*.py --use-rad-cal; keep this consistent with the
# radcal/noradcal tag of the covariance CSV above.
USE_RAD_CAL = True

# --- sensor-intensity noise model: see ported instr_err() below ---
SENSOR_BIN_FACTOR = 10      # detector pixels AVERAGED (operational; cal sim calibrated at 100)
SENSOR_GAIN_E_PER_ADU = 2.0 # conversion gain [e-/ADU]; shot noise is Poisson in ELECTRONS.
                            # Must match DetectorScaleConfig.gain_e_per_adu in the cal sim.
SENSOR_READ_NOISE_E = 5.0   # read noise, ELECTRONS per pixel read (specs are quoted in e-,
                            # so this stays fixed when the gain changes)
SENSOR_BINNING = 'digital'  # 'digital' (per-pixel reads averaged -> sigma_R/sqrt(B)) or
                            # 'onchip'  (charge summed, read once, CCD -> sigma_R/B)

# --- view-SECTOR dependent binning (Paths 1 & 2) ---
# A polarimeter pixel holds one measurement per VIEW SECTOR -- a fixed viewing
# direction built into the instrument, not a property of where the pixel landed in
# the orbit.  For a HARP-like instrument those sectors sit at +/-57 deg, and the
# detector bins many pixels in the near-nadir sectors and progressively fewer toward
# the extreme ones.  The bin factor is therefore a FIXED instrument property: it is
# indexed by sector and is identical for every ground pixel.  (It must NOT be driven
# by the per-pixel orbital vza -- that would make the detector layout drift with
# solar geometry.)
#
# Interpolated GEOMETRICALLY in |sector angle| so the transition is smooth in SNR
# (which goes as sqrt(bin)): BIN_NADIR at the nadir-most sector, BIN_EDGE at the
# most oblique.
BIN_VIEW_DEPENDENT = True
BIN_NADIR = 100.0           # pixels binned in the nadir-most view sector
BIN_EDGE = 1.0              # pixels binned in the most oblique view sector
VIEW_SECTOR_ANGLES = np.array([-57.0, -44.0, -32.0, -19.0, -6.0,
                               6.0, 19.0, 32.0, 44.0, 57.0])   # HARP2-like, degrees
VIEW_BAND_WVLS = (0.441, 0.549, 0.669, 0.873)   # band centres, micron (for band_response)

# --- radiometry: reflectance -> photons (Paths 1 & 2) ---
# GRASP works in REFLECTANCE (the cos(theta0) correction already applied), but shot
# noise depends on the PHOTON count, so the error model has to undo that:
#     L  = R * F0(lambda) * mu0 / pi        radiance
#     N_photons  ~  L * lambda / (h c)      photons carry F0*lambda, not F0
# Two factors therefore scale the signal before the detector model sees it:
#   * mu0 = cos(solar zenith): a low sun genuinely delivers fewer photons for the
#     same reflectance.  Omitting this makes shot noise sun-angle independent,
#     which is wrong by ~1.7x in sigma across a 15-70 deg SZA range.
#   * a per-band photon response from a SOLAR_T blackbody, normalised to the peak
#     of the photon-flux curve (not the energy curve -- counts are photoelectrons).
# The saturation anchor is a bright cloud (REFLECTANCE_SAT) viewed with the sun
# overhead at the photon-peak wavelength; that condition maps to counts_max.
# Band-to-band throughput, bandwidth and QE are NOT modelled -- add them here as
# an extra per-band factor when they matter.
GRASP_I_IS_REFLECTANCE = True   # False -> fit_I already carries mu0 (normalized radiance)

# --- solar spectrum source (swappable) ---
# 'blackbody' -- Planck curve at SOLAR_T, no external data needed.
# 'table'     -- interpolate a measured spectrum from SOLAR_TABLE_PATH: a two-column
#                text file of wavelength [micron] and irradiance [W m-2 um-1];
#                comment lines starting '#' or '%' are skipped.
#
# The intended upgrade is Thuillier et al. (2003) (ATLAS/SOLSPEC, 200-2400 nm);
# ASTM E490-00a AM0 and TSIS-1 HSRS are equally valid drop-ins.  A blackbody is a few
# percent low in absolute terms and, more importantly, gets the SHAPE wrong in the
# blue: the real Sun is depressed there by line blanketing, so a 5800 K curve likely
# over-weights the 0.441 band relative to 0.669/0.873.
#
# NOTE on band averaging: both paths evaluate F0 at the BAND CENTRE.  A measured
# spectrum carries Fraunhofer lines that a real instrument averages over its
# passband, so for full fidelity a table should be convolved with the band response
# first -- not done here, and a TODO if band shape ever matters.
#
# Only RATIOS between bands affect the result, because SAT_RADIANCE is derived from
# the same curve; a uniform scale error cancels exactly.
SOLAR_SPECTRUM = 'blackbody'    # 'blackbody' | 'table'
SOLAR_TABLE_PATH = None         # e.g. '.../thuillier2003.txt' when SOLAR_SPECTRUM='table'
SOLAR_T = 5800.0                # K, blackbody stand-in for the solar spectrum

# Saturation is pinned to a REAL RADIANCE in one reference band, so the whole scale
# is checkable against an instrument spec instead of being an abstract normalisation.
# Default: the radiance of a perfectly reflecting cloud (R=1) with the sun overhead
# in the red band, F0(0.669)/pi = 483 W m-2 sr-1 um-1.  Replace with the measured
# HARP2 saturation radiance when available.
# HARP2 anchors on the RED band because that is the band intended for cloud
# microphysics retrievals.  At a shared integration time this leaves the green band
# slightly over its nominal ceiling on the brightest clouds -- harmless here, since
# counts_max = 0.75*2^14 is a SOFT ceiling (75% of full scale, chosen to stay clear
# of non-linearity) and nothing in the noise model clips.  In reality green would
# just start to see mild non-linearity, which is not modelled.
SAT_REF_WVL = 0.669             # reference band centre, micron
# Radiance that saturates the reference band [W m-2 sr-1 um-1].
# None -> derived as F0(SAT_REF_WVL)/pi, i.e. DEFINED so that a perfectly reflecting
# Lambertian scene (R=1) with the sun overhead exactly saturates the reference band
# (483.3 for the 5800 K blackbody at 0.669 um).  That is a definition, not an
# instrument measurement -- real bright clouds are nearer R~0.8-0.9 at TOA -- so
# replace it with the measured HARP2 saturation radiance when that is available.
# Deriving rather than hardcoding keeps the anchor consistent if SOLAR_T or
# SAT_REF_WVL change, and makes counts_scale in the reference band reduce to mu0.
SAT_RADIANCE = None

# Per-band detector quantum efficiency, electrons per photon, keyed by band centre
# in micron.  None -> flat QE of 1.0 (no band-to-band QE effect).
# QE is the DOMINANT band-to-band term for silicon: the lambda factor above nearly
# cancels the F0 roll-off on its own (873 band ends up 0.94x the 549 band), but a
# representative back-illuminated Si QE takes it to 0.61x.  Example to drop in once
# real numbers are available:
#     QE_BY_BAND = {0.441: 0.90, 0.549: 0.95, 0.669: 0.85, 0.873: 0.40}
QE_BY_BAND = None

# How the bands are radiometrically balanced:
#   'common'   -- one integration time / gain for all bands, so a band's counts scale
#                 with QE(l)*F0(l)*l and the weakest band is genuinely noisier.
#   'balanced' -- integration time (or per-band gain) tuned so every band saturates at
#                 the same scene, which is what a designed instrument usually does.
#                 All band weights are then 1.0 and F0/QE cancel out entirely.
BAND_RESPONSE_MODE = 'common'

# --- Path 3: the inversion's OWN assumed noise (control case) ---
# Path 3 injects exactly the error the BCK YAML tells GRASP to assume, making the
# experiment self-consistent (injected error == assumed error).  Read from the YAML
# at runtime rather than hardcoded, so Path 3 automatically tracks the retrieval
# settings.  None -> the default 2-mode BCK file next to architectureMap.py.
BCK_YAML_PATH = None

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
        If USE_RAD_CAL, the fitted radiometric gain is folded into cal_mats (see
        _apply_radiometric_gain).  char_mats is the PHYSICAL truth and is never
        scaled -- the gain is a property of the calibration, not of the optics.
        Covariance is NOT read here -- it comes from cov_path via _load_covariance_matrix()."""
        import h5py
        if not os.path.isfile(h5_path):
            raise FileNotFoundError("CAL_MATRIX_H5_PATH does not exist: %s" % h5_path)
        with h5py.File(h5_path, 'r') as f:
            # drop the V column (last of the 4 Stokes weights) -> (N_instr, 3, 3)
            self.char_mats = np.array(f['results/characteristic_matrices'][:, :, :N_STOKES])
            self.cal_mats  = np.array(f['results/calibration_matrices'][:])   # (N_instr,3,3,N_cal)
            if USE_RAD_CAL:
                gainKey = 'results/radiometric/gain_counts_per_radiance'
                if gainKey not in f:
                    raise KeyError("USE_RAD_CAL=True but %s has no %s (the cal sim was run "
                                   "with --rad-cal none). Set USE_RAD_CAL=False and point "
                                   "COV_MATRIX_PATH at the noradcal covariance." % (h5_path, gainKey))
                self.cal_mats = _apply_radiometric_gain(self.cal_mats, np.array(f[gainKey][:]))

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
# Radiometric gain folding
# =============================================================================
def _apply_radiometric_gain(cal_mats, gain):
    """Fold the fitted radiometric gain into the polarimetric calibration matrices.

    Ported from nsienkie-cal-uncertainty/eval_results/covariance_matrix.py::
    apply_radiometric_gain so that the matrices drawn by Path 2 carry the same
    radiometric scatter that is already baked into the radcal covariance used by
    Path 1.  Right-multiplies each (3,3) by the inverse gain:

        scalar     (N_instr, N_cal)            -> C / gain
        per_sensor (N_instr, N_sensors, N_cal) -> C @ diag(1/gains)

    cal_mats : (N_instr, 3, 3, N_cal)
    gain     : (N_instr, N_cal) or (N_instr, N_sensors, N_cal)
    Returns a new array of the same shape as cal_mats.
    """
    if gain.ndim == 2:            # scalar: one gain per (instrument, trial)
        return cal_mats / gain[:, None, None, :]
    if gain.ndim == 3:            # per_sensor: gain indexes the COLUMN (sensor) axis
        return cal_mats / gain[:, None, :, :]
    raise ValueError("Unexpected radiometric gain dataset shape %s" % (gain.shape,))


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
    elif errId == 3:
        return _errsim_grasp_assumed(measNm, l, rsltFwd, concase=concase, orbit=orbit,
                                     lidErrDir=lidErrDir, verbose=verbose)
    else:
        raise ValueError("Unknown err_sim id in measNm=%r (expected errsim01, errsim02 or errsim03)"
                         % measNm)


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
# Radiometry and binning helpers (Paths 1 & 2)
# =============================================================================
_H_PLANCK = 6.62607015e-34
_R_SUN = 6.957e8            # m
_AU = 1.495978707e11        # m
_C_LIGHT = 2.99792458e8
_K_BOLTZ = 1.380649e-23


_SOLAR_TABLE = None


def _load_solar_table(path):
    """Load and cache a two-column solar spectrum: wavelength [um], F0 [W m-2 um-1]."""
    global _SOLAR_TABLE
    if _SOLAR_TABLE is not None and _SOLAR_TABLE[2] == path:
        return _SOLAR_TABLE
    if not os.path.isfile(path):
        raise FileNotFoundError('SOLAR_TABLE_PATH does not exist: %s' % path)
    wl, F0 = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in '#%':
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 2:
                continue
            wl.append(float(parts[0]))
            F0.append(float(parts[1]))
    if len(wl) < 2:
        raise ValueError('Fewer than 2 usable rows in solar table %s' % path)
    order = np.argsort(wl)
    _SOLAR_TABLE = (np.asarray(wl)[order], np.asarray(F0)[order], path)
    return _SOLAR_TABLE


def solar_F0_blackbody(wl_um, T=None):
    """Planck TOA solar spectral irradiance [W m-2 um-1] at wl_um [micron].

    Planck radiance at SOLAR_T scaled by the solid angle the Sun subtends at 1 AU.
    Gives 1789 W m-2 um-1 at 0.549 um; measured spectra put that nearer 1850, so the
    absolute scale is a few percent low (which cancels -- see SOLAR_SPECTRUM).
    """
    if T is None:
        T = SOLAR_T
    wl = np.asarray(wl_um, dtype=float) * 1e-6
    B = 2 * _H_PLANCK * _C_LIGHT ** 2 / wl ** 5 / np.expm1(
        _H_PLANCK * _C_LIGHT / (wl * _K_BOLTZ * T))          # W m-2 sr-1 m-1
    return B * np.pi * (_R_SUN / _AU) ** 2 * 1e-6            # W m-2 um-1


def solar_F0_table(wl_um, path=None):
    """Measured TOA solar spectral irradiance [W m-2 um-1], linearly interpolated."""
    if path is None:
        path = SOLAR_TABLE_PATH
    if path is None:
        raise ValueError("SOLAR_SPECTRUM='table' requires SOLAR_TABLE_PATH to be set")
    wlTab, f0Tab, _ = _load_solar_table(path)
    wl = np.asarray(wl_um, dtype=float)
    if np.any(wl < wlTab[0]) or np.any(wl > wlTab[-1]):
        raise ValueError('Band centre outside solar table range %.3f-%.3f um'
                         % (wlTab[0], wlTab[-1]))
    return np.interp(wl, wlTab, f0Tab)


def solar_F0(wl_um):
    """TOA solar spectral irradiance [W m-2 um-1] from the configured source.

    ENERGY units -- the factor that turns GRASP's reflectance into radiance,
    L = R * F0 * mu0 / pi.  Dispatches on SOLAR_SPECTRUM so a measured spectrum
    (Thuillier, ASTM E490, TSIS-1) drops in without touching anything else.
    """
    if SOLAR_SPECTRUM == 'blackbody':
        return solar_F0_blackbody(wl_um)
    if SOLAR_SPECTRUM == 'table':
        return solar_F0_table(wl_um)
    raise ValueError("SOLAR_SPECTRUM must be 'blackbody' or 'table', got %r"
                     % SOLAR_SPECTRUM)


def band_qe(wl_um):
    """Quantum efficiency [e-/photon] for the band centred at wl_um."""
    if not QE_BY_BAND:
        return 1.0
    key = min(QE_BY_BAND, key=lambda w: abs(w - wl_um))
    if abs(key - wl_um) > 0.02:
        warnings.warn('err_sim: no QE entry within 20 nm of %.3f um; using %.3f um'
                      % (wl_um, key))
    return float(QE_BY_BAND[key])


def responsivity(wl_um):
    """Electrons per unit RADIANCE for this band, up to a constant: QE(lambda)*lambda.

    A detector counts photons, and a photon at wavelength l carries hc/l of energy, so
    a fixed radiance delivers proportionally more photons at longer wavelengths --
    hence the factor of lambda alongside QE.  This is why the SATURATING RADIANCE is
    not the same in every band at a shared integration time: L_sat ~ 1 / (QE*lambda).
    """
    return band_qe(float(wl_um)) * float(wl_um)


def sat_radiance(wl_um):
    """Radiance that saturates the band centred at wl_um [W m-2 sr-1 um-1].

    'common'   -- one integration time for every band (HARP2), so saturation tracks
                  the inverse responsivity: L_sat(l) = SAT_RADIANCE * rho(ref)/rho(l).
                  A weak band needs MORE radiance to saturate and is noisier at a
                  given scene.
    'balanced' -- per-band integration time tuned so every band saturates at the same
                  REFLECTANCE; F0 and QE then cancel and only mu0 survives.
    """
    satRef = SAT_RADIANCE if SAT_RADIANCE is not None else solar_F0(SAT_REF_WVL) / np.pi
    if BAND_RESPONSE_MODE == 'balanced':
        return satRef * solar_F0(wl_um) / solar_F0(SAT_REF_WVL)
    if BAND_RESPONSE_MODE != 'common':
        raise ValueError("BAND_RESPONSE_MODE must be 'common' or 'balanced', got %r"
                         % BAND_RESPONSE_MODE)
    return satRef * responsivity(SAT_REF_WVL) / responsivity(wl_um)


def counts_scale(geom):
    """Factor converting a REFLECTANCE-unit signal into fraction of saturation.

    GRASP gives reflectance R; the detector model wants a normalized radiance in
    [0,1], so::

        L    = R * F0(lambda) * mu0 / pi          reflectance -> radiance
        norm = L / L_sat(lambda)                  radiance -> fraction of saturation

    and this returns k = norm / R, i.e. F0*mu0/(pi*L_sat).  The photon/QE physics
    lives entirely inside L_sat via responsivity(); instr_err then does the
    radiance -> counts -> electrons conversion as before.
    """
    wl = geom['bandWvl']
    k = solar_F0(wl) / (np.pi * sat_radiance(wl))
    if GRASP_I_IS_REFLECTANCE:
        k *= np.cos(np.radians(geom['solZen']))
    return float(k)


def bin_factor(nView):
    """Detector pixels binned in each VIEW SECTOR -- a fixed instrument property.

    Indexed by sector, using VIEW_SECTOR_ANGLES: the nadir-most sector gets
    BIN_NADIR, the most oblique gets BIN_EDGE, geometrically interpolated in
    |sector angle| so SNR (~sqrt(bin)) varies smoothly across the sectors.

    Deliberately takes only the NUMBER of views, not their orbital zenith angles:
    the sectors are fixed in the instrument frame, so every ground pixel sees the
    same binning regardless of where it fell in the swath.

    Returns a float array of length nView (>= 1).
    """
    if not BIN_VIEW_DEPENDENT:
        return np.full(nView, float(SENSOR_BIN_FACTOR))
    sect = VIEW_SECTOR_ANGLES
    if nView != len(sect):
        # Unknown sector layout: spread the requested number of views evenly over the
        # same angular span rather than silently mis-indexing the real sectors.
        sect = np.linspace(sect.min(), sect.max(), nView)
        warnings.warn('err_sim: %d views but %d VIEW_SECTOR_ANGLES; interpolating the '
                      'sector layout to match.' % (nView, len(VIEW_SECTOR_ANGLES)))
    v = np.abs(sect)
    vmin, vmax = v.min(), v.max()
    if vmax <= vmin:
        return np.full(nView, float(BIN_NADIR))
    frac = (v - vmin) / (vmax - vmin)                  # 0 at nadir-most, 1 at the edge
    return np.maximum(BIN_NADIR ** (1.0 - frac) * BIN_EDGE ** frac, 1.0)


def sensor_sigma(sensorInt, binFac, k):
    """1-sigma detector noise on sensor intensities, in the SAME units as sensorInt.

    The signal is scaled into fraction-of-saturation (where the photon statistics
    live), the detector model is applied there, and the resulting sigma is scaled
    back.  binFac may be a scalar or broadcast against sensorInt's last axis.
    """
    return instr_err(sensorInt * k, binFac) / k


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

    binFac = bin_factor(Nang)                     # (Nang,) fixed per view SECTOR
    kScale = counts_scale(geom)                   # reflectance -> fraction of saturation

    sigmaStokes = np.zeros((N_STOKES, Nang))
    sigmaDoLP   = np.zeros(Nang)
    noisyStokes = np.zeros((N_STOKES, Nang))
    for n in range(Nang):
        s = stokes[:, n]
        sensorInt = charMat @ s                                       # (3,) sensor intensities
        # (1) measurement-noise term: diagonal sensor covariance propagated through C=inv(A)
        sigI = sensor_sigma(sensorInt, binFac[n], kScale)             # (3,) per-sensor 1-sigma
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
                                wavelength=geom['bandWvl'], instrument_idx=_CURRENT_INSTRUMENT_IDX,
                                bin_factor=binFac, counts_scale=kScale)
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
    binFac      = bin_factor(stokes.shape[1])                             # (Nang,) fixed per view SECTOR
    sigSensor   = sensor_sigma(sensorInt, binFac, counts_scale(geom))     # (3, Nang) 1-sigma
    noisySensor = sensorInt + np.random.normal(size=sensorInt.shape) * sigSensor
    outStokes   = calMat @ noisySensor                                    # (3, Nang) recombine w/ imperfect C

    if verbose:
        print('[err_sim] montecarlo: l=%d wvl=%.3f Nang=%d instr=%s | mean sigma_sensor=%.3g'
              % (l, geom['bandWvl'], stokes.shape[1], _CURRENT_INSTRUMENT_IDX, sigSensor.mean()))
    return np.r_[outStokes[0], outStokes[1], outStokes[2]]


# =============================================================================
# Path 3 -- inject the inversion's OWN assumed noise (self-consistent control)
# =============================================================================
_BCK_NOISE_CACHE = {}
_BCK_POL_CACHE = {}


def read_bck_polarization(yaml_path=None):
    """Return retrieval.inversion.measurement_fitting.polarization from the BCK YAML.

    This decides WHAT QUANTITY GRASP actually fits, and therefore what the noise
    block's sigmas apply to (grasp_settings.c: iPOBS):

        absolute_polarization_components  (iPOBS 1) -> I, Q,   U
        relative_polarization_components  (iPOBS 2) -> I, Q/I, U/I
        polarized_reflectance             (iPOBS 3) -> I, sqrt(Q^2+U^2)
        degree_of_polarization            (iPOBS 4) -> I, sqrt(Q^2+U^2)/I

    Default in GRASP is 'absolute_polarization_components'.
    """
    if yaml_path is None:
        yaml_path = BCK_YAML_PATH or _default_bck_yaml()
    if yaml_path in _BCK_POL_CACHE:
        return _BCK_POL_CACHE[yaml_path]
    import yaml as _yaml
    with open(yaml_path) as f:
        dl = _yaml.safe_load(f)
    mf = dl['retrieval']['inversion'].get('measurement_fitting', {}) or {}
    mode = str(mf.get('polarization', 'absolute_polarization_components')).lower()
    _BCK_POL_CACHE[yaml_path] = mode
    return mode


def _default_bck_yaml():
    """Path to the BCK settings file used by run_experiment.py."""
    repoDir = os.path.dirname(_ERRSIM_DIR)          # .../GSFC-Retrieval-Simulators
    return os.path.join(repoDir, 'ACCP_ArchitectureAndCanonicalCases',
                        'settings_BCK_POLAR_2modes.yml')


def read_bck_noise(yaml_path=None):
    """Parse the BCK YAML 'noises' block -> {'I': (error_type, sigma), 'Q': ..., 'U': ...}.

    The block looks like::

        noises:
          noise[1]: {error_type: relative, standard_deviation: 0.03,
                     measurement_type[1]: {type: I, ...}}
          noise[2]: {error_type: absolute, standard_deviation: 0.005,
                     measurement_type[1]: {type: Q, ...}
                     measurement_type[2]: {type: U, ...}}

    One noise entry can cover several measurement types (Q and U share noise[2]),
    so every 'measurement_type*' key under an entry inherits that entry's
    error_type / standard_deviation.  Cached per path.

    NOTE: 'standard_deviation_synthetic' is deliberately ignored -- that is GRASP's
    own synthetic-noise generator, which this project leaves at 0.0 because the
    noise is added in Python (see ../ALTERING_THE_MEASUREMENT_ERROR_MODEL.md).
    """
    if yaml_path is None:
        yaml_path = BCK_YAML_PATH or _default_bck_yaml()
    if yaml_path in _BCK_NOISE_CACHE:
        return _BCK_NOISE_CACHE[yaml_path]
    import yaml as _yaml
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError("BCK YAML not found for Path 3: %s" % yaml_path)
    with open(yaml_path) as f:
        dl = _yaml.safe_load(f)
    try:
        noises = dl['retrieval']['inversion']['noises']
    except (KeyError, TypeError):
        raise KeyError("No retrieval.inversion.noises block in %s" % yaml_path)
    out = {}
    for noiseKey, noiseDct in noises.items():
        if not noiseKey.startswith('noise'):
            continue
        errType = str(noiseDct['error_type']).lower()
        sigma = float(noiseDct['standard_deviation'])
        for mtKey, mtDct in noiseDct.items():
            if not mtKey.startswith('measurement_type'):
                continue
            out[str(mtDct['type']).upper()] = (errType, sigma)
    _BCK_NOISE_CACHE[yaml_path] = out
    return out


def _errsim_grasp_assumed(measNm, l, rsltFwd, concase=None, orbit=None, lidErrDir=None,
                          verbose=False):
    """Path 3: perturb the truth with EXACTLY the noise the BCK YAML assumes.

    This is the self-consistent control for Paths 1 and 2: the error GRASP is told
    to expect is the error actually injected, so any residual retrieval error is
    attributable to the retrieval setup (regularisation, a-priori, mode structure,
    information content) rather than to a mismatch between assumed and true error.

    Applied per measurement type, exactly as GRASP interprets its own noise block:
      * error_type 'relative' -> multiplicative log-normal, sigma = ln(1+sd).
        GRASP fits relative-error measurement types in log space, so a log-normal
        is the matching perturbation (this is what the legacy polarNN models use).
      * error_type 'absolute' -> additive Gaussian, N(0, sd).

    NOTE on Q/U: the YAML assigns Q and U an ABSOLUTE sigma, and that is what is
    applied here -- independently to each, and independently of I.  This differs
    from the legacy 'polarNN' models, which instead scale Q and U by I's log-normal
    factor (preserving q=Q/I, u=U/I) and then add a DoLP error.  The literal YAML
    reading is used because the whole point of Path 3 is to match what the inversion
    assumes, not what other instrument presets do.

    Unlike Paths 1 and 2 this path touches NO calibration-sim product -- no
    characteristic matrix, no calibration matrix, no covariance.  The instrument
    index drawn by customErrModel() is therefore unused here; that is harmless.
    """
    stokes, geom = _extract_truth_and_geometry(l, rsltFwd)
    trueI, trueQ, trueU = stokes[0], stokes[1], stokes[2]
    noise = read_bck_noise()

    missing = [k for k in ('I', 'Q', 'U') if k not in noise]
    if missing:
        raise KeyError("BCK YAML noise block has no entry for %s (found %s); Path 3 needs "
                       "I, Q and U." % (', '.join(missing), ', '.join(sorted(noise))))

    def _perturb(truth, key):
        errType, sd = noise[key]
        if errType == 'relative':
            return truth * np.random.lognormal(sigma=np.log(1 + sd), size=truth.shape)
        elif errType == 'absolute':
            return truth + sd * np.random.normal(size=truth.shape)
        raise ValueError("Unsupported error_type %r for %s in the BCK YAML" % (errType, key))

    outI = _perturb(trueI, 'I')

    # WHICH QUANTITY the Q/U sigmas apply to depends on measurement_fitting.polarization.
    # With 'relative_polarization_components' (iPOBS=2) GRASP fits q=Q/I and u=U/I, so an
    # 'absolute' sigma there is absolute in q,u -- NOT in Stokes units.  Injecting it as a
    # Stokes-unit error would be wrong by a factor 1/I (>10x for dark ocean scenes, where
    # I ~ 0.085), which is both physically wrong and far larger than the inversion expects.
    # SDATA still carries Q and U (meas types 42/43), so we perturb q,u and multiply back
    # through the NOISED intensity, making the fitted q',u' carry exactly the drawn error.
    polMode = read_bck_polarization()
    if 'relative_polarization' in polMode:
        with np.errstate(divide='ignore', invalid='ignore'):
            q, u = trueQ / trueI, trueU / trueI
        outQ = _perturb(q, 'Q') * outI
        outU = _perturb(u, 'U') * outI
    elif 'absolute_polarization' in polMode:
        outQ, outU = _perturb(trueQ, 'Q'), _perturb(trueU, 'U')
    else:
        raise NotImplementedError(
            "Path 3 does not yet handle measurement_fitting.polarization=%r (only "
            "absolute_polarization_components and relative_polarization_components). "
            "The noise-block sigmas would apply to a different fitted quantity." % polMode)

    if verbose:
        print('[err_sim] grasp-assumed: l=%d wvl=%.3f Nang=%d pol=%s | %s'
              % (l, geom['bandWvl'], stokes.shape[1], polMode,
                 '  '.join('%s:%s %.4g' % (k, noise[k][0][:3], noise[k][1])
                           for k in ('I', 'Q', 'U'))))
    return np.r_[outI, outQ, outU]


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
              read_noise_e=None,
              gain_e_per_adu=None,
              binning=None):
    """Detector noise: Poisson shot noise + read noise, combined in quadrature.

    *** Kept in sync with nsienkie-cal-uncertainty/lib/cal_uncertainty/
        calibration.py::instr_err (copied, not imported, to avoid cross-repo import
        coupling).  If you change one, change the other. ***

    Worked entirely in PHOTOELECTRONS, where Poisson statistics live and where read
    noise is physically specified::

        C   = norm_intens * counts_max                      [ADU]
        N_e = C * g                                         [e-]
        digital binning:  var_e = (N_e + sigma_R**2) / B
        on-chip binning:  var_e = N_e/B + sigma_R**2 / B**2

    then divided by g and counts_max to return normalized units.  Read noise is
    incurred per pixel READ, so digital binning averages B independent samples
    (sigma_R/sqrt(B)) whereas CCD on-chip binning pays it once (sigma_R/B).

    SNR = sqrt(N_e) is independent of g, as it must be for a pure amplifier; g
    enters only because the input is a fraction of saturation rather than an
    electron count, and it sets the full well (counts_max*g).

    The legacy form was sqrt(C/B) + floor/sqrt(B): it assumed g = 1 and summed the
    two independent sources LINEARLY instead of in quadrature.
    """
    if read_noise_e is None:
        read_noise_e = SENSOR_READ_NOISE_E
    if gain_e_per_adu is None:
        gain_e_per_adu = SENSOR_GAIN_E_PER_ADU
    if binning is None:
        binning = SENSOR_BINNING
    counts = np.abs(norm_intens) * counts_max
    n_e = counts * gain_e_per_adu
    if binning == 'digital':
        var_e = (n_e + read_noise_e ** 2) / bin_size
    elif binning == 'onchip':
        var_e = n_e / bin_size + read_noise_e ** 2 / bin_size ** 2
    else:
        raise ValueError("binning must be 'digital' or 'onchip', got %r" % binning)
    return np.sqrt(var_e) / gain_e_per_adu / counts_max


# =============================================================================
# Optional eager load at import (set EAGER_LOAD=True in the header above)
# =============================================================================
if EAGER_LOAD:
    try:
        init_store(h5_path=CAL_MATRIX_H5_PATH, cov_path=COV_MATRIX_PATH)
    except Exception as _e:   # don't let import fail; fall back to lazy load
        warnings.warn("err_sim eager load failed (%s); will lazy-load on first pixel." % _e)
