#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiment.py -- one-stop err_sim retrieval experiment: forward GRASP
"truth" -> custom calibration-uncertainty error model -> GRASP inversion ->
pickle -> quicklook PNGs. Everything you'd normally tune lives in the CONFIG
block below.

Pipeline (mirrors the real simulateRetrieval.runSim path, minus the Sabrina
orbital-geometry files, which aren't on this machine -- the 'harperrsim' arch
uses a fixed in-plane HARP2-like geometry, so we just sweep (sza, phi)):

    returnPixel(INSTRUMENT) -> setupConCaseYAML(CONCASE) -> simulation.runSim
        -> <SAVE_PKL>  ->  analyzeSim console stats + scatter/spectral PNGs

Total retrievals = NSIMS * N_PIX.

Two knobs of note:
  * INSTRUMENT: 'harperrsim'   -> Path 1 (analytic error propagation)
                'harperrsimmc' -> Path 2 (Monte Carlo sensor-space noise)
  * RUN_RETRIEVAL=False re-plots an existing pkl WITHOUT rerunning GRASP.

The local GRASP build caps aerosol modes at 2, so CONCASE must be a single
2-mode case (e.g. 'marineVariable'); the big machine's build handles the
multi-case scenes like 'marineVariable+smokeVariableOcean'.

Usage:
    python err_sim/run_experiment.py
"""

import os
import sys
from pprint import pformat
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')                            # headless -> PNG only, no X11
import matplotlib.pyplot as plt

# ================================ CONFIG ================================
# --- scene / instrument ---
# 'harperrsim' (Path 1, analytic) | 'harperrsimmc' (Path 2, Monte Carlo)
# 'harperrsimbck' (Path 3, control: inject exactly the BCK YAML's assumed noise)
# Override without editing this file:  ERRSIM_INSTRUMENT=harperrsimbck python err_sim/run_experiment.py
INSTRUMENT = os.environ.get('ERRSIM_INSTRUMENT', 'harperrsim')
CONCASE    = os.environ.get('ERRSIM_CONCASE', 'marineVariable')  # 2-mode scene (local GRASP caps modes at 2)
# 'randLogNrm<medianAOD>'; sigma is hardcoded ln(2), so 95% of draws land in
# [median/4, median*4].  0.2 suits the marine default; the smoke campaign uses 0.3
# (set in slurm_task_array.sh) -- thicker plumes give a stronger polarized signal,
# which is the reason for running smoke at all.
TAU_FACTOR = os.environ.get('ERRSIM_TAU_FACTOR', 'randLogNrm0.2')

# --- how many retrievals (total = NSIMS * N_PIX) ---
# Pixels swept over geometry.  ERRSIM_NPIX=all (or 'none') -> every valid pixel in
# the nc4 (526 for the AOS file); otherwise an integer.  Default 30.
N_PIX      = (None if os.environ.get('ERRSIM_NPIX', '').lower() in ('all', 'none')
              else int(os.environ.get('ERRSIM_NPIX', 30)))
NSIMS      = 1                       # noise repeats per pixel
# Parallel GRASP processes -- one per inversion chunk, so this IS the core count the
# run will occupy (each GRASP process is single-threaded).  Defaults to the SLURM
# allocation when running under sbatch, so --cpus-per-task is the single knob there;
# ERRSIM_MAXCPU overrides both.  Local default 12 (M5 Pro: 18 cores, 6 performance).
MAX_CPU    = int(os.environ.get('ERRSIM_MAXCPU',
                                os.environ.get('SLURM_CPUS_PER_TASK', 12)))
# Max pixels per GRASP process. graspDB splits the inversion into ceil(Npix/MAX_CPU)
# chunks, and one GRASP segment cannot hold more pixels than the build's constants
# allow.  All our pixels share ix=iy=1 and differ only in time, so the binding limit
# is _KITIME (30 in this build's generic constants set), NOT _KIMAGE (=KITIME*KIX*KIY
# =120).  Leaving this None silently exceeds it once Npix > 30*MAX_CPU and GRASP dies.
MAX_T      = int(os.environ.get('ERRSIM_MAXT', 25))   # < _KITIME=30, with headroom

# --- geometry ---
# 'nc4'    -> real orbital geometry from Sabrina's subsampled AOS file, the same
#             source runRetrievalSimulation.py uses (via ACCP_functions.selectGeomSabrina).
#             Pixels are walked in cumulative index order, skipping any with
#             sza > MAX_SZA or sza <= 0 (night side), until N_PIX are collected.
# 'random' -> the previous deterministic uniform (sza, phi) sweep.
#
# NOTE on view angles: the 'harperrsim*' archs use HARP2's hardcoded view zeniths
# and ignore the nc4's per-pixel `vza`, exactly as harp02/megaharp do when
# runRetrievalSimulation.py feeds them Sabrina geometry. So 'nc4' supplies the real
# solar zenith and relative azimuth, not the real view zeniths. The file does carry
# 10 vza values per pixel (6.9-68.1 deg), which map 1:1 onto the arch's 10 angles,
# so honouring them is possible -- it would require teaching the harperrsim block in
# architectureMap.py to accept vza, and would change the scattering-angle sampling.
GEOM_SOURCE = 'nc4'                  # 'nc4' | 'random'
GEOM_NC4    = None                   # resolved below; ERRSIM_GEOM_NC4 overrides
GEOM_NC4_NAME = 'MAAP-GeometrySubSample_AOS_1330_LTAN_442km_alt_2023Aug12.nc4'
GEOM_START_IND = 0                   # first cumulative pixel index to consider
MAX_SZA     = 70.0                   # skip pixels with sza above this (matches runRetrievalSimulation)
GEOM_SEED  = 7
TAU_SEED   = 11                      # seeds the per-pixel AOD draw (global np.random inside
                                     # setupConCaseYAML). REQUIRED for comparing paths: without
                                     # it each run draws a different truth scene and the
                                     # path-to-path differences are dominated by scene scatter.
SZA_RANGE  = (20.0, 65.0)            # solar zenith deg (keep < maxSZA=70)
PHI_RANGE  = (0.0, 180.0)            # relative azimuth deg

# --- retrieval / output control ---
RND_INITIAL_GUESS = True
VERBOSE           = True
RUN_RETRIEVAL     = True             # False -> skip GRASP, just re-plot existing SAVE_PKL
MAKE_PLOTS        = True
PLOT_WAVE_IND     = 1                # wavelength index for the scatter grid (1 = 0.549 um)

# =======================================================================

_HERE   = os.path.dirname(os.path.abspath(__file__))
_REPO   = os.path.dirname(_HERE)
_PARENT = os.path.dirname(_REPO)

# ============================ MACHINE PATHS ============================
# Nothing below is machine-specific: every path is DISCOVERED relative to a base
# directory, so the same checkout runs on the laptop and the cluster unedited.
#
#   ERRSIM_BASE   root to search under.  Defaults to the repo's parent, which is
#                 '<...>/grasp_sims' on the laptop and '/gpfsm/dnb33/nsienkie/retr_sim'
#                 on the cluster (where every tree is copied side by side).  The
#                 laptop also needs its grandparent, since grasp/ and the cal-sim live
#                 one level further up there -- both are searched.
#
# Individual overrides win over discovery and are reported verbatim on failure:
#   ERRSIM_GRASP_BIN      the grasp executable
#   ERRSIM_GRASP_KERNELS  the internal_files kernel directory
#   ERRSIM_GEOM_NC4       the orbital geometry .nc4
#   CAL_UNCERTAINTY_DIR   the cal-sim checkout (read by customErrModel)
#
# Run `python err_sim/check_paths.py` on a new machine to see what resolved.


def _expand(p):
    """Expand '~' and $VARS and absolutise; os.path.join does none of these."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p))) if p else p


ERRSIM_BASE = _expand(os.environ.get('ERRSIM_BASE', _PARENT))

# Roots to search, nearest first.  The grandparent covers the laptop layout where
# grasp/ and uncertainty/ are siblings of grasp_sims/ rather than of the repo.
_SEARCH_ROOTS = [ERRSIM_BASE, _expand(os.path.join(ERRSIM_BASE, '..')), _PARENT,
                 _expand(os.path.join(_PARENT, '..'))]
_SEARCH_ROOTS = list(dict.fromkeys(r for r in _SEARCH_ROOTS if r))  # de-dup, keep order


def _resolve(envVar, relCands, isDir, what):
    """First existing candidate, or an error naming every place we looked.

    An explicit env var wins even when it does not exist, so the message reports
    what the USER set rather than a discovery fallback they never asked for.
    """
    env = os.environ.get(envVar)
    if env:
        got = _expand(env)
        ok = os.path.isdir(got) if isDir else os.path.isfile(got)
        if not ok:
            raise FileNotFoundError('%s=%s does not exist (looking for %s)'
                                    % (envVar, got, what))
        return got
    tried = []
    for root in _SEARCH_ROOTS:
        for rel in relCands:
            cand = _expand(os.path.join(root, rel))
            tried.append(cand)
            if os.path.isdir(cand) if isDir else os.path.isfile(cand):
                return cand
    raise FileNotFoundError(
        '%s not found.\n  Set %s explicitly, or place it under ERRSIM_BASE=%s\n'
        '  Looked in:\n    %s' % (what, envVar, ERRSIM_BASE, '\n    '.join(tried)))


# The grasp binary. 'build' is a symlink to the working GCC-13 build on the laptop
# (see ../../GRASP_BUILD_NOTES.md); build_g13 is checked too in case it is not.
_GRASP_BIN_CANDS = ['grasp/build/bin/grasp', 'grasp/build_g13/bin/grasp',
                    'GRASP/build/bin/grasp', 'grasp/bin/grasp', 'bin/grasp']
_GRASP_KRNL_CANDS = ['grasp/src/retrieval/internal_files',
                     'GRASP/src/retrieval/internal_files',
                     'grasp/internal_files', 'internal_files']
_GEOM_CANDS = [os.path.join(d, GEOM_NC4_NAME)
               for d in ('data_stor', 'grasp_sims/data_stor', 'data', '.')]

DIR_GRASP = _resolve('ERRSIM_GRASP_BIN', _GRASP_BIN_CANDS, False, 'the grasp executable')
KRNL_PATH = _resolve('ERRSIM_GRASP_KERNELS', _GRASP_KRNL_CANDS, True,
                     'the GRASP kernel directory (internal_files)')
if GEOM_SOURCE.lower() == 'nc4':
    GEOM_NC4 = _resolve('ERRSIM_GEOM_NC4', _GEOM_CANDS, False,
                        'the orbital geometry file %s' % GEOM_NC4_NAME)
# =======================================================================
sys.path.append(os.path.join(_PARENT, 'GSFC-GRASP-Python-Interface'))
sys.path.append(_REPO)
sys.path.append(os.path.join(_REPO, 'ACCP_ArchitectureAndCanonicalCases'))

import err_sim.np_compat  # noqa: F401 -- restores np.trapz for NumPy>=2; MUST precede runGRASP
import simulateRetrieval as rs                                   # noqa: E402
from architectureMap import returnPixel                          # noqa: E402
from canonicalCaseMap import setupConCaseYAML                    # noqa: E402
import err_sim.customErrModel as cem                             # noqa: E402

YML_DIR  = os.path.join(_REPO, 'ACCP_ArchitectureAndCanonicalCases')
FWD_YAML = os.path.join(YML_DIR, 'settings_FWD_IQU_POLAR_1lambda.yml')
# ALL THREE PATHS invert with the SAME assumed-noise settings, so differences between
# them come only from the injected error model and never from the inversion weighting.
# That file carries sigma_I = 0.01 relative (the stock GRASP settings assume 0.03, which
# these experiments showed to be far larger than a realistically calibrated polarimeter
# delivers) and sigma_Q = sigma_U = 0.005 absolute.
#
# Path 3 (the control) additionally INJECTS exactly these sigmas -- customErrModel reads
# them back out of this same file -- which is what makes it the self-consistent case:
# injected error == assumed error. Paths 1 and 2 inject calibration-derived error instead,
# while being weighted by the same assumption.
#
# To restore the stock 3% assumption for every path, point BCK_YAML at
# 'settings_BCK_POLAR_2modes.yml'. Note that runs made before this change had Paths 1/2
# at 3% and Path 3 at 1%, so they are NOT comparable with runs made after it.
#
# ERRSIM_BCK_YAML selects the file (bare name resolved against YML_DIR, or a full path):
#   settings_BCK_POLAR_2modes_errsim_I1pct.yml  default; a-priori box tuned for MARINE
#   settings_BCK_POLAR_2modes_errsim_smoke.yml  same noise, a-priori box widened so the
#                                               smoke truth is REPRESENTABLE -- see below
#
# Why a separate file for smoke.  The smoke canonical case sits on top of the marine
# box, so the retrieval cannot return the right answer for roughly half the scenes:
#   k fine       truth 0.0100 vs max 0.01   -> 49% of truth ABOVE the cap
#   rv coarse    truth 0.664  vs min 0.65   -> 46% of truth BELOW the floor
#   sigma coarse truth 0.451  vs min 0.32   ->  8% below
#   sigma fine   truth 0.401  vs min 0.25   ->  3% below
# Measured over 6000 'smokeVariable' draws. That is a bound artifact, not a retrieval
# error, and it swamps the calibration signal we are trying to measure (the 2026-09
# campaign saw |bias|/spread of 10-37 on exactly these parameters). The smoke file
# widens k max 0.01->0.05, rv coarse min 0.65->0.35, sigma coarse 0.32->0.20 and sigma
# fine 0.25->0.15, taking every parameter under 1% out-of-range while keeping the fine
# and coarse modes separated (only 0.02% of fine rv draws exceed the new coarse floor).
BCK_YAML = os.environ.get('ERRSIM_BCK_YAML', 'settings_BCK_POLAR_2modes_errsim_I1pct.yml')
if not os.path.isabs(BCK_YAML):
    BCK_YAML = os.path.join(YML_DIR, BCK_YAML)
if not os.path.isfile(BCK_YAML):
    raise FileNotFoundError('ERRSIM_BCK_YAML does not exist: %s' % BCK_YAML)
cem.BCK_YAML_PATH = BCK_YAML          # Path 3 injects exactly what every path assumes
SAVE_PKL = os.path.join(_HERE, 'experiment_%s.pkl' % INSTRUMENT)

# per-panel scatter: (title, extractor(rslt, wi) -> scalar). wi = wavelength index.
PANELS = [
    ('AOD',         lambda r, wi: r['aod'][wi]),
    ('SSA',         lambda r, wi: r['ssa'][wi]),
    ('n (fine)',    lambda r, wi: r['n'][0][wi]),
    ('n (coarse)',  lambda r, wi: r['n'][1][wi]),
    ('k (fine)',    lambda r, wi: r['k'][0][wi]),
    ('rEff',        lambda r, wi: float(r['rEff'])),
    ('rv (fine)',   lambda r, wi: r['rv'][0]),
    ('rv (coarse)', lambda r, wi: r['rv'][1]),
]


_GEOM_CACHE = {}


def _cached_geom_reader(nc4File):
    """Return a selectGeomSabrina-compatible reader that opens the .nc4 ONCE.

    ``ACCP_functions.selectGeomSabrina`` opens and closes the file on EVERY call, so
    walking the geometry costs one open per candidate index -- 660 for the full AOS
    file.  Across a 250-task SLURM array that is ~165,000 opens of the same file, a
    metadata storm on a shared filesystem (correct, just wasteful).

    This reads the three variables once per process, caches them by path, and then
    answers each index from memory.  The per-index logic below is a faithful copy of
    selectGeomSabrina's: same cumInd -> (timeInd, crossInd) mapping, same
    (-1,-1,-1) sentinel past the end, same Delta-sza warning, same conversion of vza
    to signed and phi onto [0,180].  ``tests``-style equivalence against the original
    is checked in the smoke test.

    Note the returned arrays are COPIES, because the caller's conversions
    (``phi[phi<0] += 180``) mutate in place and would otherwise corrupt the cache.
    """
    if nc4File not in _GEOM_CACHE:
        from netCDF4 import Dataset
        with Dataset(nc4File, mode='r') as nc:
            _GEOM_CACHE[nc4File] = dict(
                azimuth=np.array(nc.variables['azimuth'][:]),
                sza=np.array(nc.variables['sza'][:]),
                vza=np.array(nc.variables['vza'][:]),
                nTime=nc.dimensions['time'].size,
                nCross=nc.dimensions['ncross'].size,
            )
    g = _GEOM_CACHE[nc4File]

    def reader(_path, cumInd=None, timeInd=None, crossInd=None, addVZA=None):
        assert addVZA is None, 'cached reader does not implement addVZA'
        if timeInd is None or crossInd is None:
            assert cumInd is not None, 'cumInd must be provided unless timeInd and crossInd are both provided'
            timeInd = cumInd % g['nTime']
            crossInd = int(np.floor(cumInd / g['nTime']))
            if crossInd >= g['nCross']:        # past the end of the file
                return -1, -1, -1
        phi = np.array(g['azimuth'][timeInd, crossInd, :])      # copy: mutated below
        sza = np.array(g['sza'][timeInd, crossInd, :])
        if np.any((sza - sza[0]) > 1):
            warnings.warn('Delta-sza was greater than 1 deg at timeInd=%d and crossInd=%d'
                          % (timeInd, crossInd))
        szaAvg = sza.mean()
        vza = np.array(g['vza'][crossInd, :])
        assert np.all(vza >= 0), 'At least one element of vza was less than zero before conversion!'
        vza = np.sign(phi) * vza
        phi[phi < 0] = phi[phi < 0] + 180
        assert np.logical_and(phi >= 0, phi <= 180).all(), \
            'At least one element did not satisfy 0 <= phi <= 180 after conversion!'
        return szaAvg, phi, vza

    return reader


def make_geoms():
    """Return N_PIX (sza, phi) geometries from the configured source.

    'nc4'  -- real orbital geometry, walked in cumulative-index order exactly as
              runRetrievalSimulation.py does: selectGeomSabrina returns (-1,-1,-1)
              once the file is exhausted, and pixels outside 0 < sza <= MAX_SZA are
              skipped (the file spans the night side, sza up to 111 deg).  phi comes
              back as a per-view vector; returnPixel collapses it to the scalar
              in-plane azimuth via phiConverter, since this arch is single-azimuth.
    'random' -- the previous uniform sweep.
    """
    if GEOM_SOURCE.lower() == 'random':
        rng = np.random.default_rng(GEOM_SEED)
        return [(s_, p_, None) for s_, p_ in zip(rng.uniform(*SZA_RANGE, size=N_PIX),
                                                 rng.uniform(*PHI_RANGE, size=N_PIX))]
    if GEOM_SOURCE.lower() != 'nc4':
        raise ValueError("GEOM_SOURCE must be 'nc4' or 'random', got %r" % GEOM_SOURCE)

    assert os.path.isfile(GEOM_NC4), 'Geometry nc4 not found: %s' % GEOM_NC4
    selectGeomSabrina = _cached_geom_reader(GEOM_NC4)
    wantAll = N_PIX is None
    geoms, ind, skipped = [], GEOM_START_IND, 0
    while wantAll or len(geoms) < N_PIX:
        sza, phi, vza = selectGeomSabrina(GEOM_NC4, ind)
        if np.isscalar(sza) and sza == -1:          # file exhausted
            if wantAll:
                break                               # N_PIX=None -> take every valid pixel
            raise RuntimeError('Geometry file exhausted after %d pixels (wanted %d); '
                               'lower N_PIX or GEOM_START_IND.' % (len(geoms), N_PIX))
        ind += 1
        if not (0 < sza <= MAX_SZA):                # night side / too oblique
            skipped += 1
            continue
        geoms.append((float(sza), phi, vza))
    print('geometry: %s -> %d pixels from cumInd %d..%d (%d skipped for sza outside (0, %g])'
          % (os.path.basename(GEOM_NC4), len(geoms), GEOM_START_IND, ind - 1, skipped, MAX_SZA))
    print('  sza range %.1f..%.1f deg' % (min(g[0] for g in geoms), max(g[0] for g in geoms)))
    return geoms


def errorModelOf(pix):
    """Return the error model bound to ``pix``, for passing as runSim(radianceNoiseFun=...).

    WHY THIS EXISTS.  returnPixel() binds functools.partial(addError, errStr) into
    every measVals[n]['errorModel'], but between interface-repo commits cd80445 (Jun
    2025) and a14a872 (Sep 2026) runGRASP.pixel.populateFromRslt() only ever called an
    error model when the ``radianceNoiseFun`` ARGUMENT was supplied:

        if radianceNoiseFun:  ... = msDct['errorModel'](l, rslt, verbose=verbose)
        else:                 ... = clean forward truth

    It never consulted the already-stored measVals[n]['errorModel'], so runSim()
    without radianceNoiseFun silently inverted the NOISE-FREE forward truth and
    err_sim/customErrModel.py was never invoked.  Passing the bound partial back in
    here restored the intended behaviour without editing the read-only
    GSFC-GRASP-Python-Interface repo.

    STILL CORRECT AFTER a14a872, which fixed this upstream by falling back to the
    pixel's own errorModel.  The override is now redundant rather than wrong: it
    assigns the same bound model that the pixel already carries, and the model is
    applied exactly once.  Verified by rerunning task 0 with noiseFun=None under the
    fixed dependency -- retrieved AOD was bit-identical (max|diff| = 0).  Kept so this
    code also works against a pre-a14a872 checkout of the interface repo.

    Safe for this experiment because every wavelength of 'harperrsim[mc]' shares one
    errStr; the assert below catches any future arch where that stops being true.
    """
    models = [mv['errorModel'] for mv in pix.measVals]
    assert all(m is not None for m in models), 'returnPixel bound no error model'
    argSets = {(m.func, m.args) for m in models}     # functools.partial identity
    assert len(argSets) == 1, ('This arch uses different error models per wavelength; '
                               'radianceNoiseFun applies ONE model to all of them.')
    return models[0]


def run_retrieval():
    assert os.path.isfile(DIR_GRASP), 'GRASP binary not found: %s' % DIR_GRASP
    assert os.path.isdir(KRNL_PATH), 'GRASP kernels not found: %s' % KRNL_PATH
    cem.init_store()                             # load cal-sim matrices + covariance ONCE
    if VERBOSE:
        print('err_sim store: n_instr=%d, cov diag[0]=%.3e'
              % (cem.get_store().n_instr, np.diag(cem.get_store().cov_C)[0]))

    nowPix = [returnPixel(INSTRUMENT, sza=sza, relPhi=phi, vza=vza, concase=CONCASE)
              for sza, phi, vza in make_geoms()]   # vza=None -> hardcoded HARP2 angles
    print('Instrument=%s  case=%s  pixels=%d  Nsims=%d  -> %d retrievals  (Nλ=%d)'
          % (INSTRUMENT, CONCASE, len(nowPix), NSIMS, NSIMS * len(nowPix), nowPix[0].nwl))
    noiseFun = errorModelOf(nowPix[0])           # see errorModelOf(): runSim needs this explicitly

    np.random.seed(TAU_SEED)                     # identical truth scenes across all paths
    fwdYAML = [setupConCaseYAML(CONCASE, npix, FWD_YAML, caseLoadFctr=TAU_FACTOR)
               for npix in nowPix]
    simA = rs.simulation(nowPix)
    simA.runSim(fwdYAML, BCK_YAML, NSIMS, maxCPU=MAX_CPU, maxT=MAX_T, savePath=SAVE_PKL,
                binPathGRASP=DIR_GRASP, intrnlFileGRASP=KRNL_PATH, releaseYAML=True,
                lightSave=False, rndIntialGuess=RND_INITIAL_GUESS, dryRun=False,
                workingFileSave=False, fixRndmSeed=False, verbose=VERBOSE,
                radianceNoiseFun=noiseFun)       # REQUIRED -- without it GRASP inverts clean truth
    print('Saved retrieval pickle -> %s' % SAVE_PKL)


def print_stats(sim):
    wvls = sim.rsltFwd[0]['lambda']
    print('\n%d pixels  |  wavelengths: %s um\n'
          % (len(sim.rsltBck), np.array2string(wvls, precision=3)))
    for w, wl in enumerate(wvls):
        rmse, bias, _ = sim.analyzeSim(w)
        print('=== %.3f um (waveInd=%d) ===' % (wl, w))
        print('RMSE:', pformat({k: np.round(v, 4) for k, v in rmse.items()}, width=100))
        print('BIAS:', pformat({k: np.round(v, 4) for k, v in bias.items()}, width=100))
        print('')


def scatter_png(sim, wi, out_png):
    ncol = 4
    nrow = int(np.ceil(len(PANELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
    for p, (title, fn) in enumerate(PANELS):
        ax = axes[p // ncol][p % ncol]
        x = np.array([fn(f, wi) for f in sim.rsltFwd], dtype=float)
        y = np.array([fn(b, wi) for b in sim.rsltBck], dtype=float)
        ax.scatter(x, y, s=18, color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.3)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, 'k--', lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        bias = float(np.mean(y - x))
        ax.set_title('%s\nRMSE=%.3g  bias=%+.3g' % (title, rmse, bias), fontsize=9)
        ax.set_xlabel('truth'); ax.set_ylabel('retrieved')
        ax.tick_params(labelsize=7)
    for p in range(len(PANELS), nrow * ncol):
        axes[p // ncol][p % ncol].axis('off')
    wl = sim.rsltFwd[0]['lambda'][wi]
    fig.suptitle('%s / %s: truth vs retrieved @ %.3f um  (N=%d)'
                 % (INSTRUMENT, CONCASE, wl, len(sim.rsltBck)), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print('Saved scatter -> %s' % out_png)


def spectral_aod_png(sim, out_png):
    wvls = sim.rsltFwd[0]['lambda']
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap('viridis')
    n = len(sim.rsltBck)
    for i, (f, b) in enumerate(zip(sim.rsltFwd, sim.rsltBck)):
        c = cmap(i / max(n - 1, 1))
        ax.plot(wvls, f['aod'], '--', color=c, lw=0.8, alpha=0.7)
        ax.plot(wvls, b['aod'], '-', color=c, lw=1.0, alpha=0.9)
    ax.set_xlabel('wavelength (um)'); ax.set_ylabel('AOD')
    ax.set_title('%s / %s spectral AOD: truth (dashed) vs retrieved (solid), N=%d'
                 % (INSTRUMENT, CONCASE, n))
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print('Saved spectral AOD -> %s' % out_png)


def main():
    if RUN_RETRIEVAL:
        run_retrieval()
    assert os.path.isfile(SAVE_PKL), 'No pickle at %s (set RUN_RETRIEVAL=True first)' % SAVE_PKL

    sim = rs.simulation(picklePath=SAVE_PKL)
    print_stats(sim)
    if MAKE_PLOTS:
        base = os.path.splitext(SAVE_PKL)[0]
        wi = PLOT_WAVE_IND
        wl_nm = int(sim.rsltFwd[0]['lambda'][wi] * 1000)
        scatter_png(sim, wi, '%s_scatter_%03dnm.png' % (base, wl_nm))
        spectral_aod_png(sim, '%s_spectralAOD.png' % base)


if __name__ == '__main__':
    main()
