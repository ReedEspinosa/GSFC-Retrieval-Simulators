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
import numpy as np
import matplotlib
matplotlib.use('Agg')                            # headless -> PNG only, no X11
import matplotlib.pyplot as plt

# ================================ CONFIG ================================
# --- scene / instrument ---
INSTRUMENT = 'harperrsim'            # 'harperrsim' (Path 1) | 'harperrsimmc' (Path 2)
CONCASE    = 'marineVariable'        # single 2-mode ocean scene (local GRASP caps modes at 2)
TAU_FACTOR = 'randLogNrm5.2'         # 'randLogNrm<medianAOD>', e.g. randLogNrm0.2 for thin

# --- how many retrievals (total = NSIMS * N_PIX) ---
N_PIX      = 21                      # pixels swept over geometry
NSIMS      = 1                       # noise repeats per pixel
MAX_CPU    = 3                       # parallel GRASP procs (throughput ~= MAX_CPU pixels / 80s)

# --- geometry sweep (deterministic; fixed seed for reproducibility) ---
GEOM_SEED  = 7
SZA_RANGE  = (20.0, 65.0)            # solar zenith deg (keep < maxSZA=70)
PHI_RANGE  = (0.0, 180.0)            # relative azimuth deg

# --- retrieval / output control ---
RND_INITIAL_GUESS = True
VERBOSE           = True
RUN_RETRIEVAL     = True             # False -> skip GRASP, just re-plot existing SAVE_PKL
MAKE_PLOTS        = True
PLOT_WAVE_IND     = 1                # wavelength index for the scatter grid (1 = 0.549 um)

# --- machine paths (edit per machine) ---
DIR_GRASP = '/home/noahs/npp_projects/grasp/build/bin/grasp'
KRNL_PATH = '/usr/local/share/grasp/kernels'
# =======================================================================

_HERE   = os.path.dirname(os.path.abspath(__file__))
_REPO   = os.path.dirname(_HERE)
_PARENT = os.path.dirname(_REPO)
sys.path.append(os.path.join(_PARENT, 'GSFC-GRASP-Python-Interface'))
sys.path.append(_REPO)
sys.path.append(os.path.join(_REPO, 'ACCP_ArchitectureAndCanonicalCases'))

import simulateRetrieval as rs                                   # noqa: E402
from architectureMap import returnPixel                          # noqa: E402
from canonicalCaseMap import setupConCaseYAML                    # noqa: E402
import err_sim.customErrModel as cem                             # noqa: E402

YML_DIR  = os.path.join(_REPO, 'ACCP_ArchitectureAndCanonicalCases')
FWD_YAML = os.path.join(YML_DIR, 'settings_FWD_IQU_POLAR_1lambda.yml')
BCK_YAML = os.path.join(YML_DIR, 'settings_BCK_POLAR_2modes.yml')
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


def make_geoms():
    rng = np.random.default_rng(GEOM_SEED)
    sza = rng.uniform(*SZA_RANGE, size=N_PIX)
    phi = rng.uniform(*PHI_RANGE, size=N_PIX)
    return list(zip(sza, phi))


def run_retrieval():
    assert os.path.isfile(DIR_GRASP), 'GRASP binary not found: %s' % DIR_GRASP
    assert os.path.isdir(KRNL_PATH), 'GRASP kernels not found: %s' % KRNL_PATH
    cem.init_store()                             # load cal-sim matrices + covariance ONCE
    if VERBOSE:
        print('err_sim store: n_instr=%d, cov diag[0]=%.3e'
              % (cem.get_store().n_instr, np.diag(cem.get_store().cov_C)[0]))

    nowPix = [returnPixel(INSTRUMENT, sza=sza, relPhi=phi, concase=CONCASE)
              for sza, phi in make_geoms()]
    print('Instrument=%s  case=%s  pixels=%d  Nsims=%d  -> %d retrievals  (Nλ=%d)'
          % (INSTRUMENT, CONCASE, len(nowPix), NSIMS, NSIMS * len(nowPix), nowPix[0].nwl))

    fwdYAML = [setupConCaseYAML(CONCASE, npix, FWD_YAML, caseLoadFctr=TAU_FACTOR)
               for npix in nowPix]
    simA = rs.simulation(nowPix)
    simA.runSim(fwdYAML, BCK_YAML, NSIMS, maxCPU=MAX_CPU, savePath=SAVE_PKL,
                binPathGRASP=DIR_GRASP, intrnlFileGRASP=KRNL_PATH, releaseYAML=True,
                lightSave=False, rndIntialGuess=RND_INITIAL_GUESS, dryRun=False,
                workingFileSave=False, fixRndmSeed=False, verbose=VERBOSE)
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
