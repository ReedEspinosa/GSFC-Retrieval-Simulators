#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mc_test.py -- truncated stand-in for the retrieval simulator, to exercise the
err_sim error model exactly as the real code does and visualize what it does.

What it mimics
--------------
The real retrieval sim (simulateRetrieval.runSim) builds a dummy `pixel` via
architectureMap.returnPixel(archName), then for each forward "truth" it calls
pixel.populateFromRslt(rsltFwd), whose inner loop invokes, per wavelength l:

    measVals[l]['errorModel'](l, rsltFwd)     # == addError(errStr, l, rsltFwd)
                                              #    -> err_sim.customErrModel(...)

We reproduce that EXACT call. The only thing faked is the forward run itself:
rsltFwd here holds randomly generated Stokes vectors (fit_I/Q/U) instead of GRASP
output, plus the matching geometry (vis/fis/sza/lambda) the model reads.

Two archs are exercised:
    'harperrsim'   -> errsim01 = Path 1 (analytical error propagation)
    'harperrsimmc' -> errsim02 = Path 2 (Monte Carlo sensor-space noise)

Output
------
A matplotlib figure (saved PNG) of the histogram of (perturbed - truth) for each
of I, Q, U and DoLP, one row per path. Also prints mean/std to the console.

Usage
-----
    python err_sim/mc_test.py [N_pixels]      # default 500
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')                          # headless (WSL) -> save to file
import matplotlib.pyplot as plt

# --- make the repo importable exactly like the run scripts do ---
_HERE = os.path.dirname(os.path.abspath(__file__))               # .../GSFC-Retrieval-Simulators/err_sim
_REPO = os.path.dirname(_HERE)                                   # .../GSFC-Retrieval-Simulators
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'ACCP_ArchitectureAndCanonicalCases'))
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))

from architectureMap import returnPixel                          # noqa: E402
import err_sim.customErrModel as cem                             # noqa: E402


# ---- config ----
SZA = 40.0          # solar zenith (deg)
PHI = 30.0          # relative azimuth (deg)
ARCHS = [('harperrsim',  'Path 1: analytic'),
         ('harperrsimmc', 'Path 2: Monte Carlo')]
OUT_PNG = os.path.join(_HERE, 'mc_test_hist.png')


def generate_random_stokes(nang, nwl, rng):
    """Random but physical [I,Q,U] truth, shape (nang, nwl) each.

    (Style borrowed from the cal sim's mc_err_prop.generate_random_stokes: enforce
    DoP<=1 by construction.) I in [0.05,0.4]; DoLP in [0,0.7]; random pol angle.
    """
    I = rng.uniform(0.05, 0.40, size=(nang, nwl))
    dolp = rng.uniform(0.0, 0.7, size=(nang, nwl))
    chi = rng.uniform(0.0, np.pi, size=(nang, nwl))
    Q = I * dolp * np.cos(2 * chi)
    U = I * dolp * np.sin(2 * chi)
    return I, Q, U


def dolp(I, Q, U):
    """Degree of linear polarization, guarding I<=0."""
    out = np.full_like(I, np.nan, dtype=float)
    good = I > 0
    out[good] = np.sqrt(Q[good]**2 + U[good]**2) / I[good]
    return out


def run_arch(archName, n_pix, stokes_rng):
    """Run n_pix pixel realizations through one arch; return delta arrays."""
    # build the dummy pixel exactly like the retrieval sim (binds the errModel)
    nowPix = returnPixel(archName, sza=SZA, relPhi=PHI)
    wvls = np.array([mv['wl'] for mv in nowPix.measVals])          # ascending
    nwl = len(wvls)
    nang = int(nowPix.measVals[0]['nbvm'][0])                      # views per meas-type (I block)
    # geometry: the harperrsim view angles (signed); vis feeds only scatAng today
    view_angles = np.array([-57.0, -44.0, -32.0, -19.0, -6.0, 6.0, 19.0, 32.0, 44.0, 57.0])[:nang]
    vis = np.tile(view_angles[:, None], (1, nwl))
    fis = np.full((nang, nwl), PHI)
    sza = np.full((nang, nwl), SZA)

    d = {k: [] for k in ('I', 'Q', 'U', 'DoLP')}
    for _ in range(n_pix):
        I, Q, U = generate_random_stokes(nang, nwl, stokes_rng)
        rsltFwd = dict(fit_I=I, fit_Q=Q, fit_U=U, vis=vis, fis=fis, sza=sza)
        rsltFwd['lambda'] = wvls                                   # key is literally 'lambda' (matches GRASP rslt)
        # ---- mirror populateFromRslt's inner per-wavelength errorModel call ----
        for l in range(nwl):
            out = nowPix.measVals[l]['errorModel'](l, rsltFwd)     # -> addError -> customErrModel
            pI, pQ, pU = out[:nang], out[nang:2*nang], out[2*nang:]
            tI, tQ, tU = I[:, l], Q[:, l], U[:, l]
            d['I'].append(pI - tI)
            d['Q'].append(pQ - tQ)
            d['U'].append(pU - tU)
            d['DoLP'].append(dolp(pI, pQ, pU) - dolp(tI, tQ, tU))
    return {k: np.concatenate(v) for k, v in d.items()}


def main():
    n_pix = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    np.random.seed(0)                     # reproducible errModel noise (uses global np.random)
    stokes_rng = np.random.default_rng(1) # reproducible truth Stokes
    cem.init_store()                      # load real cal-sim matrices + covariance (header paths)
    print('err_sim store: n_instr=%d (pool), cov diag[0]=%.3e'
          % (cem.get_store().n_instr, np.diag(cem.get_store().cov_C)[0]))

    comps = ['I', 'Q', 'U', 'DoLP']
    fig, axes = plt.subplots(len(ARCHS), len(comps), figsize=(4*len(comps), 3.4*len(ARCHS)),
                             squeeze=False)
    for r, (arch, label) in enumerate(ARCHS):
        deltas = run_arch(arch, n_pix, stokes_rng)
        print('\n=== %s (%s), %d pixels ===' % (label, arch, n_pix))
        for c, comp in enumerate(comps):
            x = deltas[comp]
            x = x[np.isfinite(x)]
            mu, sd = float(np.mean(x)), float(np.std(x))
            print('  d%-4s  mean=%+.3e  std=%.3e  N=%d' % (comp, mu, sd, x.size))
            ax = axes[r][c]
            ax.hist(x, bins=80, color='steelblue', alpha=0.85)
            ax.axvline(0.0, color='k', lw=0.8)
            ax.axvline(mu, color='crimson', lw=1.0, ls='--')
            ax.set_title(r'%s  $\Delta$%s' % (label, comp), fontsize=10)
            ax.set_xlabel(r'perturbed $-$ truth')
            ax.text(0.02, 0.97, 'mean=%+.2e\nstd=%.2e' % (mu, sd), transform=ax.transAxes,
                    va='top', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    fig.suptitle('err_sim perturbation: (perturbed - truth), %d pixels' % n_pix, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=120)
    print('\nSaved histogram figure -> %s' % OUT_PNG)


if __name__ == '__main__':
    main()
