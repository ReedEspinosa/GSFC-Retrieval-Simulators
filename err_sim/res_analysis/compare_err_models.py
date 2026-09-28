#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_err_models.py -- err_sim's calibration error model vs a stock ACCP one.

Compares retrievals that share EVERYTHING except the measurement error model: same
canonical case, same AOD draw (TAU_SEED), same geometry walk, same retrieval YAML,
same initial-guess and noise seeds.  Truth is bit-identical between runs, so every
difference is the error model (and, where noted, the view-angle sampling).

The stock comparator is `harp02`, whose error string is `polar07`:

    polar07     sigma_I = 3% relative (lognormal), sigma_DoLP = 0.005 absolute
    harperrsim* sigma from the calibration Monte Carlo, ~1% on I

so polar07 injects roughly 3x the intensity noise.  It is also ANGLE-BLIND where
`harp02` is concerned: that architecture ignores the orbital `vza` and uses a fixed
-57..57 fan, while harperrsim honours the orbit's 22..68 deg.  Run the err_sim side
with ERRSIM_IGNORE_VZA=1 to remove that confound.

Usage:
    python err_sim/compare_err_models.py <pklA> <labelA> <pklB> <labelB> [outPng]
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))   # err_sim/res_analysis
_ERRSIM = os.path.dirname(_HERE)                     # err_sim
_REPO = os.path.dirname(_ERRSIM)                     # GSFC-Retrieval-Simulators
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402
import simulateRetrieval as rs  # noqa: E402

C_A, C_B = '#2a78d6', '#eb6834'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e3e2dd'
DIVERGE = 0.5     # |AOD residual| above which a pixel is a failed inversion


def _style(ax, xl, yl, ti):
    ax.set_facecolor(SURFACE)
    ax.set_title(ti, fontsize=10.5, color=INK, pad=7)
    ax.set_xlabel(xl, fontsize=9, color=INK2)
    ax.set_ylabel(yl, fontsize=9, color=INK2)
    ax.tick_params(labelsize=8, colors=INK2, length=3)
    ax.grid(True, color=GRID, lw=.7)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)


def stats(sim, wi=1):
    t = np.array([f['aod'][wi] for f in sim.rsltFwd], float)
    r = np.array([b['aod'][wi] for b in sim.rsltBck], float)
    keep = np.abs(r - t) <= DIVERGE
    ts = np.array([f['ssa'][wi] for f in sim.rsltFwd], float)
    rsa = np.array([b['ssa'][wi] for b in sim.rsltBck], float)
    return dict(t=t, r=r, keep=keep, ts=ts, rs=rsa)


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    pA, lA, pB, lB = sys.argv[1:5]
    outPng = sys.argv[5] if len(sys.argv) > 5 else os.path.join(_ERRSIM, 'err_model_comparison.png')
    A, B = rs.simulation(picklePath=pA), rs.simulation(picklePath=pB)
    n = min(len(A.rsltBck), len(B.rsltBck))
    wv = A.rsltFwd[0]['lambda']

    sA, sB = stats(A), stats(B)
    # Tolerance, not exact equality: GRASP writes its output text to 5 decimals, so
    # paired runs can differ by 1e-5 on a handful of pixels purely from rounding.
    # That is 4 orders of magnitude below the RMSE being compared.
    dTruth = np.abs(sA['t'][:n] - sB['t'][:n])
    same = dTruth.max() <= 2e-5
    print('=' * 74)
    print('%s  vs  %s        %d pixels' % (lA, lB, n))
    print('=' * 74)
    print('truth scenes paired: %s  (max|diff| %.1e on %d/%d pixels; GRASP prints 5 dp)%s'
          % (same, dTruth.max(), (dTruth > 1e-9).sum(), n, '' if same else
             '\n  <-- NOT PAIRED, differences are not attributable to the error model'))

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9), facecolor=SURFACE)
    axes = ax.ravel()

    # --- 1: retrieved vs truth, both models -------------------------------
    a = axes[0]
    for s, lbl, c in ((sA, lA, C_A), (sB, lB, C_B)):
        k = s['keep'][:n]
        a.scatter(s['t'][:n][k], s['r'][:n][k], s=8, alpha=.35, color=c,
                  edgecolor='none', label=lbl, zorder=3)
    lim = (0, max(sA['t'][:n].max(), sB['t'][:n].max()) * 1.05)
    a.plot(lim, lim, ls=(0, (4, 3)), lw=1.2, color=INK2, zorder=2)
    a.set_xlim(lim); a.set_ylim(lim)
    a.legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower right')
    _style(a, 'truth AOD (%.3f um)' % wv[1], 'retrieved AOD', 'Retrieved vs truth')

    # --- 2: residual distribution -----------------------------------------
    a = axes[1]
    bins = np.linspace(-0.2, 0.2, 61)
    for s, lbl, c in ((sA, lA, C_A), (sB, lB, C_B)):
        k = s['keep'][:n]
        d = (s['r'][:n] - s['t'][:n])[k]
        a.hist(d, bins=bins, histtype='step', lw=1.8, color=c,
               label='%s  bias %+.4f  RMSE %.4f' % (lbl, d.mean(), np.sqrt((d ** 2).mean())))
    a.axvline(0, color=INK2, lw=.9)
    a.legend(frameon=False, fontsize=8, labelcolor=INK)
    _style(a, 'AOD residual (retrieved - truth)', 'pixels', 'Residual distribution')

    # --- 3: paired per-pixel difference ------------------------------------
    a = axes[2]
    k = sA['keep'][:n] & sB['keep'][:n]
    dA = (sA['r'][:n] - sA['t'][:n])[k]
    dB = (sB['r'][:n] - sB['t'][:n])[k]
    a.scatter(dA, dB, s=9, alpha=.4, color=C_A, edgecolor='none', zorder=3)
    lim = (min(dA.min(), dB.min()) * 1.05, max(dA.max(), dB.max()) * 1.05)
    a.plot(lim, lim, ls=(0, (4, 3)), lw=1.2, color=INK2, zorder=2)
    a.axhline(0, color=INK2, lw=.7); a.axvline(0, color=INK2, lw=.7)
    a.set_xlim(lim); a.set_ylim(lim)
    a.text(.03, .97, 'same %d pixels, same truth\ncorrelation r = %+.3f'
           % (k.sum(), np.corrcoef(dA, dB)[0, 1]), transform=a.transAxes, va='top',
           fontsize=8, color=INK2,
           bbox=dict(boxstyle='round,pad=.35', fc=SURFACE, ec=GRID, lw=.7))
    _style(a, '%s residual' % lA, '%s residual' % lB, 'Paired per-pixel residuals')

    # --- 4: spectral RMSE ---------------------------------------------------
    a = axes[3]
    for sim, s, lbl, c in ((A, sA, lA, C_A), (B, sB, lB, C_B)):
        rm = []
        for w in range(len(wv)):
            t = np.array([f['aod'][w] for f in sim.rsltFwd], float)[:n][s['keep'][:n]]
            r = np.array([b['aod'][w] for b in sim.rsltBck], float)[:n][s['keep'][:n]]
            rm.append(np.sqrt(((r - t) ** 2).mean()))
        a.plot(wv, rm, marker='o', ms=6, lw=1.8, color=c, label=lbl)
    a.legend(frameon=False, fontsize=8, labelcolor=INK)
    _style(a, 'wavelength (um)', 'AOD RMSE', 'Spectral AOD RMSE')

    fig.suptitle('Measurement error model comparison on IDENTICAL truth scenes',
                 fontsize=12.5, color=INK, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(outPng, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- console table ----
    print()
    print('%-26s %10s %10s %10s %10s' % ('', 'bias', 'RMSE', 'diverged', 'SSA bias'))
    for sim, s, lbl in ((A, sA, lA), (B, sB, lB)):
        k = s['keep'][:n]
        d = (s['r'][:n] - s['t'][:n])[k]
        ds = (s['rs'][:n] - s['ts'][:n])[k]
        print('%-26s %+10.4f %10.4f %10d %+10.4f'
              % (lbl, d.mean(), np.sqrt((d ** 2).mean()), (~k).sum(), ds.mean()))
    print()
    print('spectral AOD RMSE')
    print('%-26s' % '' + ''.join('%10.3f' % w for w in wv))
    for sim, s, lbl in ((A, sA, lA), (B, sB, lB)):
        row = []
        for w in range(len(wv)):
            t = np.array([f['aod'][w] for f in sim.rsltFwd], float)[:n][s['keep'][:n]]
            r = np.array([b['aod'][w] for b in sim.rsltBck], float)[:n][s['keep'][:n]]
            row.append(np.sqrt(((r - t) ** 2).mean()))
        print('%-26s' % lbl + ''.join('%10.4f' % v for v in row))
    print()
    print('Saved figure -> %s' % outPng)
    return 0


if __name__ == '__main__':
    sys.exit(main())
