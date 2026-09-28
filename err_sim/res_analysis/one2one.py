#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""one2one.py -- retrieved vs true AOD for a whole campaign, as a density heat-map.

Pools every pixel of every task in ONE task directory (250 tasks x 526 pixels =
131,500 retrievals for the smoke campaigns) and bins them into a 2D histogram, because
at that count a plain scatter is a solid blob that hides where the mass actually sits.

Drawn on top:
  * the 1:1 line in black
  * a shaded cone at +/- RELATIVE_TOL (default 10%) of the true AOD

Run it on either campaign:
    python err_sim/res_analysis/one2one.py err_sim/tasks
    python err_sim/res_analysis/one2one.py err_sim/tasks_dolp

Usage:
    one2one.py [taskDir] [outPng] [--wvl 0.549] [--scale log|linear]
               [--tol 0.10] [--bins 130] [--include-diverged]

Why the default is a LOG scale.  Campaign AOD is drawn lognormally
(TAU_FACTOR='randLogNrm0.3', 95% of draws inside a factor of 4), so truth spans
~0.02 to ~2.7 with most of the mass below 0.5.  On linear axes the entire population
crushes into the bottom-left corner.  A log scale also makes the +/-10% cone a
constant-width band rather than a wedge, which is what makes it readable.  Pass
--scale linear if you want the untransformed view.

A note on the vertical structure.  The campaign is a PAIRED design: every task sees
the same 526 scenes, so the truth axis carries only 526 distinct values with 250
retrievals stacked on each.  The plot is therefore 526 vertical distributions, not a
continuous cloud, and pushing --bins much above ~130 resolves that comb rather than
the density.  The streaks are real -- each one is the spread across instruments for a
single scene -- not a binning artifact.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))   # err_sim/res_analysis
_ERRSIM = os.path.dirname(_HERE)                     # err_sim
_REPO = os.path.dirname(_ERRSIM)                     # GSFC-Retrieval-Simulators
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402  -- must precede runGRASP
import simulateRetrieval as rs  # noqa: E402

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e3e2dd'
CMAP = 'viridis'
CONE_COLOR = '#eb6834'
# A retrieval this far from truth is a failed inversion, not a noisy one.  Three of 526
# smoke scenes do this in nearly every task (one retrieves 0.33 as 8.8) and a single
# such pixel can carry most of the campaign's mean-square error, so they are excluded
# by default and counted in the annotation rather than silently stretching the axes.
DIVERGE = 0.5


def load_campaign(taskDir, wvl):
    """Pooled (truth, retrieved) AOD over every pixel of every task in taskDir."""
    pkls = sorted(glob.glob(os.path.join(taskDir, '*.pkl')))
    if not pkls:
        raise SystemExit('no task pickles found in %s' % taskDir)
    x, y, nTask = [], [], 0
    wi = None
    for p in pkls:
        sim = rs.simulation(picklePath=p)
        if not getattr(sim, 'rsltBck', None):
            continue
        if wi is None:
            lam = np.asarray(sim.rsltFwd[0]['lambda'], float)
            wi = int(np.argmin(np.abs(lam - wvl)))
            wvlActual = float(lam[wi])
        x.append([f['aod'][wi] for f in sim.rsltFwd])
        y.append([b['aod'][wi] for b in sim.rsltBck])
        nTask += 1
    return (np.concatenate(x).astype(float), np.concatenate(y).astype(float),
            nTask, wvlActual)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('taskDir', nargs='?', default=os.path.join(_ERRSIM, 'tasks'))
    ap.add_argument('outPng', nargs='?', default=None)
    ap.add_argument('--wvl', type=float, default=0.549, help='wavelength in um (default 0.549)')
    ap.add_argument('--scale', choices=('log', 'linear'), default='log')
    ap.add_argument('--tol', type=float, default=0.10, help='cone half-width, relative (default 0.10)')
    ap.add_argument('--bins', type=int, default=130)
    ap.add_argument('--include-diverged', action='store_true',
                    help='keep failed inversions (|retrieved-truth| > %.1f)' % DIVERGE)
    ap.add_argument('--title', default=None)
    a = ap.parse_args()

    taskDir = a.taskDir.rstrip('/')
    outPng = a.outPng or os.path.join(taskDir, 'one2one_%s_%03dnm.png'
                                      % (os.path.basename(taskDir), round(a.wvl * 1000)))

    x, y, nTask, wvl = load_campaign(taskDir, a.wvl)
    nAll = x.size
    if not a.include_diverged:
        keep = np.abs(y - x) <= DIVERGE
        nDiv = int((~keep).sum())
        x, y = x[keep], y[keep]
    else:
        nDiv = 0

    # --- binning ---------------------------------------------------------
    pos = (x > 0) & (y > 0)
    if a.scale == 'log':
        # Log bins need strictly positive data; a retrieval can legitimately come back
        # at ~0, so drop those here and report them rather than letting log10 produce
        # -inf and silently empty the histogram.
        nNonPos = int((~pos).sum())
        x, y = x[pos], y[pos]
        lo = max(min(x.min(), y.min()) * 0.85, 1e-4)
        hi = max(x.max(), y.max()) * 1.15
        edges = np.logspace(np.log10(lo), np.log10(hi), a.bins + 1)
    else:
        nNonPos = 0
        lo, hi = 0.0, max(x.max(), y.max()) * 1.05
        edges = np.linspace(lo, hi, a.bins + 1)

    H, xe, ye = np.histogram2d(x, y, bins=[edges, edges])
    H = np.ma.masked_where(H == 0, H)        # empty bins stay background, not dark

    # --- statistics tied to what is drawn --------------------------------
    d = y - x
    within = float(np.mean(np.abs(d) <= a.tol * x) * 100.0)
    bias, rmse = float(d.mean()), float(np.sqrt((d ** 2).mean()))
    # Relative statistics are the honest ones for a quantity spanning two decades.
    relBias = float(np.mean(d / x) * 100.0)

    # --- draw -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 7.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    pcm = ax.pcolormesh(xe, ye, H.T, cmap=CMAP, norm=LogNorm(vmin=1, vmax=H.max()),
                        shading='flat', zorder=2)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02, extend='min')
    cb.set_label('retrievals per bin', fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_edgecolor(GRID)

    # +/- tol cone, then the 1:1 line on top of it
    ln = np.array([lo if lo > 0 else 0.0, hi])
    ax.fill_between(ln, ln * (1 - a.tol), ln * (1 + a.tol), color=CONE_COLOR,
                    alpha=0.18, lw=0, zorder=3,
                    label='$\\pm$%.0f%% of truth' % (a.tol * 100))
    for s in (1 - a.tol, 1 + a.tol):
        ax.plot(ln, ln * s, color=CONE_COLOR, lw=1.1, alpha=0.75, zorder=4)
    ax.plot(ln, ln, color='black', lw=1.6, zorder=5, label='1:1')

    ax.set_xscale(a.scale)
    ax.set_yscale(a.scale)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

    note = ('N = %s retrievals  (%d tasks)\n'
            'within $\\pm$%.0f%%: %.1f%%\n'
            'bias %+.4f   RMSE %.4f\n'
            'mean relative bias %+.1f%%'
            % (format(x.size, ','), nTask, a.tol * 100, within, bias, rmse, relBias))
    dropped = []
    if nDiv:
        dropped.append('%d diverged (|err| > %.1f)' % (nDiv, DIVERGE))
    if nNonPos:
        dropped.append('%d non-positive' % nNonPos)
    if dropped:
        note += '\nexcluded: ' + ', '.join(dropped)
    ax.text(0.035, 0.965, note, transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5, color=INK2, zorder=6,
            bbox=dict(boxstyle='round,pad=.4', fc=SURFACE, ec=GRID, lw=.8, alpha=.92))

    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc='lower right')
    ax.set_xlabel('true AOD at %.3f $\\mu$m' % wvl, fontsize=10, color=INK2)
    ax.set_ylabel('retrieved AOD at %.3f $\\mu$m' % wvl, fontsize=10, color=INK2)
    ax.set_title(a.title or 'Retrieved vs true AOD -- %s' % os.path.basename(taskDir),
                 fontsize=11.5, color=INK, pad=9)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)
    ax.grid(True, color=GRID, lw=.6, zorder=1)
    ax.set_axisbelow(False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)

    fig.tight_layout()
    fig.savefig(outPng, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print('%s: %d tasks, %s retrievals at %.3f um' % (taskDir, nTask, format(nAll, ','), wvl))
    if nDiv:
        print('  excluded %d diverged (|retrieved-truth| > %.1f); --include-diverged keeps them'
              % (nDiv, DIVERGE))
    if nNonPos:
        print('  excluded %d non-positive (log scale)' % nNonPos)
    print('  within +/-%.0f%%: %.1f%%   bias %+.4f   RMSE %.4f   mean rel bias %+.1f%%'
          % (a.tol * 100, within, bias, rmse, relBias))
    print('Saved figure -> %s' % outPng)
    return 0


if __name__ == '__main__':
    sys.exit(main())
