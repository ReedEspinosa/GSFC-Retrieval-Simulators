#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""one2one.py -- retrieved vs true AOD for a whole campaign, as a density heat-map.

Pools every pixel of every task in ONE task directory (250 tasks x 526 pixels =
131,500 retrievals for the smoke campaigns) and bins them into a 2D histogram, because
at that count a plain scatter is a solid blob that hides where the mass actually sits.

Drawn on top:
  * the 1:1 line in black
  * a shaded AERONET-style uncertainty envelope, +/-(ABS_TOL + REL_TOL * AOD)

Run it on either campaign:
    python err_sim/res_analysis/one2one.py err_sim/tasks/tasks_250_smokeOcean/tasks
    python err_sim/res_analysis/one2one.py err_sim/tasks_dolp --all-bands

Usage:
    one2one.py [taskDir] [outPng] [--wvl 0.549] [--all-bands] [--scale log|linear]
               [--rel-tol 0.10] [--abs-tol 0.03] [--bins 130] [--include-diverged]

--all-bands draws a 2x2 panel, one wavelength per panel, each with its own colormap
keyed loosely to the band (blue / green / orange / purple for 441 / 549 / 669 / 873 nm).
All four bands are read in a single pass over the pickles.

THE ENVELOPE.  Default is the AERONET-style form +/-(0.03 + 0.10*AOD) rather than a
pure 10% cone.  A purely relative cone is a far harsher test than it looks at the thin
end -- 10% of AOD 0.05 is +/-0.005, well inside the measurement noise -- so low-AOD
scenes fail it structurally rather than because the retrieval is bad.  The additive
term is what makes the envelope meaningful across two decades of AOD.  Set
--abs-tol 0 to recover the pure relative cone.

Why the default is a LOG scale.  Campaign AOD is drawn lognormally
(TAU_FACTOR='randLogNrm0.3', 95% of draws inside a factor of 4), so truth spans
~0.02 to ~2.7 with most of the mass below 0.5.  On linear axes the entire population
crushes into the bottom-left corner.  Pass --scale linear for the untransformed view.

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
from matplotlib.colors import LogNorm, LinearSegmentedColormap

_HERE = os.path.dirname(os.path.abspath(__file__))   # err_sim/res_analysis
_ERRSIM = os.path.dirname(_HERE)                     # err_sim
_REPO = os.path.dirname(_ERRSIM)                     # GSFC-Retrieval-Simulators

# Every analysis script writes its figures here, so campaign PNGs collect in one place
# instead of scattering next to whichever task directory was passed in.
_PNGDIR = os.path.join(_HERE, 'pngs')
os.makedirs(_PNGDIR, exist_ok=True)

sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402  -- must precede runGRASP
import simulateRetrieval as rs  # noqa: E402

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e3e2dd'
ENVELOPE_COLOR = '#52514e'
DIVERGE = 0.5   # |retrieved - truth| above this is a failed inversion, not a noisy one

# Single-hue sequential maps, loosely keyed to each band so a panel is identifiable at
# a glance.  Single-hue (rather than viridis everywhere) keeps the four panels visually
# distinct; they are truncated below because the pale end of these maps is invisible
# against the near-white page background.
BAND_CMAP = [(0.441, 'Blues'), (0.549, 'Greens'), (0.669, 'Oranges'), (0.873, 'Purples')]
CMAP_FLOOR = 0.25      # drop the lightest quarter of each map


def _truncate(name, lo=CMAP_FLOOR, hi=1.0, n=256):
    """Colormap with its pale low end removed, so sparse bins stay visible."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        '%s_t' % name, base(np.linspace(lo, hi, n)), N=n)


def band_cmap(wvl):
    """Truncated colormap for the band nearest wvl."""
    i = int(np.argmin([abs(w - wvl) for w, _ in BAND_CMAP]))
    return _truncate(BAND_CMAP[i][1])


def load_campaign(taskDir):
    """(truth, retrieved, nTask, wavelengths) pooled over every pixel of every task.

    Reads ALL wavelengths in one pass -- the pickles are the expensive part, and
    --all-bands would otherwise re-read 250 files four times.
    Returned arrays are (nWvl, nRetrievals).
    """
    pkls = sorted(glob.glob(os.path.join(taskDir, '*.pkl')))
    if not pkls:
        raise SystemExit('no task pickles found in %s' % taskDir)
    x, y, nTask, lam = [], [], 0, None
    for p in pkls:
        sim = rs.simulation(picklePath=p)
        if not getattr(sim, 'rsltBck', None):
            continue
        if lam is None:
            lam = np.asarray(sim.rsltFwd[0]['lambda'], float)
        x.append(np.array([f['aod'] for f in sim.rsltFwd], float))
        y.append(np.array([b['aod'] for b in sim.rsltBck], float))
        nTask += 1
    return (np.vstack(x).T, np.vstack(y).T, nTask, lam)


def envelope(v, absTol, relTol):
    """AERONET-style half-width at true AOD v."""
    return absTol + relTol * np.asarray(v, float)


def panel(ax, x, y, a, cmap, showLegend=False):
    """One 1:1 density panel. Returns (pcolormesh, stats dict)."""
    nAll = x.size
    if not a.include_diverged:
        keep = np.abs(y - x) <= DIVERGE
        nDiv = int((~keep).sum())
        x, y = x[keep], y[keep]
    else:
        nDiv = 0

    if a.scale == 'log':
        # Log bins need strictly positive data; a retrieval can come back at ~0, so drop
        # those explicitly and report them rather than letting log10 emit -inf and
        # silently empty the histogram.
        pos = (x > 0) & (y > 0)
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
    pcm = ax.pcolormesh(xe, ye, H.T, cmap=cmap, norm=LogNorm(vmin=1, vmax=H.max()),
                        shading='flat', zorder=2)

    d = y - x
    half = envelope(x, a.abs_tol, a.rel_tol)
    st = dict(n=x.size, nAll=nAll, nDiv=nDiv, nNonPos=nNonPos,
              within=float(np.mean(np.abs(d) <= half) * 100.0),
              bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
              relBias=float(np.mean(d / x) * 100.0))

    # Envelope first, 1:1 on top.  The additive term makes it CURVED in log space, so
    # it needs a resolved line rather than two endpoints.
    ln = (np.logspace(np.log10(max(lo, 1e-6)), np.log10(hi), 400) if a.scale == 'log'
          else np.linspace(lo, hi, 400))
    halfLn = envelope(ln, a.abs_tol, a.rel_tol)
    lbl = ('$\\pm$(%.2f + %.0f%%$\\cdot\\tau$)' % (a.abs_tol, a.rel_tol * 100)
           if a.abs_tol > 0 else '$\\pm$%.0f%% of truth' % (a.rel_tol * 100))
    ax.fill_between(ln, ln - halfLn, ln + halfLn, color=ENVELOPE_COLOR, alpha=0.16,
                    lw=0, zorder=3, label=lbl)
    for sgn in (-1, 1):
        ax.plot(ln, ln + sgn * halfLn, color=ENVELOPE_COLOR, lw=1.0, alpha=0.7, zorder=4)
    ax.plot([lo, hi], [lo, hi], color='black', lw=1.5, zorder=5, label='1:1')

    ax.set_xscale(a.scale); ax.set_yscale(a.scale)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=8, colors=INK2, length=3)
    ax.grid(True, color=GRID, lw=.6, zorder=1)
    ax.set_axisbelow(False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    if showLegend:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower right')
    return pcm, st


def annotate(ax, st, compact=False):
    txt = ('within envelope: %.1f%%\nbias %+.4f   RMSE %.4f'
           % (st['within'], st['bias'], st['rmse']) if compact else
           'N = %s  (%d tasks)\nwithin envelope: %.1f%%\nbias %+.4f   RMSE %.4f\n'
           'mean rel bias %+.1f%%'
           % (format(st['n'], ','), st['nTask'], st['within'], st['bias'], st['rmse'],
              st['relBias']))
    dropped = []
    if st['nDiv']:
        dropped.append('%d diverged' % st['nDiv'])
    if st['nNonPos']:
        dropped.append('%d non-positive' % st['nNonPos'])
    if dropped:
        txt += '\nexcluded: ' + ', '.join(dropped)
    ax.text(0.035, 0.965, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=8 if compact else 8.5, color=INK2, zorder=6,
            bbox=dict(boxstyle='round,pad=.38', fc=SURFACE, ec=GRID, lw=.8, alpha=.92))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('taskDir', nargs='?', default=os.path.join(_ERRSIM, 'tasks'))
    ap.add_argument('outPng', nargs='?', default=None)
    ap.add_argument('--wvl', type=float, default=0.549, help='single-panel wavelength, um')
    ap.add_argument('--all-bands', action='store_true', help='2x2 panel, one per wavelength')
    ap.add_argument('--scale', choices=('log', 'linear'), default='log')
    ap.add_argument('--rel-tol', type=float, default=0.10, help='relative term (default 0.10)')
    ap.add_argument('--abs-tol', type=float, default=0.03,
                    help='additive term (default 0.03; 0 -> pure relative cone)')
    ap.add_argument('--bins', type=int, default=130)
    ap.add_argument('--include-diverged', action='store_true',
                    help='keep failed inversions (|retrieved-truth| > %.1f)' % DIVERGE)
    ap.add_argument('--title', default=None)
    a = ap.parse_args()

    taskDir = a.taskDir.rstrip('/')
    tag = os.path.basename(taskDir)
    if tag == 'tasks':      # .../tasks_250_smokeOcean/tasks -> name it by the campaign
        parent = os.path.basename(os.path.dirname(taskDir))
        if parent and parent != 'err_sim':
            tag = parent
    X, Y, nTask, lam = load_campaign(taskDir)
    envStr = ('+/-(%.2f + %.0f%% * AOD)' % (a.abs_tol, a.rel_tol * 100) if a.abs_tol > 0
              else '+/-%.0f%%' % (a.rel_tol * 100))
    print('%s: %d tasks, %s retrievals, bands %s um'
          % (taskDir, nTask, format(X.shape[1], ','), np.array2string(lam, precision=3)))
    print('envelope %s' % envStr)

    if a.all_bands:
        outPng = a.outPng or os.path.join(_PNGDIR, 'one2one_%s_allbands.png' % tag)
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 12.0), facecolor=SURFACE)
        for i, ax in enumerate(axes.ravel()):
            pcm, st = panel(ax, X[i], Y[i], a, band_cmap(lam[i]), showLegend=(i == 0))
            st['nTask'] = nTask
            annotate(ax, st, compact=True)
            ax.set_title('%.3f $\\mu$m' % lam[i], fontsize=11, color=INK, pad=7)
            ax.set_xlabel('true AOD', fontsize=9, color=INK2)
            ax.set_ylabel('retrieved AOD', fontsize=9, color=INK2)
            cb = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
            cb.ax.tick_params(labelsize=7, colors=INK2)
            cb.outline.set_edgecolor(GRID)
            print('  %.3f um : within %.1f%%   bias %+.4f   RMSE %.4f   %d diverged'
                  % (lam[i], st['within'], st['bias'], st['rmse'], st['nDiv']))
        fig.suptitle(a.title or 'Retrieved vs true AOD, all bands -- %s   (envelope %s)'
                     % (tag, envStr), fontsize=13, color=INK, y=.995)
        fig.tight_layout(rect=[0, 0, 1, .975])
    else:
        wi = int(np.argmin(np.abs(lam - a.wvl)))
        outPng = a.outPng or os.path.join(_PNGDIR, 'one2one_%s_%03dnm.png'
                                          % (tag, round(float(lam[wi]) * 1000)))
        fig, ax = plt.subplots(figsize=(8.4, 7.6), facecolor=SURFACE)
        pcm, st = panel(ax, X[wi], Y[wi], a, band_cmap(lam[wi]), showLegend=True)
        st['nTask'] = nTask
        annotate(ax, st)
        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label('retrievals per bin', fontsize=9, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        cb.outline.set_edgecolor(GRID)
        ax.set_xlabel('true AOD at %.3f $\\mu$m' % lam[wi], fontsize=10, color=INK2)
        ax.set_ylabel('retrieved AOD at %.3f $\\mu$m' % lam[wi], fontsize=10, color=INK2)
        ax.set_title(a.title or 'Retrieved vs true AOD -- %s' % tag,
                     fontsize=11.5, color=INK, pad=9)
        fig.tight_layout()
        print('  %.3f um : within %.1f%%   bias %+.4f   RMSE %.4f   %d diverged'
              % (lam[wi], st['within'], st['bias'], st['rmse'], st['nDiv']))

    fig.savefig(outPng, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('Saved figure -> %s' % outPng)
    return 0


if __name__ == '__main__':
    sys.exit(main())
