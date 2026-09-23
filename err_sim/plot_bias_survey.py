#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_bias_survey.py -- where the per-instrument campaign's biases actually come from.

Surveys every retrieved variable across all tasks, not just AOD, and draws the two
size/absorption parameters whose truth falls OUTSIDE the retrieval's a-priori range
together with those bounds -- which is what turns a well-retrieved AOD into badly
biased microphysics.

Usage:
    python err_sim/plot_bias_survey.py [taskDir] [outPng] [bckYaml]
"""

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402
import simulateRetrieval as rs  # noqa: E402

C_MAIN, C_ACCENT, C_OK = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e3e2dd'

# (label, extractor, a-priori min, max) -- bounds read off the BCK YAML
PANELS = [
    ('k fine (549 nm)',   lambda r: r['k'][0][1],  1e-6, 0.01),
    ('r_v coarse (um)',   lambda r: r['rv'][1],    0.65, 4.9),
    ('sigma fine',        lambda r: r['sigma'][0], 0.25, 0.65),
]
SSA = ('SSA (549 nm)', lambda r: r['ssa'][1])


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


def main():
    taskDir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, 'tasks', 'tasks')
    outPng = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, 'bias_survey.png')
    sims = [rs.simulation(picklePath=f)
            for f in sorted(glob.glob(os.path.join(taskDir, '*.pkl')))]
    if not sims:
        raise SystemExit('no task pickles in %s' % taskDir)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9), facecolor=SURFACE)
    axes = ax.ravel()

    for i, (lbl, fn, lo, hi) in enumerate(PANELS):
        a = axes[i]
        x = np.array([fn(f) for s in sims for f in s.rsltFwd], float)
        y = np.array([fn(b) for s in sims for b in s.rsltBck], float)
        outside = 100 * np.mean((x < lo) | (x > hi))
        a.scatter(x, y, s=9, color=C_MAIN, alpha=.35, edgecolor='none', zorder=3)
        lim = (min(x.min(), y.min(), lo) * .9, max(x.max(), y.max(), hi) * 1.05)
        a.plot(lim, lim, ls=(0, (4, 3)), lw=1.2, color=INK2, zorder=2)
        # the a-priori box the retrieval is allowed to return
        a.axhline(lo, color=C_ACCENT, lw=1.4)
        a.axhline(hi, color=C_ACCENT, lw=1.4)
        a.axvspan(lim[0], lo, color=C_ACCENT, alpha=.07, zorder=0)
        a.axvspan(hi, lim[1], color=C_ACCENT, alpha=.07, zorder=0)
        a.set_xlim(lim); a.set_ylim(lim)
        a.text(.03, .97, 'a-priori bounds %.4g - %.4g\n%.0f%% of TRUTH is outside them'
               % (lo, hi, outside), transform=a.transAxes, va='top', fontsize=8,
               color=INK2, bbox=dict(boxstyle='round,pad=.35', fc=SURFACE, ec=GRID, lw=.7))
        _style(a, 'truth', 'retrieved', lbl)

    # consequence panel: SSA
    a = axes[3]
    lbl, fn = SSA
    x = np.array([fn(f) for s in sims for f in s.rsltFwd], float)
    y = np.array([fn(b) for s in sims for b in s.rsltBck], float)
    a.scatter(x, y, s=9, color=C_OK, alpha=.35, edgecolor='none', zorder=3)
    lim = (min(x.min(), y.min()) * .999, max(x.max(), y.max()) * 1.001)
    a.plot(lim, lim, ls=(0, (4, 3)), lw=1.2, color=INK2, zorder=2)
    a.set_xlim(lim); a.set_ylim(lim)
    a.text(.03, .97, 'mean bias %+.4f (%+.1f%%)\nconsequence of capped fine-mode k'
           % (np.mean(y - x), 100 * np.mean(y - x) / np.mean(x)), transform=a.transAxes,
           va='top', fontsize=8, color=INK2,
           bbox=dict(boxstyle='round,pad=.35', fc=SURFACE, ec=GRID, lw=.7))
    _style(a, 'truth', 'retrieved', lbl)

    fig.suptitle('Microphysical bias is set by the a-priori range, not by calibration '
                 '(%d instruments x %d pixels)' % (len(sims), len(sims[0].rsltBck)),
                 fontsize=12.5, color=INK, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(outPng, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('Saved figure -> %s' % outPng)
    return 0


if __name__ == '__main__':
    sys.exit(main())
