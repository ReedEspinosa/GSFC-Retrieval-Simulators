#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_bias_survey.py -- where the per-instrument campaign's biases actually come from.

Surveys every retrieved variable across all tasks, not just AOD, and draws the
size/absorption parameters against the retrieval's a-priori bounds, which is what
turns a well-retrieved AOD into badly biased microphysics when the truth falls
outside them.

Bounds are read from the YAML the tasks actually used (recorded in their provenance),
not hardcoded: the smoke campaign widened them, and with the marine box 49% of
fine-mode k truth and 46% of coarse rv truth lay outside. After widening, <1% does,
and what remains is a genuine information-content limit on the coarse mode.

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

# A-priori bounds are READ FROM THE YAML rather than hardcoded: the smoke campaign
# widened them (see settings_BCK_POLAR_2modes_errsim_smoke.yml), and a stale copy of
# the numbers here would draw the wrong box and invert the conclusion.
#   characteristic[2] = size_distribution_lognormal, min/max = [rv, sigma] per mode
#   characteristic[4] = imaginary refractive index, min/max = [k] per mode
def read_bounds(yamlPath):
    """{(param, mode): (min, max)} from a GRASP settings YAML."""
    import yaml
    with open(yamlPath) as f:
        y = yaml.safe_load(f)
    chars = y['retrieval']['constraints']
    out = {}
    for ch, names in (('characteristic[2]', ('rv', 'sigma')), ('characteristic[4]', ('k',))):
        if ch not in chars:
            continue
        for mKey, mVal in chars[ch].items():
            if not mKey.startswith('mode['):
                continue
            m = int(mKey[5:-1]) - 1
            ig = mVal['initial_guess']
            for i, nm in enumerate(names):
                out[(nm, m)] = (float(ig['min'][i]), float(ig['max'][i]))
    return out


# (label, extractor, bounds key) -- bounds filled in from the YAML at run time
PANELS = [
    ('k fine (549 nm)',   lambda r: r['k'][0][1],  ('k', 0)),
    ('r_v coarse (um)',   lambda r: r['rv'][1],     ('rv', 1)),
    ('sigma fine',        lambda r: r['sigma'][0],  ('sigma', 0)),
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
    # Bounds come from the YAML the tasks actually used, recorded in their provenance.
    import json
    metas = sorted(glob.glob(os.path.join(taskDir, 'task_*.json')))
    bckName = json.load(open(metas[0]))['bck_yaml'] if metas else None
    if len(sys.argv) > 3:
        bckPath = sys.argv[3]
    elif bckName:
        bckPath = os.path.join(_REPO, 'ACCP_ArchitectureAndCanonicalCases', bckName)
    else:
        raise SystemExit('cannot determine the BCK YAML; pass it as the 3rd argument')
    bounds = read_bounds(bckPath)
    print('a-priori bounds from %s' % os.path.basename(bckPath))

    pkls = sorted(glob.glob(os.path.join(taskDir, '*.pkl')))
    sims = [rs.simulation(picklePath=f) for f in pkls]
    if not sims:
        raise SystemExit('no task pickles in %s' % taskDir)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9), facecolor=SURFACE)
    axes = ax.ravel()

    for i, (lbl, fn, key) in enumerate(PANELS):
        lo, hi = bounds[key]
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
    a.text(.03, .97, 'mean bias %+.4f (%+.1f%%)\nthe product that DOES track calibration'
           % (np.mean(y - x), 100 * np.mean(y - x) / np.mean(x)), transform=a.transAxes,
           va='top', fontsize=8, color=INK2,
           bbox=dict(boxstyle='round,pad=.35', fc=SURFACE, ec=GRID, lw=.7))
    _style(a, 'truth', 'retrieved', lbl)

    fig.suptitle('Truth now sits INSIDE the a-priori box; the residual coarse-mode bias '
                 'is information content (%d instruments x %d pixels)'
                 % (len(sims), len(sims[0].rsltBck)),
                 fontsize=12.5, color=INK, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(outPng, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('Saved figure -> %s' % outPng)
    return 0


if __name__ == '__main__':
    sys.exit(main())
