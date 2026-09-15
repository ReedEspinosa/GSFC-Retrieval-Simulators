#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_path_comparison.py -- overlay the err_sim error-model paths on one figure.

Loads the per-path pickles written by run_experiment.py and draws a multi-panel
truth-vs-retrieved scatter, one colour per path, so the paths can be compared at
a glance:

    Path 1 'harperrsim'    analytic calibration-uncertainty propagation (unbiased)
    Path 2 'harperrsimmc'  Monte Carlo sensor-space noise (random + calibration bias)
    Path 3 'harperrsimbck' the BCK YAML's own assumed noise (self-consistent control)

Only scenes common to all loaded paths are drawn, matched on the truth AOD, so
every path is scored on an identical set of pixels (a path that loses retrievals
to GRASP inversion failures would otherwise be compared on a different sample).

Usage:
    python err_sim/plot_path_comparison.py [waveIndex]      # default 1 (0.549 um)
"""

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
import err_sim.np_compat  # noqa: F401,E402  -- restores np.trapz for NumPy>=2
import simulateRetrieval as rs  # noqa: E402

# --- paths to compare: (label, instrument/arch name, categorical colour slot) ---
# Colours are the validated categorical slots 1/2/3, assigned in FIXED order so a
# path keeps its colour no matter which subset is plotted.
PATHS = [
    ('Path 1 - analytic',    'harperrsim',    '#2a78d6'),   # slot 1 blue
    ('Path 2 - Monte Carlo', 'harperrsimmc',  '#eb6834'),   # slot 2 orange
    ('Path 3 - control 1%',  'harperrsimbck', '#1baf7a'),   # slot 3 aqua
]

SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e3e2dd'
REF_LINE = '#8a8985'

# (title, extractor(rslt, waveIndex) -> scalar)
PANELS = [
    ('AOD',            lambda r, w: r['aod'][w]),
    ('SSA',            lambda r, w: r['ssa'][w]),
    ('n (fine)',       lambda r, w: r['n'][0][w]),
    ('n (coarse)',     lambda r, w: r['n'][1][w]),
    ('k (fine)',       lambda r, w: r['k'][0][w]),
    ('r_eff',          lambda r, w: float(r['rEff'])),
    ('r_v (fine)',     lambda r, w: r['rv'][0]),
    ('r_v (coarse)',   lambda r, w: r['rv'][1]),
]


def load_paths():
    """Load each path's pickle; skip any that has not been run yet."""
    out = []
    for label, inst, color in PATHS:
        pkl = os.path.join(_HERE, 'experiment_%s.pkl' % inst)
        if not os.path.isfile(pkl):
            print('SKIP %-22s (no %s)' % (label, os.path.basename(pkl)))
            continue
        out.append((label, color, rs.simulation(picklePath=pkl)))
    if not out:
        raise SystemExit('No experiment pickles found in %s' % _HERE)
    return out


def common_scenes(loaded):
    """Indices, per path, of the scenes present in EVERY path (matched on truth AOD)."""
    keys = [[round(float(f['aod'][1]), 9) for f in sim.rsltFwd] for _, _, sim in loaded]
    shared = set(keys[0])
    for k in keys[1:]:
        shared &= set(k)
    shared = sorted(shared)
    return [[k.index(v) for v in shared] for k in keys], len(shared)


def main():
    wi = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    loaded = load_paths()
    idxPerPath, nCommon = common_scenes(loaded)
    wvl = loaded[0][2].rsltFwd[0]['lambda'][wi]
    print('%d common scenes across %d path(s); wavelength %.3f um'
          % (nCommon, len(loaded), wvl))

    ncol = 4
    nrow = int(np.ceil(len(PANELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow),
                             squeeze=False, facecolor=SURFACE)

    for p, (title, fn) in enumerate(PANELS):
        ax = axes[p // ncol][p % ncol]
        ax.set_facecolor(SURFACE)
        lo, hi = np.inf, -np.inf
        stats = []
        for (label, color, sim), idx in zip(loaded, idxPerPath):
            x = np.array([fn(sim.rsltFwd[i], wi) for i in idx], dtype=float)
            y = np.array([fn(sim.rsltBck[i], wi) for i in idx], dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            x, y = x[good], y[good]
            # >=8px markers, 2px surface ring so overlapping paths stay separable
            ax.scatter(x, y, s=42, color=color, alpha=0.85, label=label,
                       edgecolor=SURFACE, linewidth=0.9, zorder=3)
            lo = min(lo, x.min(), y.min()); hi = max(hi, x.max(), y.max())
            stats.append((label.split(' - ')[0].replace('Path ', 'P'), color,
                          float(np.sqrt(np.mean((y - x) ** 2))), float(np.mean(y - x))))

        pad = 0.06 * (hi - lo + 1e-9)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, ls=(0, (4, 3)), lw=1.2, color=REF_LINE, zorder=1)  # 1:1
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal', adjustable='box')

        ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, pad=7)
        ax.set_xlabel('truth', fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel('retrieved', fontsize=9, color=TEXT_SECONDARY)
        ax.tick_params(labelsize=8, colors=TEXT_SECONDARY, length=3)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)

        # Visible per-path stat labels: identity is never colour-alone, and this is
        # the required relief for the aqua slot's sub-3:1 surface contrast.
        txt = '\n'.join('%s RMSE %.3g  bias %+.3g' % (n, r, b) for n, _, r, b in stats)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=7.2, color=TEXT_SECONDARY, linespacing=1.45,
                bbox=dict(boxstyle='round,pad=0.35', fc=SURFACE, ec=GRID, lw=0.7,
                          alpha=0.92))

    for p in range(len(PANELS), nrow * ncol):
        axes[p // ncol][p % ncol].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(loaded), frameon=False,
               fontsize=10, labelcolor=TEXT_PRIMARY, bbox_to_anchor=(0.5, 0.012),
               handletextpad=0.5, columnspacing=2.2)
    fig.suptitle('err_sim error-model paths: truth vs retrieved at %.3f um  (N=%d scenes)'
                 % (wvl, nCommon), fontsize=13, color=TEXT_PRIMARY, y=0.985)
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])

    out = os.path.join(_HERE, 'path_comparison_%03dnm.png' % int(wvl * 1000))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('Saved -> %s' % out)
    return out


if __name__ == '__main__':
    main()
