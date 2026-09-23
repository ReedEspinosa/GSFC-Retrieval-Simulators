#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_tasks.py -- what the per-instrument campaign actually shows.

Reads the per-task summary produced by ``collect_tasks.py`` plus each task's
provenance JSON, recovers the calibration matrices those tasks actually used from
the cal-sim HDF5, and asks the question the campaign was built for:

    does a more poorly calibrated instrument retrieve a more biased aerosol product?

Because the campaign fixes scenes, initial guess and noise draws across tasks, every
difference between tasks is attributable to the instrument + calibration event.

Calibration quality is summarised two ways, both derived from M = C @ A, which would
be the identity for a perfect calibration (C is meant to invert the instrument's
characteristic response A):

    calDev   ||M - I||_F      total calibration error, all channels
    iScale   M[0,0] - 1       relative radiometric error on the I channel, i.e. the
                              scene appears this much brighter/darker than truth

iScale is the physically motivated predictor of AOD bias: over a dark ocean a scene
that reads brighter than truth is fitted with more aerosol.

Usage:
    python err_sim/analyze_tasks.py [taskDir] [outPng]
"""

import glob
import json
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
import err_sim.customErrModel as cem  # noqa: E402
import simulateRetrieval as rs  # noqa: E402

# Validated categorical slots (fixed order, never cycled).
C_MAIN = '#2a78d6'      # slot 1 blue
C_ACCENT = '#eb6834'    # slot 2 orange -- used only for reference lines/fits
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e3e2dd'


def calibration_metrics(taskDir):
    """Per task: the calibration error of the matrices it actually used.

    Rebuilds C exactly as customErrModel does (radiometric gain folded when
    USE_RAD_CAL), indexes the task's own instrument/calibration, and reduces
    M = C @ A to two scalars averaged over the four bands.
    """
    store = cem.get_store()
    out = {}
    for jf in sorted(glob.glob(os.path.join(taskDir, 'task_*.json'))):
        prov = json.load(open(jf))
        devs, scales = [], []
        for instr in prov['instrument_idx']:
            A = store.char_mats[instr]                      # (3,3) truth response
            C = store.cal_mats[instr][:, :, prov['cal_idx']]  # (3,3) fitted, gain folded
            M = C @ A
            devs.append(np.linalg.norm(M - np.eye(3), 'fro'))
            scales.append(M[0, 0] - 1.0)
        out[prov['task_idx']] = dict(calDev=float(np.mean(devs)),
                                     iScale=float(np.mean(scales)),
                                     cal_idx=prov['cal_idx'])
    return out


def load_tasks(taskDir):
    """Per-task retrieval stats straight from the pickles, plus calibration metrics."""
    met = calibration_metrics(taskDir)
    rows = []
    for jf in sorted(glob.glob(os.path.join(taskDir, 'task_*.json'))):
        prov = json.load(open(jf))
        pkl = jf[:-5] + '.pkl'
        if not os.path.isfile(pkl):
            continue
        sim = rs.simulation(picklePath=pkl)
        wv = sim.rsltFwd[0]['lambda']
        r = dict(task=prov['task_idx'], **met[prov['task_idx']])
        for w in range(len(wv)):
            x = np.array([f['aod'][w] for f in sim.rsltFwd], float)
            y = np.array([b['aod'][w] for b in sim.rsltBck], float)
            r['biasAod%d' % w] = float(np.mean(y - x))
            r['rmseAod%d' % w] = float(np.sqrt(np.mean((y - x) ** 2)))
            xs = np.array([f['ssa'][w] for f in sim.rsltFwd], float)
            ys = np.array([b['ssa'][w] for b in sim.rsltBck], float)
            r['biasSsa%d' % w] = float(np.mean(ys - xs))
        rows.append(r)
    return rows, wv


def _style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10.5, color=INK, pad=7)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(labelsize=8, colors=INK2, length=3)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)


def _fit(ax, x, y):
    """Least-squares line + Pearson r, drawn only when both vary."""
    if np.ptp(x) == 0 or np.ptp(y) == 0 or len(x) < 3:
        return np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, m * xs + b, color=C_ACCENT, lw=1.6, ls='--', zorder=4,
            label='fit: r = %+.2f' % r)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return r


def main():
    taskDir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, 'tasks', 'tasks')
    outPng = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, 'task_analysis.png')
    cem.init_store()
    rows, wv = load_tasks(taskDir)
    n = len(rows)
    g = lambda k: np.array([r[k] for r in rows], float)

    wi = 1                                      # 549 nm
    bias, rmse = g('biasAod%d' % wi), g('rmseAod%d' % wi)
    calDev, iScale = g('calDev'), g('iScale')

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6), facecolor=SURFACE)

    # 1 -- spread of per-instrument AOD bias
    a = ax[0][0]
    a.hist(bias, bins=14, color=C_MAIN, alpha=.85, edgecolor='white', lw=.6)
    a.axvline(0, color=INK2, lw=1)
    a.axvline(bias.mean(), color=C_ACCENT, lw=1.6, ls='--',
              label='mean %+.4f' % bias.mean())
    a.legend(frameon=False, fontsize=8, labelcolor=INK)
    _style(a, 'AOD bias (retrieved - truth)', 'instruments',
           'Per-instrument AOD bias at %.3f um  (N=%d)' % (wv[wi], n))

    # 2 -- the campaign's core question
    a = ax[0][1]
    a.scatter(iScale * 100, bias, s=44, color=C_MAIN, alpha=.85,
              edgecolor=SURFACE, lw=.8, zorder=3)
    a.axhline(0, color=INK2, lw=.9)
    a.axvline(0, color=INK2, lw=.9)
    rIs = _fit(a, iScale * 100, bias)
    _style(a, 'radiometric error on I,  (C@A)[0,0] - 1  [%]', 'AOD bias',
           'AOD bias vs calibration radiometric error')

    # 3 -- total calibration error vs scatter
    a = ax[1][0]
    a.scatter(calDev, rmse, s=44, color=C_MAIN, alpha=.85,
              edgecolor=SURFACE, lw=.8, zorder=3)
    rCd = _fit(a, calDev, rmse)
    _style(a, r'calibration error  $\|CA-I\|_F$', 'AOD RMSE',
           'AOD RMSE vs total calibration error')

    # 4 -- spectral behaviour: wavelength is ORDERED, so it goes on the axis
    a = ax[1][1]
    mu = [np.mean(g('biasAod%d' % w)) for w in range(len(wv))]
    sd = [np.std(g('biasAod%d' % w)) for w in range(len(wv))]
    a.errorbar(wv, mu, yerr=sd, marker='o', ms=7, lw=1.8, capsize=4,
               color=C_MAIN, ecolor=C_MAIN, mfc=C_MAIN, mec=SURFACE)
    a.axhline(0, color=INK2, lw=.9)
    _style(a, 'wavelength (um)', 'AOD bias',
           'Spectral AOD bias: mean +/- spread across instruments')

    fig.suptitle('err_sim per-instrument campaign: %d instruments, %d pixels each'
                 % (n, 30), fontsize=12.5, color=INK, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(outPng, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- console summary ----
    print('=' * 72)
    print('per-instrument campaign: %d tasks' % n)
    print('=' * 72)
    print('calibration error across instruments')
    print('  ||CA-I||_F      : %.4f +/- %.4f  (min %.4f, max %.4f)'
          % (calDev.mean(), calDev.std(), calDev.min(), calDev.max()))
    print('  I-scale error   : %+.3f%% +/- %.3f%%'
          % (iScale.mean() * 100, iScale.std() * 100))
    print()
    print('retrieval error at %.3f um' % wv[wi])
    print('  AOD bias        : %+.4f +/- %.4f  (spread ACROSS instruments)'
          % (bias.mean(), bias.std()))
    print('  AOD RMSE        : %.4f +/- %.4f  (scatter WITHIN each scene)'
          % (rmse.mean(), rmse.std()))
    print('  ratio spread/RMSE: %.2f' % (bias.std() / rmse.mean()))
    print()
    print('correlations (the trend question)')
    print('  AOD bias  vs I-scale error  : r = %+.3f' % rIs)
    print('  AOD RMSE  vs ||CA-I||_F     : r = %+.3f' % rCd)
    print()
    print('spectral AOD bias (mean +/- across-instrument spread)')
    for w, wl in enumerate(wv):
        print('  %.3f um : %+.4f +/- %.4f' % (wl, mu[w], sd[w]))
    print()
    print('Saved figure -> %s' % outPng)
    return 0


if __name__ == '__main__':
    sys.exit(main())
