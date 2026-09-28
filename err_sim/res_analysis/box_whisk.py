#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""box_whisk.py -- one figure summarising how well every retrieved parameter did.

Answers "how good was this retrieval, across the board" and "how do two campaigns
compare", without leafing through 18 one2one figures per campaign.

    python err_sim/res_analysis/box_whisk.py \\
        err_sim/tasks/tasks_250_smokeOcean/tasks:"I,Q,U" \\
        err_sim/tasks_dolp:"I,DoLP"

Each campaign directory may be suffixed with ':label'; without one the directory name
is used.  Any number of campaigns can be passed.

WHAT THE BOXES ARE.  Each box is the distribution over the campaign's 250 TASKS -- one
task is one simulated instrument with one calibration event.  Per task we compute that
instrument's bias and RMSE over all 526 pixels, so the box shows the
instrument-to-instrument spread and the whiskers the 5th-95th percentile of it.  This
is the only framing that gives RMSE a distribution at all: pooled over pixels RMSE is
a single number, whereas per task there are 250 of them.  Because the campaigns are
PAIRED (identical scenes, noise and initial guess), differences between two campaigns'
boxes are attributable to what actually changed between them.

WHY PERCENT.  Parameters span wildly different units -- AOD bias ~0.01, layer height
bias ~-300 m, k bias ~-0.005.  Everything is therefore normalised by the mean truth of
that parameter, so one y-axis can carry all of them.  Raw units are printed in the
console table.

CHOICES MADE (override with flags):
  * spectral parameters are taken at --wvl (default 0.549 um)
  * two-mode parameters use the FINE mode (--mode fine|coarse|total).  The smoke case
    is fine-dominated (fine volume 0.109 vs coarse 0.035) and the coarse mode is known
    to be weakly constrained over dark ocean, so the fine mode is the informative one.
  * aodMode / ssaMode are omitted as redundant with aod / ssa.  r_eff uses rEffMode,
    NOT the scalar rEff: the scalar is the mode-integrated effective radius, which is
    dominated by the badly-retrieved coarse mode and so is not comparable with the
    fine-mode figures beside it (it reads +105% bias against +2% for r_v).
  * sph is omitted: its truth is stored in PERCENT (99.999) while the retrieval bounds
    are a fraction (0.999-1.0), so the comparison is meaningless until that is fixed.
  * pixels whose AOD inversion diverged are excluded, as everywhere else.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ERRSIM = os.path.dirname(_HERE)
_REPO = os.path.dirname(_ERRSIM)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402
from one2one import load_campaign, VARS, SURFACE, INK, INK2, GRID  # noqa: E402

_PNGDIR = os.path.join(_HERE, 'pngs', 'box_whisker')
os.makedirs(_PNGDIR, exist_ok=True)

# Validated categorical slots, fixed order, never cycled.  Campaigns are ALSO labelled
# on the legend, so identity never rests on colour alone.
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']

# (key, display name, group) in plotting order -- grouped so related parameters sit
# together on the x-axis.
PARAMS = [
    ('aod',        'AOD',          'optical'),
    ('ssa',        'SSA',          'optical'),
    ('LidarRatio', 'lidar ratio',  'optical'),
    ('rv',         r'r$_v$',       'size'),
    ('sigma',      r'$\sigma$',    'size'),
    ('rEffMode',   r'r$_{eff}$',   'size'),
    ('vol',        'volume',       'size'),
    ('n',          'n',            'refr. index'),
    ('k',          'k',            'refr. index'),
    ('height',     'height',       'vertical'),
    ('windSpd',    'wind speed',   'surface'),
]


# --- derived parameters -----------------------------------------------------
# surface_water_cox_munk_iso is ISOTROPIC: it has no wind-direction term at all.  Its
# three components are [0] water-leaving reflectance, [1] Fresnel fraction and
# [2] the Cox-Munk slope variance, which is the only one carrying wind information.
# Cox & Munk (1954): total slope variance = 0.003 + 0.00512*W, and canonicalCaseMap
# stores HALF of it (per-component), so W = (2*sigma^2 - 0.003)/0.00512.  Converting
# back to m/s makes the relative error physically meaningful -- a percentage of a
# slope variance is not.
#
# [1] is fixed at 1.0 by bounds [0.999998, 1.0] and has no freedom, so it is omitted.
# [0] is omitted too, and is worth a separate look: its truth is 1e-7 while the YAML
# bounds it to [1e-9, 3e-9], i.e. the TRUTH SITS ABOVE THE A-PRIORI MAXIMUM and the
# retrieval is pinned (it returns 1.9e-9 against a truth of 1e-7).
DERIVED = {
    'windSpd': dict(src='wtrSurf', comp=2, label='wind speed',
                    fn=lambda v: (2.0 * v - 0.003) / 0.00512),
}


def _extract(var, T, R, wi, mode):
    """Reduce a variable's leading axes to a flat per-pixel pair."""
    if var in DERIVED:
        d = DERIVED[var]
        return d['fn'](T[d['comp']][wi]), d['fn'](R[d['comp']][wi])
    kind = VARS[var]['kind']
    if kind == 'spectral':
        return T[wi], R[wi]
    if kind == 'spectral_mode':
        if mode == 'total':
            return T.sum(axis=0)[wi], R.sum(axis=0)[wi]
        m = 0 if mode == 'fine' else 1
        return T[m][wi], R[m][wi]
    if kind == 'mode':
        if mode == 'total':
            return T.sum(axis=0), R.sum(axis=0)
        m = 0 if mode == 'fine' else 1
        return T[m], R[m]
    return T, R


def per_task(var, T, R, keep, wi, mode, nTask):
    """(bias%, rmse%) per task for one parameter, plus its raw-unit counterparts.

    T/R are (..., nPix*nTask).  Leading axes are reduced first -- wavelength by
    selection, mode by selection or summation -- then the pixel axis is split back into
    (nTask, nPix) so each task gets its own statistics.
    """
    t, r = _extract(var, T, R, wi, mode)

    m = keep & np.isfinite(t) & np.isfinite(r)
    nPix = t.size // nTask
    t2 = t.reshape(nTask, nPix)
    r2 = r.reshape(nTask, nPix)
    m2 = m.reshape(nTask, nPix)

    bias, rmse, scale = [], [], []
    for i in range(nTask):
        sel = m2[i]
        if sel.sum() < 2:
            bias.append(np.nan); rmse.append(np.nan); scale.append(np.nan); continue
        d = r2[i][sel] - t2[i][sel]
        bias.append(d.mean())
        rmse.append(np.sqrt((d ** 2).mean()))
        scale.append(np.abs(t2[i][sel]).mean())
    bias, rmse, scale = map(np.asarray, (bias, rmse, scale))
    with np.errstate(divide='ignore', invalid='ignore'):
        return 100 * bias / scale, 100 * rmse / scale, bias, rmse


def pooled(var, T, R, keep, wi, mode):
    """Per-PIXEL relative error (%) pooled over every task, plus mean bias and RMSE.

    This is the "typical retrieval" distribution: every retrieval from every
    calibration in one population.  Median and percentiles are used rather than RMSE
    because RMSE is not robust here -- 0.42% of pixels (2.2 per task, only 7-8 distinct
    scenes) blow past 100% error on r_eff while passing the AOD divergence mask, and a
    SINGLE such pixel can carry 90% of a task's mean-square error.  The median error on
    the same data is ~8%.
    """
    t, r = _extract(var, T, R, wi, mode)
    m = keep & np.isfinite(t) & np.isfinite(r) & (t != 0)
    t, r = t[m], r[m]
    d = r - t
    rel = 100.0 * d / t
    scale = np.abs(t).mean()
    return rel, 100.0 * d.mean() / scale, 100.0 * np.sqrt((d ** 2).mean()) / scale


def _style(ax, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
    if title:
        ax.set_title(title, fontsize=11, color=INK, pad=7)
    ax.tick_params(labelsize=9, colors=INK2, length=3)
    ax.grid(True, axis='y', color=GRID, lw=.7)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('campaigns', nargs='+', help="taskDir[:label]")
    ap.add_argument('--outPng', default=None)
    ap.add_argument('--wvl', type=float, default=0.549)
    ap.add_argument('--mode', choices=('fine', 'coarse', 'total'), default='fine')
    ap.add_argument('--whis', type=float, nargs=2, default=(5.0, 95.0),
                    help='whisker percentiles (default 5 95)')
    ap.add_argument('--rmse-bars', action='store_true',
                    help='overlay +/- mean RMSE on each box. Off by default because '
                         'RMSE here is outlier-dominated (a few blown-up scenes) and '
                         'the bars dwarf the actual distribution.')
    ap.add_argument('--clip', type=float, default=None,
                    help='clip the y-axes to +/- this percent, for readability')
    a = ap.parse_args()

    keys = [p[0] for p in PARAMS]
    # derived parameters need their SOURCE array loaded, not themselves
    loadKeys = sorted({DERIVED[k]['src'] if k in DERIVED else k for k in keys}
                      | {'aod'})
    campaigns = []
    for spec in a.campaigns:
        d, _, lbl = spec.partition(':')
        d = d.rstrip('/')
        lbl = lbl or (os.path.basename(os.path.dirname(d))
                      if os.path.basename(d) == 'tasks' else os.path.basename(d))
        data, nTask, lam, keep = load_campaign(d, loadKeys)
        wi = int(np.argmin(np.abs(lam - a.wvl)))
        res, pool = {}, {}
        for k in keys:
            T, R = data[DERIVED[k]['src'] if k in DERIVED else k]
            res[k] = per_task(k, T, R, keep, wi, a.mode, nTask)
            pool[k] = pooled(k, T, R, keep, wi, a.mode)
        campaigns.append(dict(label=lbl, dir=d, nTask=nTask, res=res, pooled=pool,
                              wvl=float(lam[wi]), nDiv=int((~keep).sum())))
        print('%-46s %d tasks, %d diverged pixels masked' % (d, nTask, int((~keep).sum())))

    nC = len(campaigns)
    width = 0.8 / nC
    xs = np.arange(len(PARAMS))
    fig, ax = plt.subplots(figsize=(max(11.0, 1.3 * len(PARAMS) + 3), 7.2),
                           facecolor=SURFACE)

    for ci, c in enumerate(campaigns):
        off = (ci - (nC - 1) / 2) * width
        vals = [c['pooled'][k][0] for k in keys]
        bp = ax.boxplot(vals, positions=xs + off, widths=width * 0.82,
                        whis=a.whis, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.5),
                        boxprops=dict(facecolor=SLOTS[ci % len(SLOTS)], alpha=.70,
                                      edgecolor=SLOTS[ci % len(SLOTS)], lw=1.0),
                        whiskerprops=dict(color=SLOTS[ci % len(SLOTS)], lw=1.0),
                        capprops=dict(color=SLOTS[ci % len(SLOTS)], lw=1.0))
        bp['boxes'][0].set_label(c['label'])
        # mean bias across all calibrations -- the "typical" offset
        mb = [c['pooled'][k][1] for k in keys]
        ax.plot(xs + off, mb, marker='D', ms=5, ls='none', mfc='white',
                mec=INK, mew=1.1, zorder=6,
                label='mean bias' if ci == 0 else None)
        if a.rmse_bars:
            rm = [c['pooled'][k][2] for k in keys]
            ax.errorbar(xs + off, mb, yerr=rm, fmt='none', ecolor=INK2, elinewidth=1.0,
                        capsize=3, alpha=.8, zorder=5,
                        label='$\\pm$mean RMSE' if ci == 0 else None)

    ax.axhline(0, color=INK2, lw=1.2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='upper left', ncol=nC + 2)
    if a.clip:
        ax.set_ylim(-a.clip, a.clip)
    _style(ax, 'retrieval error  [% of truth]')

    ax.set_xticks(xs)
    ax.set_xticklabels([p[1] for p in PARAMS], fontsize=10.5, color=INK)
    groups = [p[2] for p in PARAMS]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axvline(i - 0.5, color=GRID, lw=1.0, ls=(0, (3, 3)), zorder=0)
    for g in dict.fromkeys(groups):
        idx = [j for j, gg in enumerate(groups) if gg == g]
        ax.text(np.mean(idx), 1.02, g, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9, color=INK2, style='italic')

    c0 = campaigns[0]
    fig.suptitle('Typical retrieval error by parameter -- %s, %s mode, all %d calibrations '
                 'pooled (box = IQR, whiskers %g-%gth pct, diamond = mean bias)'
                 % ('%.3f $\\mu$m' % c0['wvl'], a.mode, c0['nTask'], *a.whis),
                 fontsize=12, color=INK, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .955])

    out = a.outPng or os.path.join(
        _PNGDIR, 'box_whisk_%s_%s_%03dnm.png'
        % ('_vs_'.join(c['label'].replace(',', '').replace(' ', '') for c in campaigns),
           a.mode, round(c0['wvl'] * 1000)))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- console table, raw units so the normalisation is auditable ----
    print()
    hdr = '%-13s' % 'parameter'
    for c in campaigns:
        hdr += '%24s' % c['label']
    print(hdr)
    print('%-13s' % '' + ''.join('%12s%12s' % ('bias', 'RMSE') for _ in campaigns))
    for k, name, _ in PARAMS:
        row = '%-13s' % k
        for c in campaigns:
            _, _, b, r = c['res'][k]
            row += '%12.4g%12.4g' % (np.nanmedian(b), np.nanmedian(r))
        print(row)
    print('(median over tasks, RAW units; the figure shows relative error as %% of truth)')
    print()
    hdr = '%-13s' % 'parameter'
    for c in campaigns:
        hdr += '%34s' % c['label']
    print(hdr)
    print('%-13s' % '' + ''.join('%11s%11s%12s' % ('median%', 'IQR%', 'meanRMSE%')
                                 for _ in campaigns))
    for k, name, _ in PARAMS:
        row = '%-13s' % k
        for c in campaigns:
            rel, mb, rm = c['pooled'][k]
            q1, q3 = np.percentile(rel, [25, 75])
            row += '%11.2f%11.2f%12.1f' % (np.median(rel), q3 - q1, rm)
        print(row)
    print('(pooled over every retrieval; median/IQR are robust, meanRMSE is NOT --')
    print(' a few blown-up scenes dominate it, see the r_eff note in the docstring)')
    print()
    print('Saved figure -> %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
