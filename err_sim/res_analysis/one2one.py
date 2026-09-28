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
               [--rel-tol 0.10] [--abs-tol 0.03] [--bins 130] [--cbar percent|count]
               [--include-diverged]

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

THE COLOUR AXIS is logarithmic in both modes: bin occupancy spans ~4 decades (a few
retrievals in the tails against >1000 in the dense core), so a linear norm would flatten
everything outside the core into one near-empty shade.  --cbar percent (default) shows
each bin as a SHARE of that panel's retrievals, which is comparable between panels and
between campaigns; --cbar count shows raw occupancy, which is not, because the number of
surviving retrievals differs once the divergence cut is applied.

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
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap

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


# Log is the default, so only a linear run tags its filename -- that keeps the existing
# log filenames stable while stopping the two scales from overwriting each other.
def _SCALE_TAG(scale, cnorm='log'):
    # Log is the default for both, so only non-default runs tag the filename.
    return ('' if scale == 'log' else '_linear') + ('' if cnorm == 'log' else '_cnormlin')


_CBAR_LABEL = {'percent': 'share of retrievals per bin  [%]',
               'count': 'retrievals per bin'}


def _fmt_cbar(cb, mode):
    """Readable decade labels.  Percent shares run from ~8e-4 % to a few %, which the
    default LogFormatter renders as unhelpful mantissa-less ticks."""
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, FuncFormatter
    cb.ax.yaxis.set_major_locator(LogLocator(base=10))
    if mode == 'percent':
        cb.ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: ('%g' % v) if v >= 0.01 else ('%.0e' % v).replace('e-0', 'e-')))
    else:
        cb.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    cb.ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=12))


# =============================================================================
# Variable registry
# =============================================================================
# Each entry: kind, axis scale, and the envelope to draw.  The AOD envelope is the
# AERONET form +/-(0.03 + 0.10*tau); every other quantity needs its OWN tolerance,
# because +/-(0.03 + 10%) is meaningless for a refractive index or a single-scattering
# albedo.  Scales differ too -- SSA lives in 0.9..1.0 and n in 1.33..1.6, where a log
# axis is useless, while k spans decades and needs one.
#
#   kind 'spectral'      (nWvl,)        -> one 2x2 panel, a wavelength per panel
#   kind 'spectral_mode' (nMode, nWvl)  -> one 2x2 panel PER MODE
#   kind 'mode'          (nMode,)       -> one figure, a panel per mode
#   kind 'scalar'        ()             -> a single panel
#
# abs/rel are the envelope half-width terms: half = abs + rel * truth.
VARS = {
    # --- spectral, single-valued -------------------------------------------
    'aod':        dict(kind='spectral',      label='AOD',                 scale='log',
                       abs=0.03, rel=0.10),
    'ssa':        dict(kind='spectral',      label='SSA',                 scale='linear',
                       abs=0.03, rel=0.0),
    'LidarRatio': dict(kind='spectral',      label='lidar ratio (sr)',    scale='linear',
                       abs=0.0,  rel=0.20),
    # --- spectral, per mode -------------------------------------------------
    'k':          dict(kind='spectral_mode', label='imag. refr. index k', scale='log',
                       abs=0.0,  rel=0.30),
    'n':          dict(kind='spectral_mode', label='real refr. index n',  scale='linear',
                       abs=0.02, rel=0.0),
    'aodMode':    dict(kind='spectral_mode', label='modal AOD',           scale='log',
                       abs=0.03, rel=0.10),
    'ssaMode':    dict(kind='spectral_mode', label='modal SSA',           scale='linear',
                       abs=0.03, rel=0.0),
    # --- non-spectral, per mode --------------------------------------------
    'rv':         dict(kind='mode',          label='r$_v$ ($\\mu$m)',      scale='log',
                       abs=0.0,  rel=0.10),
    'sigma':      dict(kind='mode',          label='$\\sigma$',            scale='linear',
                       abs=0.0,  rel=0.10),
    'vol':        dict(kind='mode',          label='volume conc.',        scale='log',
                       abs=0.0,  rel=0.10),
    'sph':        dict(kind='mode',          label='sphere fraction',     scale='linear',
                       abs=0.05, rel=0.0),
    'height':     dict(kind='mode',          label='layer height (m)',    scale='linear',
                       abs=0.0,  rel=0.10),
    'rEffMode':   dict(kind='mode',          label='r$_{eff}$ ($\\mu$m)',  scale='log',
                       abs=0.0,  rel=0.10),
    # --- scalar -------------------------------------------------------------
    'rEff':       dict(kind='scalar',        label='r$_{eff}$ ($\\mu$m)',  scale='log',
                       abs=0.0,  rel=0.10),
}
MODE_NAME = {0: 'fine', 1: 'coarse'}
# Colour for a non-spectral panel (no band to key off).
NEUTRAL_CMAP = 'viridis'


def _truncate(name, lo=CMAP_FLOOR, hi=1.0, n=256):
    """Colormap with its pale low end removed, so sparse bins stay visible."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        '%s_t' % name, base(np.linspace(lo, hi, n)), N=n)


def band_cmap(wvl):
    """Truncated colormap for the band nearest wvl."""
    i = int(np.argmin([abs(w - wvl) for w, _ in BAND_CMAP]))
    return _truncate(BAND_CMAP[i][1])


def load_campaign(taskDir, varNames):
    """Pooled truth/retrieved arrays for every requested variable, in ONE pass.

    The pickles are the expensive part (250 files), so everything is read at once
    rather than re-opening them per variable.

    Returns (data, nTask, lam, keepMask) where data[v] = (truth, retrieved) shaped
    (..., nRetrievals) and keepMask marks pixels whose AOD inversion did not diverge.
    The mask is derived from AOD at 549 nm for EVERY variable, so all figures describe
    the same population -- a pixel whose AOD ran away to 8.8 has meaningless
    microphysics too, and masking each variable on itself would quietly compare
    different pixel sets between panels.
    """
    pkls = sorted(glob.glob(os.path.join(taskDir, '*.pkl')))
    if not pkls:
        raise SystemExit('no task pickles found in %s' % taskDir)
    acc = {v: ([], []) for v in varNames}
    nTask, lam = 0, None
    for p in pkls:
        sim = rs.simulation(picklePath=p)
        if not getattr(sim, 'rsltBck', None):
            continue
        if lam is None:
            lam = np.asarray(sim.rsltFwd[0]['lambda'], float)
        for v in varNames:
            acc[v][0].append(np.array([f[v] for f in sim.rsltFwd], float))
            acc[v][1].append(np.array([b[v] for b in sim.rsltBck], float))
        nTask += 1

    data = {}
    for v in varNames:
        t = np.concatenate(acc[v][0], axis=0)     # (N, ...) -- pixels lead
        r = np.concatenate(acc[v][1], axis=0)
        # move the pixel axis last so panels index the leading (mode/wavelength) axes
        data[v] = (np.moveaxis(t, 0, -1), np.moveaxis(r, 0, -1))
    wi = int(np.argmin(np.abs(lam - 0.549)))
    ta, ra = data['aod']
    keep = np.abs(ra[wi] - ta[wi]) <= DIVERGE
    return data, nTask, lam, keep


def envelope(v, absTol, relTol):
    """AERONET-style half-width at true AOD v."""
    return absTol + relTol * np.asarray(v, float)


def panel(ax, x, y, a, spec, cmap, keep, showLegend=False):
    """One 1:1 density panel. Returns (pcolormesh, stats dict)."""
    nAll = x.size
    if a.include_diverged:
        nDiv = 0
    else:
        nDiv = int((~keep).sum())
        x, y = x[keep], y[keep]

    scale = a.scale or spec['scale']
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if scale == 'log':
        # Log bins need strictly positive data; a retrieval can come back at ~0, so drop
        # those explicitly and report them rather than letting log10 emit -inf and
        # silently empty the histogram.
        pos = (x > 0) & (y > 0)
        nNonPos = int((~pos).sum())
        x, y = x[pos], y[pos]
        if x.size == 0:
            return None, None
        lo = max(min(x.min(), y.min()) * 0.85, 1e-12)
        hi = max(x.max(), y.max()) * 1.15
        edges = np.logspace(np.log10(lo), np.log10(hi), a.bins + 1)
    else:
        nNonPos = 0
        if x.size == 0:
            return None, None
        span = max(x.max(), y.max()) - min(x.min(), y.min())
        lo = min(x.min(), y.min()) - 0.03 * span
        hi = max(x.max(), y.max()) + 0.03 * span
        edges = np.linspace(lo, hi, a.bins + 1)

    H, xe, ye = np.histogram2d(x, y, bins=[edges, edges])
    # The colour norm is LOGARITHMIC by default: bin occupancy spans ~4 decades (a
    # handful of retrievals in the tails against >1000 in the dense core), so a linear
    # norm would render everything but the core as the same near-empty shade.
    #
    # 'percent' rescales counts to a share of this panel's retrievals, which makes the
    # colour comparable between panels and between campaigns -- a raw count depends on
    # how many retrievals survived the divergence cut, so the same colour means
    # different things in two panels.  vmin is the share of ONE retrieval, the smallest
    # non-empty bin possible.
    if a.cbar == 'percent':
        H = 100.0 * H / float(x.size)
        vmin = 100.0 / float(x.size)
    else:
        vmin = 1.0
    H = np.ma.masked_where(H <= 0, H)        # empty bins stay background, not dark
    cnorm = (LogNorm(vmin=vmin, vmax=H.max()) if a.cnorm == 'log'
             else Normalize(vmin=0, vmax=H.max()))
    pcm = ax.pcolormesh(xe, ye, H.T, cmap=cmap, norm=cnorm, shading='flat', zorder=2)

    absTol = spec['abs'] if a.abs_tol is None else a.abs_tol
    relTol = spec['rel'] if a.rel_tol is None else a.rel_tol
    d = y - x
    half = envelope(x, absTol, relTol)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(x != 0, d / x, np.nan)
    st = dict(n=x.size, nAll=nAll, nDiv=nDiv, nNonPos=nNonPos,
              within=float(np.mean(np.abs(d) <= half) * 100.0),
              bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
              relBias=float(np.nanmean(rel) * 100.0))

    # Envelope first, 1:1 on top.  An additive term makes it CURVED in log space, so it
    # needs a resolved line rather than two endpoints.
    ln = (np.logspace(np.log10(max(lo, 1e-12)), np.log10(hi), 400) if scale == 'log'
          else np.linspace(lo, hi, 400))
    halfLn = envelope(ln, absTol, relTol)
    if absTol > 0 and relTol > 0:
        lbl = r'$\pm$(%.3g + %.0f%%$\cdot$x)' % (absTol, relTol * 100)
    elif absTol > 0:
        lbl = r'$\pm$%.3g' % absTol
    else:
        lbl = r'$\pm$%.0f%%' % (relTol * 100)
    ax.fill_between(ln, ln - halfLn, ln + halfLn, color=ENVELOPE_COLOR, alpha=0.16,
                    lw=0, zorder=3, label=lbl)
    for sgn in (-1, 1):
        ax.plot(ln, ln + sgn * halfLn, color=ENVELOPE_COLOR, lw=1.0, alpha=0.7, zorder=4)
    ax.plot([lo, hi], [lo, hi], color='black', lw=1.5, zorder=5, label='1:1')

    ax.set_xscale(scale); ax.set_yscale(scale)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=8, colors=INK2, length=3)
    ax.grid(True, color=GRID, lw=.6, zorder=1)
    ax.set_axisbelow(False)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    if showLegend:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower right')
    return pcm, st


def annotate(ax, st, compact=False):
    txt = ('within envelope: %.1f%%\nbias %+.4g   RMSE %.4g'
           % (st['within'], st['bias'], st['rmse']) if compact else
           'N = %s  (%d tasks)\nwithin envelope: %.1f%%\nbias %+.4g   RMSE %.4g\n'
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


def _finish(fig, ax, pcm, a, label):
    cb = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(_CBAR_LABEL[a.cbar], fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_edgecolor(GRID)
    if a.cnorm == 'log':
        _fmt_cbar(cb, a.cbar)
    ax.set_xlabel('true %s' % label, fontsize=9, color=INK2)
    ax.set_ylabel('retrieved %s' % label, fontsize=9, color=INK2)


def make_figure(var, spec, T, R, keep, lam, nTask, a, tag, outDir):
    """Draw every figure this variable needs. Returns list of paths written."""
    kind, label = spec['kind'], spec['label']
    envParts = []
    absTol = spec['abs'] if a.abs_tol is None else a.abs_tol
    relTol = spec['rel'] if a.rel_tol is None else a.rel_tol
    if absTol:
        envParts.append('%.3g' % absTol)
    if relTol:
        envParts.append('%.0f%%*x' % (relTol * 100))
    envStr = '+/-(' + ' + '.join(envParts) + ')' if envParts else 'none'
    written = []

    def _save(fig, suffix):
        out = os.path.join(outDir, 'one2one_%s_%s%s%s.png'
                           % (tag, var, suffix, _SCALE_TAG(a.scale or spec['scale'],
                                                           a.cnorm)))
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(out)
        return out

    def _spectral_grid(t2, r2, titleExtra, suffix):
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 12.0), facecolor=SURFACE)
        any_ok = False
        for i, ax in enumerate(axes.ravel()):
            pcm, st = panel(ax, t2[i], r2[i], a, spec, band_cmap(lam[i]), keep,
                            showLegend=(i == 0))
            if pcm is None:
                ax.set_visible(False)
                continue
            any_ok = True
            st['nTask'] = nTask
            annotate(ax, st, compact=True)
            ax.set_title(r'%.3f $\mu$m' % lam[i], fontsize=11, color=INK, pad=7)
            _finish(fig, ax, pcm, a, label)
            print('  %-10s %.3f um : within %5.1f%%   bias %+.4g   RMSE %.4g'
                  % (var + titleExtra, lam[i], st['within'], st['bias'], st['rmse']))
        if not any_ok:
            plt.close(fig)
            return None
        fig.suptitle('%s%s -- %s   (envelope %s)' % (label, titleExtra, tag, envStr),
                     fontsize=13, color=INK, y=.995)
        fig.tight_layout(rect=[0, 0, 1, .975])
        return _save(fig, suffix)

    if kind == 'spectral':
        _spectral_grid(T, R, '', '')
    elif kind == 'spectral_mode':
        for m in range(T.shape[0]):
            _spectral_grid(T[m], R[m], '  (%s mode)' % MODE_NAME.get(m, 'mode%d' % (m + 1)),
                           '_%s' % MODE_NAME.get(m, 'mode%d' % (m + 1)))
    elif kind == 'mode':
        nM = T.shape[0]
        fig, axes = plt.subplots(1, nM, figsize=(6.4 * nM, 6.2), facecolor=SURFACE)
        axes = np.atleast_1d(axes)
        ok = False
        for m, ax in enumerate(axes):
            pcm, st = panel(ax, T[m], R[m], a, spec, _truncate(NEUTRAL_CMAP), keep,
                            showLegend=(m == 0))
            if pcm is None:
                ax.set_visible(False)
                continue
            ok = True
            st['nTask'] = nTask
            annotate(ax, st, compact=True)
            ax.set_title('%s mode' % MODE_NAME.get(m, 'mode %d' % (m + 1)),
                         fontsize=11, color=INK, pad=7)
            _finish(fig, ax, pcm, a, label)
            print('  %-10s %-7s: within %5.1f%%   bias %+.4g   RMSE %.4g'
                  % (var, MODE_NAME.get(m, m), st['within'], st['bias'], st['rmse']))
        if ok:
            fig.suptitle('%s -- %s   (envelope %s)' % (label, tag, envStr),
                         fontsize=13, color=INK, y=.99)
            fig.tight_layout(rect=[0, 0, 1, .96])
            _save(fig, '')
        else:
            plt.close(fig)
    elif kind == 'scalar':
        fig, ax = plt.subplots(figsize=(7.4, 6.8), facecolor=SURFACE)
        pcm, st = panel(ax, T, R, a, spec, _truncate(NEUTRAL_CMAP), keep, showLegend=True)
        if pcm is None:
            plt.close(fig)
        else:
            st['nTask'] = nTask
            annotate(ax, st)
            _finish(fig, ax, pcm, a, label)
            ax.set_title('%s -- %s   (envelope %s)' % (label, tag, envStr),
                         fontsize=11.5, color=INK, pad=9)
            fig.tight_layout()
            print('  %-10s          : within %5.1f%%   bias %+.4g   RMSE %.4g'
                  % (var, st['within'], st['bias'], st['rmse']))
            _save(fig, '')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('taskDir', nargs='?', default=os.path.join(_ERRSIM, 'tasks'))
    ap.add_argument('--var', default='aod',
                    help="variable, comma-separated, or 'all' (default aod). Known: "
                         + ', '.join(sorted(VARS)))
    ap.add_argument('--out-dir', default=None,
                    help='default res_analysis/pngs/one2one')
    ap.add_argument('--scale', choices=('log', 'linear'), default=None,
                    help='axis scale; default is per-variable (see VARS)')
    ap.add_argument('--cnorm', choices=('log', 'linear'), default='log',
                    help='colour NORM (default log). Distinct from --scale, which sets '
                         'the AXES. Occupancy spans ~4 decades, so linear collapses '
                         'everything outside the densest bins into one shade.')
    ap.add_argument('--cbar', choices=('percent', 'count'), default='percent',
                    help='colour by share of retrievals (default) or raw bin count')
    ap.add_argument('--rel-tol', type=float, default=None,
                    help='override the per-variable relative envelope term')
    ap.add_argument('--abs-tol', type=float, default=None,
                    help='override the per-variable additive envelope term')
    ap.add_argument('--bins', type=int, default=130)
    ap.add_argument('--include-diverged', action='store_true',
                    help='keep pixels whose AOD inversion diverged (|err| > %.1f)' % DIVERGE)
    a = ap.parse_args()

    varNames = sorted(VARS) if a.var == 'all' else [v.strip() for v in a.var.split(',')]
    unknown = [v for v in varNames if v not in VARS]
    if unknown:
        raise SystemExit('unknown variable(s): %s\nknown: %s'
                         % (', '.join(unknown), ', '.join(sorted(VARS))))
    outDir = a.out_dir or os.path.join(_PNGDIR, 'one2one')
    os.makedirs(outDir, exist_ok=True)

    taskDir = a.taskDir.rstrip('/')
    tag = os.path.basename(taskDir)
    if tag == 'tasks':      # .../tasks_250_smokeOcean/tasks -> name it by the campaign
        parent = os.path.basename(os.path.dirname(taskDir))
        if parent and parent != 'err_sim':
            tag = parent

    # 'aod' is always loaded: the divergence mask is derived from it.
    need = sorted(set(varNames) | {'aod'})
    data, nTask, lam, keep = load_campaign(taskDir, need)
    print('%s: %d tasks, %s retrievals, bands %s um  (%d diverged pixels masked)'
          % (taskDir, nTask, format(keep.size, ','),
             np.array2string(lam, precision=3), int((~keep).sum())))

    allWritten = []
    for v in varNames:
        T, R = data[v]
        allWritten += make_figure(v, VARS[v], T, R, keep, lam, nTask, a, tag, outDir)
    print('\nwrote %d figure(s) -> %s' % (len(allWritten), outDir))
    for w in allWritten:
        print('  %s' % os.path.basename(w))
    return 0


if __name__ == '__main__':
    sys.exit(main())
