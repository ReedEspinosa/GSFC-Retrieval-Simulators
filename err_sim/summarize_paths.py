#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize_paths.py -- one readable report comparing the err_sim error-model paths.

Reads the per-path pickles written by run_experiment.py and writes a plain-text
summary (also printed to stdout) with, per wavelength and per path: RMSE and bias
for AOD, SSA, the fine-mode refractive index, and rEff.

All paths are scored on the SAME scenes -- the intersection matched on truth AOD --
so a path that loses retrievals to inversion failures cannot flatter itself by being
graded on an easier subset.

Usage:
    python err_sim/summarize_paths.py [outputFile]     # default err_sim/path_summary.txt
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402
import simulateRetrieval as rs  # noqa: E402

PATHS = [('Path 1 analytic', 'harperrsim'),
         ('Path 2 MonteCarlo', 'harperrsimmc'),
         ('Path 3 control', 'harperrsimbck')]

METRICS = [
    ('AOD',      lambda r, w: r['aod'][w]),
    ('SSA',      lambda r, w: r['ssa'][w]),
    ('n_fine',   lambda r, w: r['n'][0][w]),
    ('k_fine',   lambda r, w: r['k'][0][w]),
]


def _stat(fw, bk, fn, w):
    x = np.array([fn(f, w) for f in fw], dtype=float)
    y = np.array([fn(b, w) for b in bk], dtype=float)
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    if x.size == 0:
        return np.nan, np.nan
    return float(np.sqrt(np.mean((y - x) ** 2))), float(np.mean(y - x))


def main():
    outPath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, 'path_summary.txt')
    L = []
    def emit(line=''):
        print(line)
        L.append(line)

    loaded = []
    for label, inst in PATHS:
        pkl = os.path.join(_HERE, 'experiment_%s.pkl' % inst)
        if os.path.isfile(pkl):
            loaded.append((label, inst, rs.simulation(picklePath=pkl), os.path.getmtime(pkl)))
        else:
            emit('MISSING: %s -- %s not found (that path did not produce a result)'
                 % (label, os.path.basename(pkl)))
    if not loaded:
        emit('No experiment pickles found -- nothing to summarise.')
        open(outPath, 'w').write('\n'.join(L) + '\n')
        return 1

    key = lambda sim: [round(float(f['aod'][1]), 9) for f in sim.rsltFwd]
    keys = [key(s) for _, _, s, _ in loaded]
    common = sorted(set(keys[0]).intersection(*[set(k) for k in keys[1:]]))
    idxs = [[k.index(c) for c in common] for k in keys]

    wv = loaded[0][2].rsltFwd[0]['lambda']
    emit('=' * 78)
    emit('err_sim path comparison')
    emit('=' * 78)
    # Print each pickle's age: a result left over from an earlier run is then obvious
    # rather than silently reported as if it belonged to this campaign.
    now = time.time()
    for label, inst, sim, mtime in loaded:
        age = (now - mtime) / 60.0
        flag = '   <-- STALE?' if age > 180 else ''
        emit('  %-20s %-16s retrieved %4d of %4d pixels   written %s (%.0f min ago)%s'
             % (label, inst, len(sim.rsltBck), len(sim.rsltFwd),
                time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)), age, flag))
    # Differing pixel counts have two very different causes, so distinguish them:
    #   benign -- simulateRetrieval.runSim PRUNES rsltFwd to the pixels whose inversion
    #             succeeded, so a path that lost retrievals has fewer entries but its
    #             scenes are a SUBSET of the fuller path's.
    #   serious -- pickles actually came from different campaigns, so the scene sets
    #             diverge and the comparison is meaningless.
    counts = {len(sim.rsltFwd) for _, _, sim, _ in loaded}
    if len(counts) > 1:
        sets = [set(k) for k in keys]
        biggest = max(sets, key=len)
        nested = all(st <= biggest for st in sets)
        if nested:
            emit('  NOTE: pixel counts differ %s. Each smaller set is a subset of the'
                 % sorted(counts))
            emit('        largest, i.e. those paths simply lost retrievals to inversion')
            emit('        failures. Scoring below is on the common subset, so this is fair.')
        else:
            emit('  WARNING: pixel counts differ %s AND the scene sets diverge -- these'
                 % sorted(counts))
            emit('           pickles are from DIFFERENT runs. Treat this comparison as invalid.')
    emit('  common scenes scored: %d' % len(common))
    if common:
        emit('  truth AOD(549nm): min %.3f  median %.3f  max %.3f'
             % (min(common), float(np.median(common)), max(common)))
    sza = [float(f['sza'][0, 1]) for f in loaded[0][2].rsltFwd]
    emit('  SZA range: %.1f .. %.1f deg' % (min(sza), max(sza)))
    emit('  wavelengths: %s um' % np.array2string(wv, precision=3))
    emit()

    for w, wl in enumerate(wv):
        emit('--- %.3f um ' % wl + '-' * 58)
        hdr = '%-20s' % '' + ''.join('%13s%13s' % ('RMSE_' + m, 'BIAS_' + m) for m, _ in METRICS)
        emit(hdr)
        for (label, _, sim, _mt), idx in zip(loaded, idxs):
            fw = [sim.rsltFwd[i] for i in idx]
            bk = [sim.rsltBck[i] for i in idx]
            row = '%-20s' % label
            for _, fn in METRICS:
                r, b = _stat(fw, bk, fn, w)
                row += '%13.4f%+13.4f' % (r, b)
            emit(row)
        emit()

    emit('--- size parameters (wavelength independent) ' + '-' * 32)
    emit('%-20s%13s%13s%13s%13s' % ('', 'RMSE_rEff', 'BIAS_rEff', 'RMSE_rvF', 'BIAS_rvF'))
    for (label, _, sim, _mt), idx in zip(loaded, idxs):
        fw = [sim.rsltFwd[i] for i in idx]
        bk = [sim.rsltBck[i] for i in idx]
        r1, b1 = _stat(fw, bk, lambda r, w: float(r['rEff']), 0)
        r2, b2 = _stat(fw, bk, lambda r, w: r['rv'][0], 0)
        emit('%-20s%13.4f%+13.4f%13.4f%+13.4f' % (label, r1, b1, r2, b2))
    emit()
    emit('NOTE: bias is mean(retrieved - truth). All paths scored on identical scenes.')

    with open(outPath, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\nSaved summary -> %s' % outPath)
    return 0


if __name__ == '__main__':
    sys.exit(main())
