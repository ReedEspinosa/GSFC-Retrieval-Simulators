#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_tasks.py -- gather per-task retrieval results into one tidy table.

Reads the pickles and provenance JSONs written by ``run_task.py`` and emits ONE ROW
PER TASK: the instrument and calibration indices it used, followed by RMSE and bias
for each retrieved variable at each wavelength.  That is the table you regress
against to find trends -- e.g. does a more poorly calibrated instrument retrieve a
more biased AOD.

Rows carry the raw indices rather than derived calibration metrics, so any statistic
(||C - inv(A)||, per-element deviations, implied sigma_DoLP, ...) can be worked back
later from the calibration HDF5 using instrument_idx / cal_idx.

Usage:
    python err_sim/collect_tasks.py [taskDir] [outCsv]

Defaults: err_sim/tasks -> err_sim/tasks/task_summary.csv
"""

import csv
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402
import simulateRetrieval as rs  # noqa: E402

# (name, extractor(rslt, waveIndex)) -- wavelength-resolved
WAVE_VARS = [
    ('aod',    lambda r, w: r['aod'][w]),
    ('ssa',    lambda r, w: r['ssa'][w]),
    ('n_fine', lambda r, w: r['n'][0][w]),
    ('k_fine', lambda r, w: r['k'][0][w]),
]
# (name, extractor(rslt)) -- scalar per pixel
SCALAR_VARS = [
    ('rEff',   lambda r: float(r['rEff'])),
    ('rv_fine', lambda r: r['rv'][0]),
]


def _stat(fw, bk, fn):
    x = np.array([fn(f) for f in fw], dtype=float)
    y = np.array([fn(b) for b in bk], dtype=float)
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    if x.size == 0:
        return np.nan, np.nan
    return float(np.sqrt(np.mean((y - x) ** 2))), float(np.mean(y - x))


def main():
    taskDir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, 'tasks')
    outCsv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(taskDir, 'task_summary.csv')

    metas = sorted(glob.glob(os.path.join(taskDir, 'task_*.json')))
    if not metas:
        raise SystemExit('No task_*.json found in %s' % taskDir)

    rows, wvls, skipped = [], None, []
    for mpath in metas:
        with open(mpath) as f:
            prov = json.load(f)
        pkl = mpath[:-5] + '.pkl'
        if not os.path.isfile(pkl):
            skipped.append((prov.get('task_idx'), 'no pickle -- task likely failed'))
            continue
        try:
            sim = rs.simulation(picklePath=pkl)
        except Exception as e:                                   # noqa: BLE001
            skipped.append((prov.get('task_idx'), 'unreadable: %s' % e))
            continue
        if not sim.rsltBck:
            skipped.append((prov.get('task_idx'), 'no retrievals'))
            continue

        fw, bk = sim.rsltFwd, sim.rsltBck
        n = min(len(fw), len(bk))
        fw, bk = list(fw)[:n], list(bk)[:n]
        wv = fw[0]['lambda']
        wvls = wv if wvls is None else wvls

        row = dict(task_idx=prov['task_idx'],
                   instrument=prov.get('instrument', ''),
                   cal_idx=prov['cal_idx'],
                   noise_seed=prov.get('noise_seed', ''),
                   n_pix=n)
        for i, idx in enumerate(prov['instrument_idx']):
            row['instr_idx_wl%d' % i] = idx
        for w, wl in enumerate(wv):
            nm = '%03dnm' % int(round(wl * 1000))
            for var, fn in WAVE_VARS:
                r, b = _stat(fw, bk, lambda x, fn=fn, w=w: fn(x, w))
                row['rmse_%s_%s' % (var, nm)] = r
                row['bias_%s_%s' % (var, nm)] = b
        for var, fn in SCALAR_VARS:
            r, b = _stat(fw, bk, fn)
            row['rmse_%s' % var] = r
            row['bias_%s' % var] = b
        rows.append(row)

    if not rows:
        raise SystemExit('No readable task results in %s' % taskDir)

    cols = list(rows[0].keys())
    with open(outCsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print('collected %d task(s) -> %s' % (len(rows), outCsv))
    if skipped:
        print('SKIPPED %d task(s):' % len(skipped))
        for t, why in skipped:
            print('   task %s: %s' % (t, why))
    print()
    print('%-8s %-22s %-8s %-7s %10s %10s'
          % ('task', 'instruments', 'cal', 'n_pix', 'rmse_aod*', 'bias_aod*'))
    key = [c for c in cols if c.startswith('rmse_aod_')]
    kb = [c for c in cols if c.startswith('bias_aod_')]
    for r in rows:
        ii = [r[c] for c in cols if c.startswith('instr_idx_wl')]
        print('%-8s %-22s %-8s %-7s %10.4f %+10.4f'
              % (r['task_idx'], ','.join(str(x) for x in ii), r['cal_idx'], r['n_pix'],
                 np.nanmean([r[c] for c in key]), np.nanmean([r[c] for c in kb])))
    print('(* mean over wavelengths; per-wavelength columns are in the CSV)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
