#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_task.py -- one err_sim retrieval task: ONE instrument, ONE calibration.

Designed to be the body of a SLURM array job:

    python err_sim/run_task.py $SLURM_ARRAY_TASK_ID

Each task pins itself to a distinct set of pool entries (one instrument per
wavelength channel) and to a single calibration event, runs the full geometry, and
writes its own pickle plus a small JSON of provenance.  Collect them afterwards with
``collect_tasks.py``.

Why one calibration per task
----------------------------
Unpinned, Path 2 draws a fresh calibration matrix on every call, so calibration error
acts like extra RANDOM noise that averages away across a scene.  A real instrument has
exactly one calibration, so its error is a fixed SYSTEMATIC that biases every pixel the
same way.  Pinning makes that bias visible within a task; the spread ACROSS tasks is
then the instrument-to-instrument distribution.

Experimental design
-------------------
* **Paired scenes.**  Geometry and the AOD draw are seeded identically in EVERY task
  (``TAU_SEED``/``GEOM_*`` from run_experiment), so across-task differences are
  attributable to calibration alone rather than to scene variability.
* **Reproducible.**  Instrument indices, calibration index and the measurement-noise
  stream all derive from the task index, so any task replays exactly.
* **Channels stay distinct.**  A task consumes ``n_wvl`` consecutive pool entries, one
  per band, preserving the per-wavelength-instrument model.  1000 pool entries
  therefore supply 250 tasks at 4 bands.

Usage
-----
    python err_sim/run_task.py <task_index> [n_pix]

``n_pix`` defaults to every valid geometry pixel; pass a small number for a smoke test.
Environment overrides: ``ERRSIM_INSTRUMENT`` (default harperrsimmc -- the Monte Carlo
path is the one that carries calibration bias), ``ERRSIM_TASK_OUT``.
"""

import json
import os
import random
import sys
import tempfile

import numpy as np

# --- per-task temp isolation (MUST happen before tempfile is first used) ------
# graspYAML(newTmpFile=...) builds its "unique" filename from np.random.randint on
# the GLOBAL numpy stream -- its own comment says this is "needed to prevent
# identical FN w/ many parallel runs".  But paired scenes require seeding that same
# stream identically in every task, which makes every task generate the SAME temp
# YAML names in the SHARED temp directory.  Content is identical too (that is the
# point of pairing), but copyfile/writeYAML are not atomic, so one task can read a
# file another is midway through truncating.
#
# Rather than perturb the scene RNG (which would break pairing), give each task its
# own TMPDIR: identical names then land in different directories and cannot collide.
# GRASP's mkdtemp working dirs follow TMPDIR too, which additionally keeps that I/O
# on node-local scratch instead of a shared filesystem.
_TASK_TMP = os.environ.get('ERRSIM_TASK_TMPDIR')
if _TASK_TMP is None and len(sys.argv) > 1:
    _base = os.environ.get('SLURM_TMPDIR') or tempfile.gettempdir()
    _TASK_TMP = os.path.join(_base, 'errsim_task_%s' % sys.argv[1])
if _TASK_TMP:
    os.makedirs(_TASK_TMP, exist_ok=True)
    os.environ['TMPDIR'] = _TASK_TMP
    tempfile.tempdir = _TASK_TMP        # reset tempfile's cached value

# HDF5 keeps a lock file alongside anything it opens, which fails on some parallel
# filesystems when many readers hit the same file.  Every access here is read-only.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'ACCP_ArchitectureAndCanonicalCases'))
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402  -- must precede runGRASP
import err_sim.customErrModel as cem  # noqa: E402

# ================================ CONFIG ================================
INSTRUMENT = os.environ.get('ERRSIM_INSTRUMENT', 'harperrsimmc')
OUT_DIR = os.environ.get('ERRSIM_TASK_OUT', os.path.join(_HERE, 'tasks'))
NOISE_SEED_BASE = 700000       # measurement-noise stream = NOISE_SEED_BASE + task_idx
CAL_SEED_BASE = 90000          # calibration-event draw    = CAL_SEED_BASE + task_idx
# =======================================================================


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    taskIdx = int(sys.argv[1])
    nPixArg = sys.argv[2] if len(sys.argv) > 2 else None

    # run_experiment holds the scene/geometry/GRASP configuration; import it with the
    # pixel count already decided so its module-level CONFIG resolves correctly.
    if nPixArg is not None:
        os.environ['ERRSIM_NPIX'] = str(nPixArg)
    os.environ['ERRSIM_INSTRUMENT'] = INSTRUMENT
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'run_experiment_task', os.path.join(_HERE, 'run_experiment.py'))
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)

    os.makedirs(OUT_DIR, exist_ok=True)
    pkl = os.path.join(OUT_DIR, 'task_%05d_%s.pkl' % (taskIdx, INSTRUMENT))
    meta = os.path.join(OUT_DIR, 'task_%05d_%s.json' % (taskIdx, INSTRUMENT))

    # --- pin this task to its instrument + calibration -------------------
    cem.init_store()
    nWvl = len(cem.VIEW_BAND_WVLS)
    prov = cem.pin_task(taskIdx, n_wvl=nWvl, cal_seed_base=CAL_SEED_BASE)
    prov.update(instrument=INSTRUMENT,
                n_wvl=nWvl,
                cal_h5=os.path.basename(cem.CAL_MATRIX_H5_PATH),
                cov_csv=os.path.basename(cem.COV_MATRIX_PATH),
                noise_seed=NOISE_SEED_BASE + taskIdx)
    prov['tmpdir'] = tempfile.gettempdir()
    print('task %d: instruments %s, calibration %d'
          % (taskIdx, prov['instrument_idx'], prov['cal_idx']))
    print('         tmpdir %s' % prov['tmpdir'])

    # --- build the (identical in every task) scenes, then run ------------
    geoms = rx.make_geoms()
    nowPix = [rx.returnPixel(INSTRUMENT, sza=sza, relPhi=phi, vza=vza, concase=rx.CONCASE)
              for sza, phi, vza in geoms]
    np.random.seed(rx.TAU_SEED)          # paired scenes: same AOD draw in every task
    fwdYAML = [rx.setupConCaseYAML(rx.CONCASE, npix, rx.FWD_YAML,
                                   caseLoadFctr=rx.TAU_FACTOR) for npix in nowPix]

    # Seed BOTH generators.  numpy covers the measurement noise and most of
    # scrambleInitialGuess, but miscFunctions.loguniform -- which sets the initial
    # guess for aerosol concentration and imaginary refractive index -- uses the
    # STDLIB random module (`from random import random`), which np.random.seed does
    # NOT touch.  Seeding only numpy leaves the initial guess unseeded, and two runs
    # of the same task then retrieve different answers from identical measurements.
    np.random.seed(prov['noise_seed'])
    random.seed(prov['noise_seed'])
    noiseFun = rx.errorModelOf(nowPix[0])
    simA = rx.rs.simulation(nowPix)
    simA.runSim(fwdYAML, rx.BCK_YAML, rx.NSIMS, maxCPU=rx.MAX_CPU, maxT=rx.MAX_T,
                savePath=pkl, binPathGRASP=rx.DIR_GRASP, intrnlFileGRASP=rx.KRNL_PATH,
                releaseYAML=True, lightSave=False, rndIntialGuess=rx.RND_INITIAL_GUESS,
                dryRun=False, workingFileSave=False, fixRndmSeed=False,
                verbose=rx.VERBOSE, radianceNoiseFun=noiseFun)

    prov['n_pix_requested'] = len(nowPix)
    with open(meta, 'w') as f:
        json.dump(prov, f, indent=2)
    print('task %d done -> %s' % (taskIdx, os.path.basename(pkl)))
    print('             -> %s' % os.path.basename(meta))
    return 0


if __name__ == '__main__':
    sys.exit(main())
