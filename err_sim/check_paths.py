#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_paths.py -- resolve every external path err_sim needs, and say what is missing.

Run this FIRST on any new machine (and before submitting a cluster campaign).  It
imports nothing that launches GRASP, so it is instant and safe, and it prints the exact
environment variable to set for anything it could not find.

    python err_sim/check_paths.py

Layouts this understands.  Everything is discovered relative to ERRSIM_BASE, which
defaults to the repo's parent directory:

    cluster -- every tree copied side by side under one base
        /gpfsm/dnb33/nsienkie/retr_sim/
            GSFC-Retrieval-Simulators/      <- this repo
            GSFC-GRASP-Python-Interface/
            grasp/
            nsienkie-cal-uncertainty/
            data_stor/

    laptop -- trees scattered across a work directory; the grandparent is searched too
        ~/working/
            grasp_sims/GSFC-Retrieval-Simulators/   <- this repo
            grasp_sims/GSFC-GRASP-Python-Interface/
            grasp_sims/data_stor/
            grasp/
            uncertainty/nsienkie-cal-uncertainty/

Exit status is 0 when everything needed to run a campaign resolved, 1 otherwise.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

OK, BAD, WARN = '  OK  ', 'MISSING', ' WARN '


def _expand(p):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p))) if p else p


def _row(status, label, value, note=''):
    print('[%s] %-26s %s%s' % (status, label, value, ('   <- ' + note) if note else ''))


def main():
    print('=' * 78)
    print('err_sim path check')
    print('=' * 78)

    failures = []

    # --- what the environment is asking for --------------------------------
    print('\nEnvironment')
    for var in ('ERRSIM_BASE', 'ERRSIM_GRASP_BIN', 'ERRSIM_GRASP_KERNELS',
                'ERRSIM_GEOM_NC4', 'CAL_UNCERTAINTY_DIR', 'ERRSIM_CAL_H5',
                'ERRSIM_COV_CSV', 'ERRSIM_CONCASE', 'ERRSIM_BCK_YAML',
                'ERRSIM_TAU_FACTOR', 'ERRSIM_NPIX', 'ERRSIM_MAXCPU',
                'ERRSIM_TASK_OUT'):
        val = os.environ.get(var)
        if val:
            _row(OK, var, val)
    if not any(os.environ.get(v) for v in ('ERRSIM_BASE', 'CAL_UNCERTAINTY_DIR')):
        print('   (none of the path variables are set -- relying on discovery)')

    # --- the interface repo, needed before anything imports runGRASP -------
    print('\nRepositories')
    iface = os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface')
    base_guess = _expand(os.environ.get('ERRSIM_BASE', os.path.dirname(_REPO)))
    if not os.path.isdir(iface):
        alt = os.path.join(base_guess, 'GSFC-GRASP-Python-Interface')
        iface = alt if os.path.isdir(alt) else iface
    _row(OK if os.path.isdir(iface) else BAD, 'GRASP python interface', iface,
         '' if os.path.isdir(iface) else 'must sit beside this repo')
    if not os.path.isdir(iface):
        failures.append('GSFC-GRASP-Python-Interface')
    _row(OK, 'retrieval simulators', _REPO)

    # --- grasp binary, kernels, geometry -----------------------------------
    # Importing run_experiment resolves all three and raises a message naming every
    # place it looked, which is exactly what we want to show the user.
    print('\nGRASP and geometry')
    rx = None
    try:
        sys.path.insert(0, iface)
        import err_sim.np_compat  # noqa: F401
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'run_experiment_check', os.path.join(_HERE, 'run_experiment.py'))
        rx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rx)
    except FileNotFoundError as e:
        # exec_module leaves a HALF-BUILT module bound to rx, so drop it: the
        # attributes we want to report were never assigned.
        rx = None
        print(str(e))
        failures.append('grasp binary / kernels / geometry')
    except Exception as e:                       # import-time failure, not a path issue
        rx = None
        _row(WARN, 'run_experiment import', '%s: %s' % (type(e).__name__, e))
        failures.append('run_experiment import')

    if rx is not None:
        _row(OK, 'ERRSIM_BASE (effective)', rx.ERRSIM_BASE)
        _row(OK, 'grasp binary', rx.DIR_GRASP,
             '' if os.access(rx.DIR_GRASP, os.X_OK) else 'NOT EXECUTABLE')
        if not os.access(rx.DIR_GRASP, os.X_OK):
            failures.append('grasp binary is not executable')
        nKrnl = len(os.listdir(rx.KRNL_PATH)) if os.path.isdir(rx.KRNL_PATH) else 0
        _row(OK, 'grasp kernels', rx.KRNL_PATH, '%d entries' % nKrnl)
        if rx.GEOM_NC4:
            _row(OK, 'geometry nc4', rx.GEOM_NC4,
                 '%.0f kB' % (os.path.getsize(rx.GEOM_NC4) / 1024))
        print('\nScene configuration')
        _row(OK, 'canonical case', rx.CONCASE)
        _row(OK, 'AOD draw', rx.TAU_FACTOR)
        _row(OK, 'forward YAML', os.path.basename(rx.FWD_YAML))
        _row(OK if os.path.isfile(rx.BCK_YAML) else BAD, 'retrieval YAML',
             os.path.basename(rx.BCK_YAML))
        _row(OK, 'pixels', 'all (526)' if rx.N_PIX is None else str(rx.N_PIX))
        _row(OK, 'max CPU', str(rx.MAX_CPU))

    # --- calibration-sim products ------------------------------------------
    print('\nCalibration simulation inputs')
    try:
        import err_sim.customErrModel as cem
        _row(OK if os.path.isdir(cem._CALSIM_DIR) else BAD,
             'cal-sim checkout', cem._CALSIM_DIR)
        if not os.path.isdir(cem._CALSIM_DIR):
            failures.append('cal-sim checkout')
            print('   candidates tried:')
            for c in cem._calsim_candidates():
                print('     %s' % c)
        for label, path, pattern in (
                ('calibration matrices', cem.CAL_MATRIX_H5_PATH, '*.h5'),
                ('covariance matrix', cem.COV_MATRIX_PATH, '*.csv')):
            p = _expand(path)
            if os.path.isfile(p):
                _row(OK, label, p, '%.0f MB' % (os.path.getsize(p) / 1e6))
            else:
                _row(BAD, label, p)
                failures.append(label)
                d = os.path.dirname(p)
                if os.path.isdir(d):
                    import glob
                    have = sorted(glob.glob(os.path.join(d, pattern)))
                    print('   directory exists; it contains %d %s file(s):' % (len(have), pattern))
                    for h in have[:10]:
                        print('     %s' % os.path.basename(h))
                    if have:
                        print('   -> point %s at one of these'
                              % ('ERRSIM_CAL_H5' if pattern == '*.h5' else 'ERRSIM_COV_CSV'))
                else:
                    print('   directory does not exist: %s' % d)
    except Exception as e:
        _row(BAD, 'customErrModel', '%s: %s' % (type(e).__name__, e))
        failures.append('customErrModel import')

    # --- verdict ------------------------------------------------------------
    print('\n' + '=' * 78)
    if failures:
        print('NOT READY -- %d problem(s): %s' % (len(failures), ', '.join(failures)))
        print('Set the named variable, or copy the missing tree under ERRSIM_BASE.')
        return 1
    print('READY -- every path resolved. Safe to submit.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
