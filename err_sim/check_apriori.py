#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_apriori.py -- does a canonical case's TRUTH fit inside a settings file's box?

Run this BEFORE any campaign with a new scene or a new settings file.  A retrieved
parameter whose truth lies outside its a-priori range is pinned at the bound: it cannot
respond to anything the campaign is trying to measure, and the resulting "bias" is an
artefact of the settings file rather than a property of the retrieval.

That is not hypothetical.  The 2026-09 smoke campaign had 49% of fine-mode k truth
above its cap and 46% of coarse r_v below its floor, which produced enormous
microphysical biases that had nothing to do with calibration.

    python err_sim/check_apriori.py smokeVariable settings_BCK_POLAR_2modes_errsim_smoke.yml
    python err_sim/check_apriori.py dustVariableDesert settings_..._dust_desert.yml

Exit status is 0 when every retrieved parameter is within tolerance, 1 otherwise.

WHAT IS COMPARED.  conCaseDefinitions() is sampled --n times (the 'Variable' suffix
adds per-pixel scatter, so one draw is not enough) and each retrieved characteristic is
matched to the truth field that feeds it:

    characteristic[1] aerosol_concentration        <- vol
    characteristic[2] size_distribution_lognormal  <- lgrnm [rv, sigma]
    characteristic[3] real_part_of_refr_index      <- n
    characteristic[4] imag_part_of_refr_index      <- k
    characteristic[5] surface_water_cox_munk_iso   <- cxMnk   (ocean only)
    characteristic[6] vertical_profile_height      <- vrtHght
    characteristic[7] surface_land_brdf_ross_li    <- brdf    (land only)
    characteristic[8] sphere_fraction              <- sph
    characteristic[9] surface_land_maignan_breon   <- bpdf    (land only)

Surface characteristics are skipped when the case does not use them -- a land case has
no cxMnk and an ocean case has no brdf/bpdf -- since GRASP selects between them on the
SDATA land percentage.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_YML = os.path.join(_REPO, 'ACCP_ArchitectureAndCanonicalCases')
sys.path.insert(0, _REPO)
sys.path.insert(0, _YML)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO), 'GSFC-GRASP-Python-Interface'))
import err_sim.np_compat  # noqa: F401,E402
from canonicalCaseMap import conCaseDefinitions  # noqa: E402
from architectureMap import returnPixel  # noqa: E402

# characteristic index -> (truth key, per-mode value names).  A characteristic whose
# YAML bounds carry several numbers per mode (size distribution, spectral BRDF) has one
# name per slot; a scalar characteristic has one.
CHARS = {
    1: ('vol',     ['conc']),
    2: ('lgrnm',   ['rv', 'sigma']),
    3: ('n',       ['n']),
    4: ('k',       ['k']),
    5: ('cxMnk',   ['cxMnk']),
    6: ('vrtHght', ['height']),
    7: ('brdf',    ['brdf']),
    8: ('sph',     ['sph']),
    9: ('bpdf',    ['bpdf']),
}
SURFACE_OCEAN, SURFACE_LAND = {5}, {7, 9}


def sample_truth(case, nDraw, arch='harperrsimmc'):
    """nDraw independent truth states for a canonical case."""
    pix = returnPixel(arch, sza=30., relPhi=0., vza=None)
    out, landPrct = {}, None
    for _ in range(nDraw):
        vals, landPrct = conCaseDefinitions(case, pix)[:2]
        for k, v in vals.items():
            out.setdefault(k, []).append(np.atleast_2d(np.asarray(v, float)))
    return {k: np.stack(v) for k, v in out.items()}, landPrct


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', help="canonical case, e.g. 'dustVariableDesert'")
    ap.add_argument('yaml', help='settings file (bare name resolves against the YAML dir)')
    ap.add_argument('--n', type=int, default=3000, help='truth draws (default 3000)')
    ap.add_argument('--tol', type=float, default=1.0,
                    help='percent of truth allowed outside a bound (default 1.0)')
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()

    import yaml as _yaml
    path = a.yaml if os.sep in a.yaml else os.path.join(_YML, a.yaml)
    if not os.path.isfile(path):
        raise SystemExit('settings file not found: %s' % path)
    cons = _yaml.safe_load(open(path))['retrieval']['constraints']

    np.random.seed(a.seed)
    truth, landPrct = sample_truth(a.case, a.n)

    print('=' * 92)
    print('case %s   vs   %s' % (a.case, os.path.basename(path)))
    print('%d draws, land_prct = %d (%s)' % (a.n, landPrct,
                                             'LAND' if landPrct > 0 else 'OCEAN'))
    print('=' * 92)
    print('%-26s %5s %12s %12s %12s %9s' % ('parameter', 'mode', 'truth med', 'min', 'max', '%outside'))
    print('-' * 92)

    failures = []
    for ci in sorted(CHARS):
        key = 'characteristic[%d]' % ci
        if key not in cons or not cons[key].get('retrieved'):
            continue
        # skip the surface family the case does not use
        if ci in SURFACE_OCEAN and landPrct >= 100:
            continue
        if ci in SURFACE_LAND and landPrct <= 0:
            continue
        tKey, names = CHARS[ci]
        if tKey not in truth:
            print('%-26s %5s %12s %12s %12s %9s' % (tKey, '-', 'NOT SET by the case',
                                                    '', '', 'skipped'))
            continue
        T = truth[tKey]                       # (nDraw, nRow, nCol)
        for mKey, mVal in sorted(cons[key].items()):
            if not mKey.startswith('mode['):
                continue
            m = int(mKey[5:-1]) - 1
            if m >= T.shape[1]:
                continue
            ig = mVal['initial_guess']
            mn = np.atleast_1d(np.asarray(ig['min'], float))
            mx = np.atleast_1d(np.asarray(ig['max'], float))
            row = T[:, m, :]                  # (nDraw, nCol)
            for j, nm in enumerate(names):
                # size distribution stores [rv, sigma] along the column axis; every
                # other characteristic is spectral, so compare against band 0 and, for
                # spectrally-bounded characteristics, the matching bound slot.
                col = row[:, j] if len(names) > 1 else row[:, 0]
                lo = mn[j] if len(names) > 1 else mn[0]
                hi = mx[j] if len(names) > 1 else mx[0]
                out = 100.0 * np.mean((col < lo) | (col > hi))
                flag = ''
                if out > a.tol:
                    flag = '  <-- FAILS'
                    failures.append('%s mode%d (%.0f%% outside [%.4g, %.4g], truth median %.4g)'
                                    % (nm, m + 1, out, lo, hi, np.median(col)))
                print('%-26s %5d %12.5g %12.5g %12.5g %8.1f%%%s'
                      % (nm, m + 1, np.median(col), lo, hi, out, flag))
            # spectral characteristics whose bounds vary by band: check every band
            if len(names) == 1 and row.shape[1] > 1 and mn.size > 1:
                nB = min(row.shape[1], mn.size)
                for b in range(nB):
                    out = 100.0 * np.mean((row[:, b] < mn[b]) | (row[:, b] > mx[b]))
                    if out > a.tol:
                        failures.append('%s mode%d band%d (%.0f%% outside [%.4g, %.4g], '
                                        'truth median %.4g)'
                                        % (names[0], m + 1, b, out, mn[b], mx[b],
                                           np.median(row[:, b])))
                        print('%-26s %5d %12.5g %12.5g %12.5g %8.1f%%  <-- FAILS (band %d)'
                              % (names[0] + ' band%d' % b, m + 1, np.median(row[:, b]),
                                 mn[b], mx[b], out, b))

    print('=' * 92)
    if failures:
        print('NOT USABLE -- %d parameter(s) have truth outside the a-priori box:' % len(failures))
        for f in failures:
            print('   %s' % f)
        print('Widen those bounds, or the campaign measures the settings file rather than')
        print('the retrieval.')
        return 1
    print('OK -- every retrieved parameter has <= %.1f%% of its truth outside the box.' % a.tol)
    return 0


if __name__ == '__main__':
    sys.exit(main())
