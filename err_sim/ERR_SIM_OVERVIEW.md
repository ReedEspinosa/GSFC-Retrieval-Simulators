# err_sim — Calibration-Uncertainty Error Model for Retrieval Simulations

> Onboarding doc for a future AI instance (or human). Read this together with
> `../ALTERING_THE_MEASUREMENT_ERROR_MODEL.md` (how the retrieval simulator's
> error model works in general and how to add/rerun one). This file covers the
> `err_sim/` experiment specifically: what it does, how the code is wired, and
> what is left to do.

## 1. Goal

Investigate how the **current GRASP retrieval settings** respond to realistic
**polarimeter calibration uncertainty**. Two complementary error injections:

- **Path 1 — analytical, purely random.** Realistic, intensity-dependent
  measurement noise **plus** calibration-matrix uncertainty, propagated
  analytically into a per-measurement Stokes 1-sigma, then applied as a Gaussian
  realization. Unbiased.
- **Path 2 — Monte Carlo, random + biased.** Inverts the truth Stokes to sensor
  intensities with the *true* characteristic matrix, adds detector noise, then
  recombines with a *randomly drawn (imperfect) calibration matrix*. This carries
  random measurement noise, random calibration scatter, **and** the systematic
  bias of an inaccurate calibration.

- **Path 3 — control.** Injects exactly the noise the inversion's BCK YAML *assumes*,
  making the experiment self-consistent (injected error == assumed error). It touches
  no calibration-sim product at all, so it isolates how much of the Path 1/2 error is
  the retrieval setup rather than calibration uncertainty.

Paths 1 and 2 draw their instrument data from the **nsienkie-cal-uncertainty** Monte
Carlo calibration model (a 3-sensor division-of-amplitude polarimeter).

## 2. The three repositories (context)

- `grasp/` — the radiative-transfer + inversion binary (SDATA in, text out).
- `GSFC-GRASP-Python-Interface/` — `runGRASP.py`: `pixel`, `graspRun`, `graspDB`,
  `graspYAML`. Builds pixels, writes SDATA, runs GRASP, parses output.
- `GSFC-Retrieval-Simulators/` — **this repo**. `simulateRetrieval.py::simulation`
  orchestrates forward→noise→inversion. Instruments/scenes/error models live in
  `ACCP_ArchitectureAndCanonicalCases/` (`architectureMap.py`, `canonicalCaseMap.py`).
- `nsienkie-cal-uncertainty/` — the calibration MC. Produces the HDF5 of
  characteristic + calibration matrices and the covariance CSVs consumed here.

## 3. How err_sim plugs into the retrieval simulator

Call chain (unchanged from any other instrument error model):

```
runRetrievalSimulation*.py
  -> returnPixel(archName)                    # architectureMap.py
       -> nowPix.addMeas(..., errModel=functools.partial(addError, errStr))
  -> simulation.runSim(...)                   # simulateRetrieval.py
       -> pixel.populateFromRslt(rsltFwd)     # runGRASP.py, MAIN process, serial
            -> errorModel(l, rsltFwd)         # per (pixel, wavelength)
                 = addError(errStr, l, rsltFwd)         # architectureMap.py
                      -> customErrModel(errStr, l, rsltFwd)   # err_sim/customErrModel.py
```

Two architectures were added to `returnPixel()` (HARP2-like: wavelengths
`[0.441, 0.549, 0.669, 0.873]`, 10 view angles, msTyp `[41,42,43]`=I,Q,U):

| archName | errStr | path |
|---|---|---|
| `harperrsim`    | `errsim01` | Path 1 (analytic) |
| `harperrsimmc`  | `errsim02` | Path 2 (Monte Carlo) |
| `harperrsimbck` | `errsim03` | Path 3 (control: the BCK YAML's own assumed noise) |

`addError()` has an `errsim` branch that dispatches to `customErrModel`. (Single
arch block, suffix-selected, so `'harperrsim' in 'harperrsimmc'` doesn't double-fire.)
`architectureMap.py` also appends the repo dir to `sys.path` so `import err_sim` works.

Return contract (same as all addError models): `np.r_[I, Q, U]`, length `3*Nang`,
ascending measurement-type order.

> **The `radianceNoiseFun` gate (important).** `pixel.populateFromRslt()` only calls an
> error model when the `radianceNoiseFun` **argument** is supplied; it never consults the
> `measVals[n]['errorModel']` that `returnPixel` bound. Calling `runSim()` without it
> silently inverts the **noise-free** forward truth. `run_experiment.py` therefore recovers
> the bound partial with `errorModelOf(pix)` and passes it as `radianceNoiseFun` — see the
> docstring there. `mc_test.py` is unaffected (it calls `errorModel` directly).

## 4. The three error paths (math)

Notation: `A` = (3,3) characteristic matrix (Stokes→sensor, V column dropped);
`C` = (3,3) calibration matrix (sensor→Stokes, ≈ `inv(A)`); `Σ` = (9,9) covariance
of `vec(C)` indexed `k = 3i+j`; `sensorInt = A @ [I,Q,U]`.

### Path 1 — `_errsim_analytic` (errsim01)
Per view angle, build the total Stokes covariance from two terms and draw a
Gaussian realization:

```
Cov(S) =  C · diag(σ_I²) · Cᵀ                     # measurement noise, C = inv(A)
        + Σ_ab  sensorInt_a · sensorInt_b · Σ[3i+a, 3j+b]   # calibration-matrix uncertainty
σ_Stokes = sqrt(diag(Cov(S)))                     # per-component 1-sigma
noised[:,n] = truth[:,n] + N(0, σ_Stokes)         # diagonal (uncorrelated) realization
```
`σ_I` is the per-sensor detector noise (`instr_err`, below) at each channel's
intensity — the measurement covariance is **diagonal** (independent sensors).
DoLP 1-sigma is computed by the delta method on the total `Cov(S)`. Per-angle
`σ_Stokes` and `σ_DoLP` are stashed on module global `LAST_ANALYTIC_SIGMAS`
(they are **not** yet fed to the inversion — see open items).

### Radiometric calibration (applies to BOTH paths)
`USE_RAD_CAL = True` folds the fitted radiometric gain into `C` at load time
(`_apply_radiometric_gain`, ported from the cal sim's `covariance_matrix.py`):
`scalar -> C/gain`, `per_sensor -> C @ diag(1/gains)`. `char_mats` (physical truth) is
never scaled. This is required for consistency: `COV_MATRIX_PATH` points at the **radcal**
covariance, so the Path 2 matrices must be radcal too. Unfolded, they carry ~16x more
scatter in sigma (mean diag Var(C) 7.7e-4 vs 3.0e-6) and the two paths would model
different instruments. Note a *scalar* gain cancels out of DoLP entirely (a ratio is
invariant under a common scale), so radcal folding changes absolute I/Q/U only.

### Path 2 — `_errsim_montecarlo` (errsim02)
```
sensorInt   = A_true @ [I,Q,U]                    # (3, Nang)
noisySensor = sensorInt + N(0, σ_sensor)          # σ_sensor = instr_err(sensorInt)
out         = C @ noisySensor                      # C = one random draw from this instrument
```
`C ≠ inv(A_true)` → the recombination imparts the **systematic calibration bias**
plus random scatter. Vectorized over angles.

### Detector noise — `instr_err` (kept in sync with the cal sim)

Mirrors `nsienkie-cal-uncertainty/lib/cal_uncertainty/calibration.py::instr_err`
(copied, not imported, to avoid cross-repo coupling — **change one, change the
other**). Poisson statistics live in PHOTOELECTRONS, so the electron count is formed
explicitly:

```
C     = norm_intens * counts_max                 counts_max = 0.75*2^14 = 12288 ADU
N_e   = C * g                                    g = SENSOR_GAIN_E_PER_ADU = 2
var_e = (N_e + read_noise_e**2) / bin_size       digital binning; read_noise_e = 5 e-
sigma = sqrt(var_e) / g / counts_max             back to normalized units
```

Three corrections relative to the original model, which together cut sigma 35-61%:

* shot and read noise are combined in **quadrature**, not summed linearly;
* the gain `g` is explicit — the old `sqrt(C/bin)` form silently assumed 1 e-/ADU;
* the read floor is in **electrons** (detector specs are quoted that way), so
  changing `g` no longer silently changes the physical read noise.

SNR = `sqrt(N_e)` is invariant under `g`, as it must be for a pure amplifier; `g`
matters only because the input is a fraction of saturation rather than an electron
count, so it sets the full well (`counts_max*g` = 24576 e-) and the SNR ceiling.
`binning='onchip'` (charge summed, read once, CCD-style, `sigma_R/B`) is available
but unused; HARP2 mixes digital and on-chip and the difference is ~2% here.

### View-SECTOR binning (Paths 1 & 2)

A polarimeter pixel holds one measurement per **view sector** — a fixed viewing
direction built into the instrument, at +/-57 deg for a HARP-like design. The
detector bins many pixels near nadir and fewer at the extremes, so the bin factor is
a **fixed instrument property indexed by sector**, identical for every ground pixel.
It is deliberately NOT driven by the per-pixel orbital `vza`, which would make the
detector layout drift with solar geometry.

| sector (deg) | -57 | -44 | -32 | -19 | -6 | +6 | +19 | +32 | +44 | +57 |
|---|---|---|---|---|---|---|---|---|---|---|
| bins | 1 | 3.2 | 9.6 | 30.9 | 100 | 100 | 30.9 | 9.6 | 3.2 | 1 |

Geometric interpolation between `BIN_NADIR` and `BIN_EDGE` so SNR (~`sqrt(bin)`)
varies smoothly. `BIN_VIEW_DEPENDENT=False` falls back to a flat `SENSOR_BIN_FACTOR`.

### Radiometry: reflectance -> radiance -> saturation fraction (Paths 1 & 2)

GRASP works in **reflectance**, but shot noise depends on photons, so the error model
undoes that before applying the detector model:

```
L    = R * F0(lambda) * mu0 / pi          reflectance -> radiance   (F0 in ENERGY units)
norm = L / L_sat(lambda)                  radiance -> fraction of saturation
```

`counts_scale()` returns `norm/R`; `instr_err` then does radiance -> counts ->
electrons as usual. Two wavelength-dependent pieces:

* **`mu0 = cos(SZA)`.** A low sun genuinely delivers fewer photons for the same
  reflectance. Omitting it made shot noise sun-angle independent — wrong by ~1.7x in
  sigma over a 15-70 deg SZA range. Disable with `GRASP_I_IS_REFLECTANCE=False` if
  `fit_I` turns out to already carry mu0. **This convention was not independently
  verified** (a two-SZA test was confounded by geometry); a Lambertian surface with
  aerosol and molecular scattering off would settle it.
* **`L_sat(lambda)`.** At HARP2's shared integration time, saturation tracks the
  inverse detector responsivity `rho = QE(lambda)*lambda`, so
  `L_sat(l) = SAT_RADIANCE * rho(ref)/rho(l)`. The `lambda` is the photon-energy
  conversion (a fixed radiance yields more photons at longer wavelengths) and is a
  SEPARATE step from `F0`, not an alternative to it.

The anchor is the **red band** (0.669 um), the band intended for cloud microphysics.
`SAT_RADIANCE=None` derives it as `F0(0.669)/pi = 483.3 W m-2 sr-1 um-1`, i.e. it is
DEFINED so a perfectly reflecting Lambertian scene (R=1) with the sun overhead exactly
saturates the reference band — a definition, not an instrument measurement (real
bright clouds are nearer R~0.8-0.9). Deriving rather than hardcoding keeps the anchor
consistent if `SOLAR_T`/`SAT_REF_WVL` change, and makes `counts_scale` in the red band
reduce exactly to `mu0`.

With a shared integration time the green band lands slightly over its nominal ceiling
on the brightest clouds. That is harmless: `counts_max` is 75% of full scale, a SOFT
ceiling, and nothing in the noise model clips — in reality green would see mild
non-linearity, which is not modelled.

`QE_BY_BAND=None` gives flat QE. QE is the dominant band-to-band term: the `lambda`
factor nearly cancels the F0 roll-off on its own (873 band at 0.94x the 549 band), but
a representative silicon QE takes it to 0.61x. `BAND_RESPONSE_MODE='balanced'` models
per-band integration tuning, where F0 and QE cancel and only `mu0` survives.

**Solar spectrum is swappable.** `SOLAR_SPECTRUM='blackbody'` (Planck at `SOLAR_T`)
or `'table'` (two-column wavelength[um]/irradiance[W m-2 um-1] file via
`SOLAR_TABLE_PATH`). Intended upgrade: **Thuillier et al. 2003**; ASTM E490-00a AM0
and TSIS-1 HSRS are equivalent drop-ins. The blackbody gives 1789 W m-2 um-1 at
0.549 um against a measured ~1850, and gets the blue SHAPE wrong (line blanketing
depresses the real Sun there), so the 0.441 band is probably over-weighted. Absolute
scale errors cancel exactly because `SAT_RADIANCE` derives from the same curve; only
band-to-band shape matters. Both paths evaluate F0 at the band CENTRE — a measured
table should ideally be convolved with the band response first.

### Path 3 — `_errsim_grasp_assumed` (errsim03)

Injects exactly the BCK YAML's `noises` block, read at runtime so it tracks the
retrieval settings. Touches no calibration-sim product.

**The quantity the sigmas apply to depends on `measurement_fitting.polarization`**
(GRASP's `iPOBS`). With `relative_polarization_components` (iPOBS=2, what these YAMLs
use) GRASP fits `q=Q/I` and `u=U/I`, so an `absolute` sigma there is absolute in
**q,u units, not Stokes units**. Path 3 perturbs q,u and multiplies back through the
noised intensity so the fitted q',u' carry exactly the drawn error. Injecting it as a
Stokes-unit error — as the code originally did — was wrong by ~10x. Unsupported
`iPOBS` modes raise rather than guessing.

## 5. Per-wavelength instrument model (important)

**Each wavelength channel is treated as an independent instrument.** On every
`customErrModel` call a fresh instrument index is drawn (WITHOUT replacement); its
characteristic matrix `A` is used for all view-angles of that wavelength; Path 2
also draws one random `C` from that same instrument.

Rationale: per-channel optics (polarizer transmission/extinction, compensator
retardance, absorber) are spectrally distinct, so the cal sim's
instrument-to-instrument spread is a reasonable proxy for one instrument's
channel-to-channel variation (picture a spectral beam splitter feeding a separate
polarizer/sensor per wavelength).

**Consequence:** calibration error is currently **uncorrelated across
wavelengths** by construction. See open items for the planned fix.

## 6. In-memory matrix store (`_MatrixStore`)

Loaded **once per process** (module-level singleton), held in RAM; per-pixel access
is pure array indexing — no per-pixel file I/O.

- `char_mats` — (N_instr, 3, 3) true characteristic matrices (V column dropped).
- `cal_mats`  — (N_instr, 3, 3, N_cal) fitted calibration matrices.
- `cov_C`     — (9, 9) covariance of `vec(C)`, **loaded explicitly** from a CSV
  (never recomputed from the MC data — the cal sim uses a custom solver centered on
  the true mean `μ = vec(inv(A))`, and we want it swappable for test cases).
- `draw_index()` — WITHOUT-replacement deck (shuffled permutation). On exhaustion:
  reshuffle + warn once (default), or raise if `reshuffle_on_exhaust=False`.

### Configuration (header of `customErrModel.py`)

```
CAL_UNCERTAINTY_DIR   env var           # overrides the cal-sim repo location per machine
CAL_MATRIX_H5_PATH = <cal run .h5>       # None -> dummy synthetic pool
COV_MATRIX_PATH    = <covariance CSV>    # None -> dummy diagonal covariance
USE_RAD_CAL        = True                # fold radiometric gain into C (see above)
EAGER_LOAD         = False               # True -> build store at import

SENSOR_BIN_FACTOR  = 10                  # only when BIN_VIEW_DEPENDENT=False
SENSOR_GAIN_E_PER_ADU = 2.0              # must match the cal sim's gain_e_per_adu
SENSOR_READ_NOISE_E   = 5.0              # electrons per pixel read
SENSOR_BINNING     = 'digital'           # or 'onchip'

BIN_VIEW_DEPENDENT = True                # per view SECTOR, 100 at nadir -> 1 at edge
BIN_NADIR/BIN_EDGE = 100 / 1
VIEW_SECTOR_ANGLES = +/-[57,44,32,19,6]
VIEW_BAND_WVLS     = (0.441, 0.549, 0.669, 0.873)

GRASP_I_IS_REFLECTANCE = True            # apply the mu0 correction
SOLAR_SPECTRUM     = 'blackbody'         # or 'table' + SOLAR_TABLE_PATH (Thuillier)
SOLAR_T            = 5800.0
SAT_REF_WVL        = 0.669               # red: the cloud-microphysics band
SAT_RADIANCE       = None                # None -> F0(ref)/pi = 483.3 W m-2 sr-1 um-1
QE_BY_BAND         = None                # None -> flat QE
BAND_RESPONSE_MODE = 'common'            # shared integration time; or 'balanced'
BCK_YAML_PATH      = None                # Path 3 reads its sigmas here
```

Currently pointed at `/Users/nsienkie/working/uncertainty/nsienkie-cal-uncertainty`:
- `stor_data/eval_test/2026-09-15T16:46:06.h5`
  (N_instr=1000, N_cal=500, rad-cal=scalar, expected-Stokes model=tilted,
  calibration bin_size=100, read_noise_e=5, gain=2, seed unset)
- `eval_results/output/csv/covariance_matrix_radcal.csv`
  (9x9, radcal, bin-10; regenerate with `./eval_results/run_eval.sh`)

**Both must be regenerated together.** The covariance CSV is derived from the .h5,
and both encode the detector noise model in force when the cal sim ran — a pool built
with an older `instr_err` is inconsistent with a newer one.

### Loaders / readers
- `_load_matrices_from_hdf5` — `h5py` read of `results/characteristic_matrices`
  (drop V → (N,3,3)) and `results/calibration_matrices` (N,3,3,N_cal).
- `_read_cov_csv` — stdlib `csv` (handles the quoted `C[i,j]` labels from
  covariance_matrix.py; also accepts a plain numeric 9×9).
- Dummy fallbacks: `_load_dummy_matrices` (ideal 0/45/90 Pickering-form A perturbed
  ~2%), `_dummy_covariance` (diagonal, σ=0.02).

## 7. `mc_test.py` — measurement-space test harness

A truncated stand-in for the retrieval sim: builds the pixel via `returnPixel`,
fakes the forward run with **randomly generated physical Stokes** (DoP≤1), and calls
`measVals[l]['errorModel'](l, rsltFwd)` per wavelength — the *exact* call
`populateFromRslt` makes (so it exercises the full addError→customErrModel→store
path). Produces a saved matplotlib histogram of `(perturbed − truth)` for I, Q, U,
DoLP, one row per path.

```
python err_sim/mc_test.py [N_pixels]      # default 500 -> err_sim/mc_test_hist.png
```
Representative result (1000-instrument pool, radcal folded, 400 pixels x 4λ):

| | ΔI std | ΔQ std | ΔU std | ΔDoLP std |
|---|---|---|---|---|
| Path 1 | 1.95e-3 | 1.98e-3 | 3.36e-3 | 1.59e-2 |
| Path 2 | 1.96e-3 | 1.96e-3 | 3.38e-3 | 1.56e-2 |

The two paths now agree component-by-component, and both are consistent with the cal
sim's own `error_propagation_summary_radcal_bin10.csv` (σ_MC vs σ_prop_cov match to 4
digits there, over 1000 instruments x 500 cal x 100 Stokes). Path 2's residual bias is
small (ΔI mean ~3e-6) because the scalar gain removes most of the systematic scale error.
At bin=10 the error budget is **detector-noise dominated**: calibration contributes only
3.6% of the total covariance by Frobenius norm (88% if the gain is left unfolded).

## 8. Environment

Needs `numpy`, `scipy`, `pandas`, `pyyaml` (for the `runGRASP`/`architectureMap`
import chain), `h5py` (matrix read), `matplotlib` (plotting) and `netCDF4` (orbital
geometry).

- This Mac: the **`grasp_sim`** conda env (`~/miniforge3/envs/grasp_sim`).
  `optics` runs the cal sim and `customErrModel` standalone but lacks pandas/pyyaml.
- GRASP must be the **GCC 13** build — see `../../../GRASP_BUILD_NOTES.md`. GCC 15/16
  produce a binary that segfaults in the SOS RT path.

**NumPy 2 shim (`np_compat.py`).** The read-only `GSFC-GRASP-Python-Interface`
predates NumPy 2 and breaks on it twice: `np.trapz` was removed (9 call sites), and
`miscFunctions.loguniform` passes a 1-element ndarray to `math.log`, which NumPy 2 no
longer auto-converts. Both are patched from our side rather than editing the
dependency. **Import `err_sim.np_compat` before anything that pulls in `runGRASP`** —
`run_experiment.py`, `mc_test.py`, `summarize_paths.py` and `plot_path_comparison.py`
all do.

## 8b. Running an experiment (`run_experiment.py`)

Single entry point: forward GRASP "truth" -> err_sim error model -> GRASP inversion ->
pickle -> `analyzeSim` stats + PNGs. Tunables sit in the `CONFIG` block.

- `INSTRUMENT` — `'harperrsim'` | `'harperrsimmc'` | `'harperrsimbck'`.
  Override without editing: `ERRSIM_INSTRUMENT=harperrsimbck python ...`.
- `N_PIX` — pixel count; `ERRSIM_NPIX=all` takes every valid pixel (526 for the AOS
  file). `NSIMS` noise repeats per pixel. `MAX_CPU` parallel GRASP procs.
- `MAX_T = 25` — **max pixels per GRASP process.** `graspDB` splits the inversion into
  `ceil(Npix/MAX_CPU)` chunks; all our pixels share `ix=iy=1` and differ only in time,
  so the binding constant is `_KITIME` (**30** in this build), NOT `_KIMAGE` (=120).
  Leaving this unset silently exceeds it once `Npix > 30*MAX_CPU` and GRASP dies.
- `GEOM_SOURCE = 'nc4'` — real orbital geometry via `ACCP_functions.selectGeomSabrina`,
  the same source `runRetrievalSimulation.py` uses; pixels walked in cumulative-index
  order, skipping `sza` outside `(0, MAX_SZA]`. `'random'` restores the old uniform
  sweep. The `harperrsim*` archs now honour the file's per-pixel `vza` and per-view
  azimuth (falling back to HARP2's hardcoded angles when `vza is None`).
- `TAU_SEED` — **seeds the AOD draw.** Required for comparing paths: without it each
  run draws a different truth scene and path differences drown in scene scatter.
- `RUN_RETRIEVAL=False` re-plots an existing pkl. `PLOT_WAVE_IND`, `MAKE_PLOTS`,
  `RND_INITIAL_GUESS`, `DIR_GRASP`, `KRNL_PATH`.

**All three paths invert with the same BCK YAML**
(`settings_BCK_POLAR_2modes_errsim_I1pct.yml`, sigma_I = 0.01 relative, sigma_q =
sigma_u = 0.005 absolute) so differences come only from the injected error model.
Path 3 additionally *injects* those sigmas, which is what makes it the control.
Runs made before this change had Paths 1/2 at 3% and are NOT comparable.

**AOD scaling gotcha.** `marineVariable` gives AOD(549) = **1.455 per unit loading**,
not 1.0 as `canonicalCaseMap.py` claims, so `randLogNrm<m>` yields a median AOD of
~`1.455*m`. `TAU_FACTOR='randLogNrm0.2'` is therefore a median AOD near 0.29.

**Local GRASP build caveat.** `CONSTANTS_SET=generic` caps aerosol modes at 2
(`_KSD 2`), so `CONCASE` must be a single 2-mode case. **Timing:** ~5 s/pixel at
`MAX_CPU=12`, so 526 pixels ~44 min/path, ~2.2 h for all three.

## 8c. Overnight campaigns (`run_overnight.sh`)

`./err_sim/run_overnight.sh [Npix]` runs all three paths over every geometry pixel,
then `summarize_paths.py` and `plot_path_comparison.py`. Results land in a timestamped
`err_sim/runs/<stamp>/` with `SUMMARY.txt`, `path_comparison_*.png`, per-path logs and
pickles. Read `SUMMARY.txt` in the morning.

Hard-won details, each from a failure:

- **`ulimit -n 8192` is set by the script.** `graspDB.processData` opens a
  `Popen(stdout=PIPE)` per `graspRun` and closes them only at the end, so the forward
  stage holds one descriptor per pixel. macOS's default 256 dies ~1/3 of the way
  through a 526-pixel run with `OSError: [Errno 24]`.
- **Pre-existing artefacts are stashed** into `_stale_pickles_from_previous_runs/`
  before the run. `summarize_paths.py` reads `experiment_*.pkl` by name, so a failed
  path would otherwise be silently summarised from an EARLIER run's pickles — which
  is exactly how one overnight campaign reported healthy 6-pixel results after all
  three paths had errored.
- **The summary is skipped entirely if every path fails**, and `SUMMARY.txt` says so.
- `summarize_paths.py` prints each pickle's write time, flags anything >3 h old as
  `STALE?`, and distinguishes "smaller scene set is a SUBSET of the largest" (benign:
  inversion failures) from "scene sets diverge" (different runs: invalid comparison).

## 9. Open items / future work

- **Solar spectrum -> Thuillier.** `SOLAR_SPECTRUM='table'` + `SOLAR_TABLE_PATH` is
  wired and tested; just needs the data file. Blackbody gets the blue shape wrong.
- **QE per band.** `QE_BY_BAND` hook exists, currently flat. This is the dominant
  band-to-band term (silicon takes the 873 band to 0.61x the 549 band in SNR), so it
  matters more than the solar-spectrum upgrade.
- **Verify the reflectance convention.** The `mu0` factor rests on GRASP's `fit_I`
  being reflectance. A Lambertian surface with aerosol and molecular scattering off
  would settle it: reflectance gives SZA-flat `I = A`, normalized radiance gives
  `I = A*mu0`.
- **Path 3 inversion failures.** An earlier 526-pixel campaign lost 350 pixels to
  whole-chunk GRASP crashes (14 of 22 processes). Replaying the same data crashed
  zero times, so it was never root-caused; the q,u units bug (Path 3 was injecting
  ~10x wrong Q/U noise) is the prime suspect and may have fixed it. The untested
  remaining difference was `rndIntialGuess=True`, which the replay did not apply.
- **Cross-wavelength covariance.** Each wavelength still draws an independent
  instrument, so calibration error is uncorrelated across bands by construction.
  Planned fix: rerun the cal sim with wavelength-specific means and the same seed.
- **sigma downstream.** Path 1's `LAST_ANALYTIC_SIGMAS` (now also carrying
  `bin_factor` and `counts_scale`) are computed but only stashed.
- **Correlation caveats.** Path 1 draws independent per-angle, diagonal-only noise;
  real calibration error is correlated across angles (same C) and across I/Q/U.
  Path 2 captures the across-angle correlation; Path 1 does not.
- **Instrument pool size.** N_instr=1000; a campaign draws `N_PIX*NSIMS*Nlambda`
  instruments, so >250 pixels at 4 bands triggers the reshuffle warning.
- **`NSIMS=1`.** Single realization per pixel; raise it for statistically resolvable
  path differences.

## 10. File map (err_sim/)
- `customErrModel.py` — the error model: store, loaders, dispatcher, all three paths,
  detector noise, sector binning, radiometry, dummy fallbacks.
- `run_experiment.py` — full-pipeline runner (CONFIG block -> retrieval -> stats+PNGs).
- `run_overnight.sh` — unattended all-paths campaign; see 8c.
- `summarize_paths.py` — one readable three-path report, scored on common scenes.
- `plot_path_comparison.py` — overlay scatter, one colour per path.
- `np_compat.py` — NumPy 2 shims for the read-only interface repo; import first.
- `mc_test.py` — measurement-space test harness + histogram (no GRASP).
- `ERR_SIM_OVERVIEW.md` — this file.
- Generated (git-ignored): `experiment_*.pkl`, `*.png`, `runs/`.
- `__init__.py` — makes `err_sim` importable.

Related (outside err_sim): `../ALTERING_THE_MEASUREMENT_ERROR_MODEL.md`,
`../ACCP_ArchitectureAndCanonicalCases/architectureMap.py` (`returnPixel`, `addError`),
`settings_BCK_POLAR_2modes_errsim_I1pct.yml` (the shared assumed-noise settings),
and `../../../GRASP_BUILD_NOTES.md` (why GRASP must be built with GCC 13).
