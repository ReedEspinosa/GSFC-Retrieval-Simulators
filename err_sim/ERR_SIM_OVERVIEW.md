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

Both draw their instrument data from the **nsienkie-cal-uncertainty** Monte Carlo
calibration model (a 3-sensor division-of-amplitude polarimeter).

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
| `harperrsim`   | `errsim01` | Path 1 (analytic) |
| `harperrsimmc` | `errsim02` | Path 2 (Monte Carlo) |

`addError()` has an `errsim` branch that dispatches to `customErrModel`. (Single
arch block, suffix-selected, so `'harperrsim' in 'harperrsimmc'` doesn't double-fire.)
`architectureMap.py` also appends the repo dir to `sys.path` so `import err_sim` works.

Return contract (same as all addError models): `np.r_[I, Q, U]`, length `3*Nang`,
ascending measurement-type order.

## 4. The two error paths (math)

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

### Path 2 — `_errsim_montecarlo` (errsim02)
```
sensorInt   = A_true @ [I,Q,U]                    # (3, Nang)
noisySensor = sensorInt + N(0, σ_sensor)          # σ_sensor = instr_err(sensorInt)
out         = C @ noisySensor                      # C = one random draw from this instrument
```
`C ≠ inv(A_true)` → the recombination imparts the **systematic calibration bias**
plus random scatter. Vectorized over angles.

### Detector noise — `instr_err` (ported verbatim)
Copied (not imported) from `nsienkie-cal-uncertainty/lib/cal_uncertainty/
calibration.py`:
```
counts     = norm_intens * counts_max             # counts_max = 0.75*2^14 = 12288 (14-bit @ 75%)
counts_err = sqrt(|counts| / bin_size)            # shot noise
counts_err += (noise_floor*counts_max) / sqrt(bin_size)   # read-noise floor (10 counts)
sigma      = counts_err / counts_max              # back to normalized units
```
`SENSOR_BIN_FACTOR = 10` (matches the `radcal` bin-10 covariance we load).

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
CAL_MATRIX_H5_PATH = <cal run .h5>       # None -> dummy synthetic pool
COV_MATRIX_PATH    = <covariance CSV>    # None -> dummy diagonal covariance
EAGER_LOAD         = False               # True -> build store at import
SENSOR_BIN_FACTOR  = 10
```
Currently pointed at the sibling repo:
- `nsienkie-cal-uncertainty/stor_data/eval_test/2026-07-27T19:53:18.h5`
  (N_instr=100, N_cal=500)
- `nsienkie-cal-uncertainty/eval_results/output/csv/covariance_matrix_radcal.csv`
  (9×9, radcal, bin-10)

`init_store(h5_path=_UNSET, cov_path=_UNSET, ...)` — omit an arg to use the header
path; pass explicit `None` to force the corresponding dummy. `get_store()` lazy-inits
from the header paths on first use.

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
Representative result (real files, 500 pixels): Path 1 unbiased (ΔI std ~1.9e-3,
ΔDoLP std ~1.6e-2); Path 2 biased (ΔI mean ~+5e-5, ΔDoLP std ~1.5e-2, right-skewed).
Path 1 and Path 2 ΔDoLP spreads agree in magnitude — good analytic-vs-MC sanity check.

## 8. Environment

Runs in the `grasp` conda env (`/home/noahs/miniforge3/envs/grasp`). Dependencies
added during development: `pandas`, `pyyaml` (for `runGRASP`/`architectureMap`
import), `h5py` (real matrix read), `matplotlib` (mc_test plotting).

## 9. Open items / future work

- **Cross-wavelength covariance (planned approach).** The per-wavelength-independent
  model drops spectral correlation. Intended fix: rerun the calibration sim with
  wavelength-specific **mean** values matching real optics' spectral response (e.g.
  monotonic transmission vs λ), reusing the **same seed** so the angular/spatial
  (physical-rotation) uncertainties are identical across bands while
  transmission/phase uncertainties sit around different means. This retains
  cross-wavelength covariance. Requires reworking the cal-sim code — tabled.
- **Run the actual retrievals.** `mc_test` is measurement-space only. The science
  goal needs the full GRASP forward→noise→inversion loop (`runRetrievalSimulation*`
  with `instrument='harperrsim'`/`'harperrsimmc'`) and retrieval-vs-truth analysis.
- **σ downstream.** Path 1's `LAST_ANALYTIC_SIGMAS` are computed but only stashed.
  Decide whether they should also drive the inversion (BCK YAML `noises` block). Note
  the deliberate current design tests the retrieval under *incomplete* error info
  (hardcoded YAML noise ≠ injected calibration error).
- **Correlation caveats.** Path 1 draws independent per-angle, diagonal-only noise;
  real calibration error is correlated across angles (same C) and across I/Q/U
  (off-diagonal `Cov`). Path 2 captures the across-angle correlation; Path 1 does not.
- **Instrument pool size.** Real pool N_instr=100; with per-wavelength draws a
  typical campaign cycles it quickly (reshuffle warning). Rerun the cal sim with more
  instruments for tighter bias/covariance statistics.
- **Wavelength trend / shared analyzer angle.** Second-order refinements folded into
  the cross-wavelength plan above.

## 10. File map (err_sim/)
- `customErrModel.py` — the error model: store, loaders, dispatcher, both paths,
  `instr_err`, dummy fallbacks.
- `mc_test.py` — measurement-space test harness + histogram.
- `mc_test_hist.png` — latest test figure.
- `ERR_SIM_OVERVIEW.md` — this file.
- `__init__.py` — makes `err_sim` importable.

Related (outside err_sim): `../ALTERING_THE_MEASUREMENT_ERROR_MODEL.md`,
`../ACCP_ArchitectureAndCanonicalCases/architectureMap.py` (`returnPixel`,
`addError` errsim branch).
