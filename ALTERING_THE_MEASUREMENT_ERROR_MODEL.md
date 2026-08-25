# Altering the Measurement Error Model and Re-running Retrieval Simulations

This guide explains how the measurement error (noise) model works in
`GSFC-Retrieval-Simulators`, exactly where it lives, how to change it, and how to
re-run a single case or a large batch of retrieval simulations with the modified
model. It references the two supporting repositories where needed:

- **`grasp`** – the radiative transfer + inversion binary. Reads an SDATA text
  file + a settings YAML, runs either a *forward* calculation or an *inversion*.
- **`GSFC-GRASP-Python-Interface`** – Python glue (`runGRASP.py`) that builds
  pixels, writes SDATA, spawns the GRASP binary, and parses its text output.
- **`GSFC-Retrieval-Simulators`** – this repo. The orchestration layer that
  defines the scenes, the instruments, **the error model**, and drives the
  forward→noise→inversion loop.

> **Only this repository (`GSFC-Retrieval-Simulators`) should be edited.** The
> other two are treated as read-only dependencies.

---

## 1. Mental model: what a "retrieval simulation" actually does

A single simulated retrieval is a round trip:

```
                  scene definition (canonicalCaseMap.py)
                             │
                             ▼
   [1] FORWARD   GRASP forward mode  ──►  rsltFwd  ("truth" I/Q/U, βext, βbks …)
                             │
                             ▼
   [2] ADD NOISE   error model perturbs the truth   ◄── THIS is what you change
                             │
                             ▼
   [3] INVERSION  GRASP inversion mode ──► rsltBck  (retrieved state parameters)
                             │
                             ▼
   [4] ANALYSIS   compare rsltBck vs rsltFwd  (error stats, plots, netCDF)
```

The **measurement error model** is step **[2]**: it takes the clean forward
"truth" radiances/signals and adds realistic instrument noise, producing the
synthetic measurements that are then handed to GRASP's inversion. Changing the
error model changes how noisy the pseudo-observations are, which is the whole
point of an OSSE-style "how good can this instrument retrieve?" study.

The top-level class that runs this loop is `simulation` in
`simulateRetrieval.py`; its `runSim()` method executes steps [1]–[4].

### Two distinct "errors" — do not confuse them

| | Where it lives | What it does |
|---|---|---|
| **(A) Forward noise added to the synthetic obs** | `addError()` in `ACCP_ArchitectureAndCanonicalCases/architectureMap.py` | The *actual* random perturbation applied to the truth in step [2]. **This is the "measurement error model" you almost always want to change.** |
| **(B) Assumed noise in the inversion (BCK) YAML** | `retrieval.inversion.noises` block of the `settings_BCK_*.yml` file | The noise GRASP *assumes* when weighting the cost function during inversion (step [3]). Values come from the template YAML; the sim code only lightly adjusts wavelength indexing. |

For a self-consistent experiment, if you change the magnitude of (A) you usually
want the assumed inversion noise (B) to be consistent with it. They are edited in
**different files** (see §5).

---

## 2. Where the forward error model lives, in detail

Everything for the forward noise is in one file:

```
GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
```

Two functions matter:

### `returnPixel(archName, sza, relPhi, vza, …)`  (line ~20)

Builds a dummy `pixel` object (an instance of `runGRASP.pixel`) describing the
instrument: wavelengths, viewing geometry, and measurement types. For each
wavelength it calls:

```python
errModel = functools.partial(addError, errStr)   # bind the error-model name
nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas, errModel=errModel)
```

The crucial line is the choice of **`errStr`** — a short string like `'polar07'`,
`'harp02'`, `'lidar05'`, `'modismisr01'`. That string selects *which* branch of
`addError()` runs. Each instrument block picks its `errStr`; e.g.:

- `harp02`  →  `errStr = 'polar07'`  (3 % I, 0.5 % DoLP)
- `harp20`  →  `errStr = 'polar10'`  (5 % I, 1.0 % DoLP)
- `rsp`     →  `errStr = 'polar1002'`
- `megaharp1noah` → `errStr = 'harp02'`  (angle-dependent HARP2 model)
- adding `'00'` to an arch name (e.g. `harp0200`, `polar0700`) selects the
  near-noise-free `polar700` variant — the "clean"/perfect instrument.

`addMeas()` (in `GSFC-GRASP-Python-Interface/runGRASP.py`, line ~1177) stores the
bound function under `measVals[n]['errorModel']`. It is **not** written into
SDATA — it is a Python callable applied later, in memory.

### `addError(measNm, l, rsltFwd, …)`  (line ~281)

This is the actual noise generator. `measNm` is the `errStr` from above; `l` is
the wavelength index; `rsltFwd` is the forward "truth" dict. It parses the
trailing number of `measNm` (e.g. `polar` + `07`) and selects hard-coded 1-sigma
error magnitudes, then perturbs the truth. The three instrument families:

**Standard polarimeters** (`polarNN`) — lines ~289–335:
```python
relErr    = 0.03    # 1-sigma relative error on intensity I  (log-normal)
absDoLPErr = 0.005  # 1-sigma absolute error on DoLP          (normal)
...
noiseVctI = np.random.lognormal(sigma=np.log(1+relErr), size=len(trueSimI))
fwdSimI = trueSimI*noiseVctI            # scale I
fwdSimQ = trueSimQ*noiseVctI            # scale Q,U by same factor to preserve q,u
fwdSimU = trueSimU*noiseVctI
# then add independent DoLP noise to Q and U via absDoLPErr
return np.r_[fwdSimI, fwdSimQ, fwdSimU]
```
The `int(mtch.group(2))` cases enumerate the presets: 7/8→(3 %,0.005),
1/2/3→(5 %,0.005), 5→(2 %,0.003), 9→(3 %,0.003), 10→(5 %,0.010), 11 (POLDER),
12 (Noah's HARP2 RMSE), 700 (≈noise-free), and `≥1000` → custom DoLP = (N−1000)/1000.

**HARP angle-dependent polarimeters** (`harpNN`) — lines ~337–374: same idea but
the 1-sigma error grows with view angle:
```python
noiseFunc = lambda view, base : np.abs(view/view.max()/100) + base
relErr_byView   = noiseFunc(viewAng, relErr)
dolpErr_byView  = noiseFunc(viewAng, absDoLPErr)
```

**Lidar** (`lidarNN`) — lines ~376–441: handles attenuated backscatter (`LS`,
type 31) and HSRL (`VExt` 36 + `VBS` 39). Supports fixed relative/absolute errors,
"perfect" variants (500/600/900), and **Kathy's uncertainty models** (50/60/90)
read from files via `readKathysLidarσ()` (needs `concase`, `orbit`, `lidErrDir`).

**Intensity-only** (`modismisrNN`) — lines ~443–448: simple relative log-normal
noise on I.

If `measNm` matches none of these, `addError()` asserts and halts — so a new
instrument must have a matching branch (or you get
`'No error model found for …'`).

### How the model is actually applied

In `simulateRetrieval.runSim()` (line ~107) the loop over pixels does:
```python
nowPix.populateFromRslt(self.rsltFwd[i], radianceNoiseFun=radianceNoiseFun, …)
```
Inside `pixel.populateFromRslt()` (`runGRASP.py` ~1198) each measurement's stored
`errorModel` is called to perturb the truth:
```python
noisyMeas = msDct['errorModel'](l, rslt, verbose=…)   # == addError(errStr, l, rslt)
```
This happens **once per noise realization**. `Nsims` in the run scripts controls
how many independent noisy draws (and therefore retrievals) are performed per
forward scene — each draws fresh `np.random` noise. `fixRndmSeed=True` freezes the
seed so every pixel gets identical noise (only useful for debugging).

**Override hook:** `runSim(..., radianceNoiseFun=fn)` — if you pass a callable
here it *overrides* the per-measurement `errorModel` entirely
(`populateFromRslt` overwrites `measVals[n]['errorModel']` with it). Its signature
is `fn(wavelengthIndex_l, rsltDict, verbose)` returning the noisy measurement
vector — exactly the signature of `functools.partial(addError, errStr)`. This is
the cleanest way to inject a completely custom model without touching
`architectureMap.py` (see §4, Option C).

---

## 3. SDATA / YAML context (for reference)

You rarely touch these directly, but knowing the flow helps:

- The `pixel` object → written to an **SDATA** text file by
  `runGRASP.graspRun.writeSDATA()` / `pixel.genString()`. SDATA holds geometry,
  wavelengths, measurement-type codes (41=I, 42=Q, 43=U, 31=LS, 35=DP, 36=VExt,
  39=VBS, 12=AOD …) and the measurement values — **but not the noise model**. The
  noise has already been baked into the measurement values by step [2].
- The **forward** YAML (`settings_FWD_IQU_POLAR_1lambda.yml`) runs GRASP in
  `mode: forward` to make the truth. The simulator heavily rewrites it per scene.
- The **inversion / BCK** YAML (`settings_BCK_POLAR_2modes.yml`, etc.) runs
  `mode: inversion`. Its `retrieval.inversion.noises` block is error concept (B).
- Working examples of the SDATA format live in `grasp/examples/*/*.sdat`; run form
  is `grasp <settings.yml>` from inside the example folder.

---

## 4. Three ways to alter the measurement error model

Pick based on how invasive you want to be. **Option B is recommended** for a new
experiment because it is additive and non-destructive.

### Option A — Tweak an existing preset (fastest, but changes every arch that uses it)

Edit the magnitudes inside `addError()`. Example: make the standard polarimeter
noisier:
```python
# architectureMap.py, addError(), polar branch
elif int(mtch.group(2)) in [7, 8]:
    relErr    = 0.05     # was 0.03
    absDoLPErr = 0.010   # was 0.005
```
⚠️ Every instrument whose `errStr` resolves to `polar07`/`polar08` now changes.
Good for a quick global sensitivity test, risky if you want to keep the original.

### Option B — Add a new preset + wire a new instrument (recommended)

1. **Add a new numeric preset** to `addError()` (an unused number, e.g. 20):
```python
elif int(mtch.group(2)) in [20]:   # my custom polarimeter
    relErr    = 0.04
    absDoLPErr = 0.007
```
2. **Add (or point) an instrument block** in `returnPixel()` to use it. Either
   add a brand-new `if 'mycustom' in archName.lower():` block, or set the
   `errStr` of an existing one:
```python
if 'megaharp1custom' in archName.lower():
    ...
    errStr = 'polar20'      # links to the preset added above
    ...
    errModel = functools.partial(addError, errStr)
    nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas, errModel=errModel)
```
3. In your run script set `instrument = 'megaharp1custom'` (or `'mycustom'`).

For a fully custom functional form (angle-, wavelength-, or signal-dependent),
add a whole new family branch, e.g. `if mtch.group(1).lower() == 'mypol':` that
returns `np.r_[fwdSimI, fwdSimQ, fwdSimU]` in the same ascending measurement-type
order the rest of the code guarantees.

### Option C — External override via `radianceNoiseFun` (no edits to architectureMap.py)

Define your own function and pass it to `runSim`:
```python
import numpy as np
def myNoise(l, rsltFwd, verbose=False):
    I = rsltFwd['fit_I'][:, l]
    Q = rsltFwd['fit_Q'][:, l]
    U = rsltFwd['fit_U'][:, l]
    n = np.random.lognormal(sigma=np.log(1+0.04), size=len(I))
    return np.r_[I*n, Q*n, U*n]     # must match the measurement layout for this arch

simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, …, radianceNoiseFun=myNoise)
```
This overrides the built-in model for *all* measurements. Best when prototyping or
when the model doesn't belong in the shared instrument map. Note it must return
the measurement vector in the exact per-type/per-angle order GRASP expects.

### Don't forget concept (B): make the inversion's assumed noise consistent

If your goal is a fair OSSE, edit the assumed noise in the **BCK YAML** to match
the new forward noise. In `ACCP_ArchitectureAndCanonicalCases/settings_BCK_*.yml`:
```yaml
retrieval:
  inversion:
    noises:
      noise[1]:                 # intensity I
        error_type: relative
        standard_deviation: 0.04     # match your new relErr
      noise[2]:                 # Q and U
        error_type: absolute
        standard_deviation: 0.007    # roughly match your DoLP error budget
```
The simulator will still auto-repeat these across wavelengths, so you normally set
one value per measurement type. (Leave `standard_deviation_synthetic: 0.0` — the
synthetic noise is added by our Python code, not by GRASP.)

---

## 5. Files you will edit — checklist

| Purpose | File |
|---|---|
| Forward noise magnitudes / new preset | `ACCP_ArchitectureAndCanonicalCases/architectureMap.py` → `addError()` |
| Wire a preset to an instrument name | `ACCP_ArchitectureAndCanonicalCases/architectureMap.py` → `returnPixel()` |
| Assumed inversion noise (consistency) | `ACCP_ArchitectureAndCanonicalCases/settings_BCK_*.yml` → `noises` |
| Choose instrument / scene / run params | your copy of `Examples/runRetrievalSimulation.py` (single) or `Examples/runRetrievalSimulationSlurm.py` + `camp2ex-configurations.yml` (batch) |

---

## 6. Re-running the retrieval

### 6a. Single case (smoke test — do this first)

Start from `Examples/runRetrievalSimulation.py`. Set the paths and the instrument
to your customized one, then run it:

```python
# Examples/runRetrievalSimulation.py  (edit these)
savePath   = './job/myCustomErrTest.pkl'
path2repoGRASP = '/path/to/grasp'                 # base GRASP repo
binGRASP   = os.path.join(path2repoGRASP, 'build/bin/grasp')
krnlPath   = os.path.join(path2repoGRASP, 'src/retrieval/internal_files')
fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpath      = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')

Nsims      = 10                      # 10 independent noise realizations
maxCPU     = 10                      # ideally >= Nsims so each noisy pixel is unique
conCase    = 'dustVariableOcean'
SZA, Phi   = 40, 5
instrument = 'megaharp1custom'       # <-- your new arch from §4 Option B
```

Then:
```bash
cd GSFC-Retrieval-Simulators/Examples
python runRetrievalSimulation.py
```
It builds the pixel, generates the forward truth, applies **your** error model
`Nsims` times, inverts each, and pickles results to `savePath`. It prints RMS
(retrieved − truth). Confirm the noise level changed as expected before scaling up.

> Tip: set `instrument='megaharp1custom00'`-style "clean" variant (near-zero
> noise) as a control run to verify the retrieval recovers the truth when noise
> is off — isolates model bias from noise sensitivity.

### 6b. Large batch (many scenes / AODs / geometries)

Two batch templates exist:

- `Examples/runRetrievalSimulation_loop.py` — plain Python nested loops over
  instrument / conCase / SZA / τ.
- `Examples/runRetrievalSimulationSlurm.py` — driven by a config YAML
  (`ACCP_ArchitectureAndCanonicalCases/camp2ex-configurations.yml`) and designed
  for **SLURM job arrays**. It loops over AOD (`tau`), flights, layers, and
  configurations, calling `runMultiple() → simA.runSim()` for each.

The batch scripts still call the *same* `returnPixel`/`runSim` path, so your error
model changes apply automatically. To point a batch run at your custom
instrument, set the `instrument` field:

- In `runRetrievalSimulationSlurm.py` it comes from `sys.argv[2]` (or the
  `ymlData['default']['forward']['instrument']` config), so pass it on the
  command line or set it in `camp2ex-configurations.yml`.
- The BCK YAML used per run is chosen from `ymlData['default']['retrieval']['yaml']`
  — make sure it points at the BCK file whose `noises` block you updated.

Submit with the provided helpers (they create the sbatch array and launch it):
```bash
cd GSFC-Retrieval-Simulators/Examples
python create_sbatch_array.py         # generate the array job
sbatch SLURM_runSimulation.sh         # or use slurmRunAuto.sh / stackSLURM.sh
```
Each array task runs one `(instrument, conCase, τ, geometry)` combination with
`Nsims` noisy retrievals, writing its own `.pkl`. Merge afterwards with
`mergeMultiPklFile.py`, and plot/analyze with the `plot*` /
`simulationQuickLook*` scripts.

### 6c. Key `runSim` arguments worth knowing

From `simulateRetrieval.runSim(fwdData, bckYAML, Nsims, …)`:

- `Nsims` — number of noise realizations per forward scene (= retrievals).
- `maxCPU` — parallel GRASP processes; set `>= Nsims` so identical-pixel runs
  don't collapse to the same noise (the code warns otherwise).
- `rndIntialGuess` — randomize the inversion's first guess (tests retrieval
  robustness independent of noise).
- `radianceNoiseFun` — the §4 Option C override.
- `dryRun=True` — forward + noise only, **no inversion** (fast check that your
  noise looks right; nothing is saved).
- `fixRndmSeed` — identical noise on every pixel (debugging only).
- `savePath` — output pickle; `lightSave=True` drops bulky fields.

---

## 7. Quick recipe (TL;DR)

To change the measurement error model and re-run:

1. **Edit** `architectureMap.py`:
   - add a numeric preset in `addError()` with your `relErr` / `absDoLPErr`
     (or new functional form), and
   - wire an instrument name to it in `returnPixel()` (Option B), **or** just
     pass a `radianceNoiseFun` to `runSim` (Option C).
2. **(Consistency)** update the `noises` block in the relevant
   `settings_BCK_*.yml`.
3. **Smoke test** with `Examples/runRetrievalSimulation.py`
   (`Nsims` small, one scene); use `dryRun=True` first to eyeball the noise, and a
   `…00` "clean" arch as a control.
4. **Scale up** via `runRetrievalSimulationSlurm.py` + `create_sbatch_array.py`
   (SLURM array), pointing `instrument` at your new arch and the BCK YAML at your
   updated file.
5. **Analyze** with `mergeMultiPklFile.py` + `plotErrorVsAOD.py` /
   `simulationQuickLook*.py`.
