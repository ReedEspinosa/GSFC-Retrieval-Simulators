# How the retrieval's initial guess is chosen — and why concentration is excluded

Investigated 2026-09-28 while interpreting the 250-instrument smoke campaigns. Records
what the initial guess actually does, because it is not obvious from the config and it
bears on how the campaign results should be read.

## How the guess is picked

`graspYAML.scrambleInitialGuess()` — `GSFC-GRASP-Python-Interface/runGRASP.py:1433`:

```python
rngBnd  = fracOfSpace*(uprBnd - lowBnd)/2
meanBnd = (lowBnd + uprBnd)/2
newGuess = np.random.uniform(meanBnd-rngBnd, meanBnd+rngBnd)
```

**The guess is entirely uncorrelated with truth.** It is drawn uniformly across the
whole a-priori box, not perturbed around an expected value. `RND_INITIAL_GUESS = True`
gives `fracOfSpace = 1`, so the draw covers essentially all of `[min, max]` (clamped to
0.999 so it never lands exactly on a bound).

Concretely for the smoke case: fine-mode `k` truth is 0.01 and the guess is drawn
log-uniformly anywhere in `[1e-6, 0.05]`, up to four decades away. Coarse `rv` truth is
0.66 and the guess is uniform in `[0.35, 4.9]`.

`k` and `aerosol_concentration` are drawn LOG-uniformly; everything else is linear.

### Shared across tasks, varying by chunk

`run_task.py` sets `GUESS_SEED_MODE = 'shared'`. The wrapper reseeds **both** numpy and
stdlib `random` from `GUESS_SEED_BASE + call_index` around each scramble call, then
restores the previous RNG state, so:

* task 0 and task 249 get **bit-identical** initial guesses — deliberate, so the
  across-task spread is attributable to calibration alone rather than guess luck;
* `scrambleInitialGuess` is called **once per GRASP chunk**, not per pixel
  (`runGRASP.py:93`), so at ~106 chunks of 5 pixels, **the 5 pixels sharing a chunk
  share an initial guess**. Their errors are therefore not fully independent — a
  correlation structure that is not obvious from the design description.

Both generators must be seeded because `miscFunctions.loguniform` uses stdlib `random`,
not numpy; seeding only numpy left the log-uniform draws unreproducible.

## Why `aerosol_concentration` is excluded

`skipTypes=['aerosol_concentration']` is the default, so concentration keeps its YAML
`value` and is never scrambled. No comment explains it. Three lines of evidence:

### 1. Git history

The original version excluded nothing (`7fe9fff`, `skipTypes=[]`). The exclusion appeared
in **`664418c`** (2019-11-22, *"adding option to randomize initial guess in simulation"*)
— the same commit that first wired the scramble into the simulation loop:

```diff
+   if rndGuess: self.grObjs[i].yamlObj.scrambleInitialGuess()
-   def scrambleInitialGuess(self, skipTypes=[]):
+   def scrambleInitialGuess(self, skipTypes=['aerosol_concentration']):
```

Written as a no-op, then excluded the instant it was first used for real. That reads as
an empirical fix rather than a design decision.

### 2. The numbers make it near-inevitable

| | value |
|---|---|
| a-priori box, fine mode | `[1e-8, 1.0]` — **8 decades** |
| truth concentration | 0.0029 – 0.237, median 0.035 |
| log-uniform draw median | 1e-4 |
| typical starting error | **~350x too low** |

No other retrieved parameter is close: `n` spans a factor of 1.2, coarse `rv` a factor
of 14, concentration **1e8**. The upper bound is not even physical — the YAML comment
says *"τfactor should not exceed unity"*, a sanity cap rather than prior knowledge.

Concentration also sets AOD almost linearly. Starting 350x low means starting from an
essentially aerosol-free atmosphere and asking a bounded iteration (nIter caps at 20
here) to climb three decades while every other parameter is simultaneously scrambled.
It starts in the optically-thin regime where sensitivity to the microphysical parameters
is weakest, so the early iterations have little gradient to work with.

### 3. Literature — supportive, not decisive

No source was found stating specifically why concentration's initial guess is excluded.
What the literature does support:

* GRASP-AOD sets its default initial guesses **from the observed AOD and Ångström
  exponent**, i.e. concentration's starting point is derived from the measurement —
  the opposite of a uniform draw ([AMT 14, 4471](https://amt.copernicus.org/articles/14/4471/2021/)).
* Multi-term LSM is a statistically-optimised iterative fit, hence a local optimiser;
  convergence from a far-off start is not guaranteed
  ([AMT 18, 7651](https://amt.copernicus.org/articles/18/7651/2025/)).
* Fine-mode volume concentration retrieval is described as stable and precise only under
  stated conditions ([RS 15, 5010](https://doi.org/10.3390/rs15205010)).

## Loose end: dead code

`scrambleInitialGuess` has a log-uniform branch **naming `aerosol_concentration`
explicitly**:

```python
if char['type'] in ['imaginary_part_of_refractive_index_spectral_dependent',
                    'aerosol_concentration']:
    newGuess = mf.loguniform(...)
```

That branch is unreachable under the default: concentration is always in `skipTypes`,
and the call site `scrambleInitialGuess(rndGuess)` passes positionally into
`fracOfSpace`, so `skipTypes` is never overridden. Someone implemented correct log-space
handling for an 8-decade range and excluded the parameter anyway — suggesting
log-uniform was tried and still was not enough.

## Consequence for the campaigns

Concentration's initial guess is **fixed at 0.001 (fine) / 0.002 (coarse) for every
pixel and every task**, against truth medians of 0.035 / 0.011 — about 35x and 5x low,
identically everywhere.

This is a candidate explanation for the `vol` result in `box_whisk`: median **+11.0%**,
the largest positive median of any parameter, IQR 38%. A systematically low, identical
starting point is a plausible driver of a systematic high bias in the converged answer.

**Untested.** To check: raise the YAML `value` for `characteristic[1]` toward the truth
medians, rerun a handful of tasks, and see whether the `vol` median bias moves. If it
does, the bias is a setup artifact rather than an information-content limit, and the
same question should be asked of every other parameter whose guess is fixed.
