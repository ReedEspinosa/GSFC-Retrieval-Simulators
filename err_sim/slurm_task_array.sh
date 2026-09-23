#!/usr/local/bin/bash
#SBATCH --job-name=errsimTask
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH -o errsim.%A-%a.out
#SBATCH -e errsim.%A-%a.err
#SBATCH --account=s3324
#SBATCH --array=0-249%84
# =============================================================================
# One err_sim task per array index: ONE instrument (one pool entry per wavelength
# channel) with ONE calibration event, over the FULL 526-pixel geometry.
#
#   array range  0-249  -- the FULL instrument pool.  The calibration HDF5 holds 1000
#                          instruments and each task consumes 4 (one per band), so 250
#                          tasks is the exact maximum; task 249 takes 996-999.  Going
#                          further needs a larger calibration run -- run_task.py
#                          refuses rather than silently reusing instruments.
#   %84               -- cap on tasks resident at once, NOT a reservation: SLURM simply
#                          runs as many as it can schedule, so a higher cap can only
#                          shorten the campaign, never lengthen it.  84 makes 250 tasks
#                          drain in 3 passes.  Lower it to be a quieter neighbour on a
#                          busy cluster; raise it toward 250 to finish in one.
#
# Every task uses IDENTICAL scenes (paired design), so the spread across tasks is
# attributable to calibration alone.  Instrument/calibration indices, the initial
# guess and the measurement-noise stream all derive from the array index, so any task
# replays exactly.
#
# -----------------------------------------------------------------------------
# SIZING -- measured, not guessed.  From the 51-task 2026-09-23 run (log/*.out):
#
#   * 63.7 CPU-s per pixel (forward + inversion) for the marine scene, spread only
#     +-9% across tasks (1743-2102 CPU-s for 30 pixels).  A local A/B put the smoke
#     scene at 0.65x that, i.e. ~42 CPU-s/px; we plan with 55 for margin.
#   * That run requested --cpus-per-task=40 and was ALLOCATED 126 -- these nodes are
#     handed out whole -- so 86 cores sat idle and whole-job CPU utilisation was
#     10.3%.  Hence --exclusive plus the core autodetect below.
#   * Memory is driven by CONCURRENCY, not pixel count: 30 concurrent GRASP processes
#     peaked at 42 GB, i.e. ~1.4 GB each.  At ~106 concurrent that is ~150 GB against
#     the 488 GB the node reports.  Comfortable.
#   * Walltime used was 2.5 min of the 2 h requested (2%).
#
# Expected cost of THIS configuration, per task:
#     526 px, chunk = ceil(526/126) = 5 px -> 106 chunks, all resident in one wave
#     ~8.0 CPU-hours -> ~8 min wall (vs ~22 min if we had stayed at 40 cores)
#     -> ~12 min with slack, i.e. a 10x margin inside the 2 h limit.
# Whole array: 250 tasks at 84 resident = 250 x ~7 min / 84 ~= 21 min, or ~36 min at
# the conservative 12 min/task.  ~2000 CPU-hours, 131,500 retrievals, ~1.5 GB of
# pickles.  Tasks are NOT wave-synchronised -- SLURM backfills each slot as it frees --
# so the elapsed time is total-work/concurrency rather than passes x slowest-task.
#
# Statistics this buys: 5x the instruments of the 2026-09-23 run and 17.5x the pixels
# each, so the per-task AOD bias standard error falls by sqrt(526/30) = 4.2x, from
# ~0.008 to ~0.0019 -- below the ~0.003 calibration signal, which is the first time
# the instrument-to-instrument trend should be resolvable at all.
#
# Queue wait is NOT charged against --time (the last run pended 55 min), so the 2 h
# limit applies only to the running task.
# -----------------------------------------------------------------------------
#
# Submit:   mkdir -p log && sbatch -o log/errsim.%A-%a.out -e log/errsim.%A-%a.err \
#               err_sim/slurm_task_array.sh
# Collect:  python err_sim/collect_tasks.py err_sim/tasks tasks_summary.csv
# Analyse:  python err_sim/analyze_tasks.py err_sim/tasks
#           python err_sim/plot_bias_survey.py err_sim/tasks
#
# LOGS.  -o is stdout, -e is stderr; %A is the array's master job id and %a the task
# index, so task 3 of job 12345 writes errsim.12345-3.out / .err.
#
# These paths are relative to the directory you ran `sbatch` FROM (SLURM sets the
# job's working directory to the submission directory), NOT to this script.  They are
# deliberately written into that directory rather than a log/ subdirectory: SLURM
# opens the output files BEFORE the job script runs, and it does NOT create missing
# parent directories, so `mkdir -p log` inside the script is far too late -- the job
# fails or the output is discarded before line 1 executes.  Create log/ before
# submitting and pass it explicitly, as in the Submit line above.
# =============================================================================

# Descriptor headroom: graspDB holds one open pipe per forward pixel -- the default
# 256 dies partway through a 526-pixel run.
ulimit -n 8192 2>/dev/null || echo "WARNING: could not raise ulimit -n"

# --- external trees ----------------------------------------------------------
# Everything err_sim needs from outside this repo -- the grasp binary, the GRASP
# kernels, the orbital geometry .nc4 and the cal-sim HDF5/CSV -- is DISCOVERED under
# ERRSIM_BASE rather than hardcoded, so laptop and cluster share one checkout.  On the
# cluster the trees are copied side by side under a single base:
#
#     /gpfsm/dnb33/nsienkie/retr_sim/
#         GSFC-Retrieval-Simulators/      <- this repo (SLURM_SUBMIT_DIR, usually)
#         GSFC-GRASP-Python-Interface/
#         grasp/
#         nsienkie-cal-uncertainty/
#         data_stor/
#
# The default (this repo's parent) is already correct for that layout, so this line is
# belt and braces -- it keeps the run working if the job is submitted from elsewhere.
# Override any single item with ERRSIM_GRASP_BIN / ERRSIM_GRASP_KERNELS /
# ERRSIM_GEOM_NC4 / CAL_UNCERTAINTY_DIR / ERRSIM_CAL_H5 / ERRSIM_COV_CSV.
#
# Verify a new machine BEFORE submitting:   python err_sim/check_paths.py
export ERRSIM_BASE="${ERRSIM_BASE:-/gpfsm/dnb33/nsienkie/retr_sim}"
[[ -d "$ERRSIM_BASE" ]] || echo "WARNING: ERRSIM_BASE=$ERRSIM_BASE is not a directory; falling back to discovery"

# --- scene under test --------------------------------------------------------
# SMOKE OVER OCEAN.  Polarimetric signal is strongest for smoke, which is why this
# campaign uses it rather than the marine default.
#
# 'smokeVariable' IS an ocean case: canonicalCaseMap sets
#     landPrct = 100 if ('vegetation' or 'desert') in caseStr else 0
# so smoke defaults to ocean and only the literal tokens 'Desert'/'Vegetation' make
# it land.  BEWARE: a 'Land' suffix is NOT tested anywhere -- 'smokeLand...' silently
# produces an OCEAN scene.  Write 'smokeDesert'/'smokeVegetation' if you ever want land.
# 'Variable' adds 1-sigma 20% scatter to rv, sigma, height, n and k per pixel.
export ERRSIM_CONCASE="${ERRSIM_CONCASE:-smokeVariable}"

# Median AOD 0.3 (sigma_g = ln 2, so 95% of draws land in ~[0.075, 1.2]).  Thicker
# than the 0.2 marine default: smoke plumes are optically thicker in reality and the
# polarized signal this campaign measures grows with optical depth.
export ERRSIM_TAU_FACTOR="${ERRSIM_TAU_FACTOR:-randLogNrm0.3}"

# Retrieval settings whose a-priori box can actually REPRESENT the smoke truth.  With
# the marine box, 49% of fine-mode k truth lay above the 0.01 cap and 46% of coarse rv
# below the 0.65 floor, so the retrieval was pinned and could not respond to
# calibration error at all.  See the header of this file's companion YAML.
export ERRSIM_BCK_YAML="${ERRSIM_BCK_YAML:-settings_BCK_POLAR_2modes_errsim_smoke.yml}"

# Path 2 (Monte Carlo) is the path that carries calibration bias.
export ERRSIM_INSTRUMENT="${ERRSIM_INSTRUMENT:-harperrsimmc}"

# --- parallelism -------------------------------------------------------------
# run_experiment resolves MAX_CPU as ERRSIM_MAXCPU -> SLURM_CPUS_PER_TASK -> 12.
# Under --exclusive, SLURM_CPUS_PER_TASK is 1, so set it from the cores actually on
# the node.  Leave a few for the parent python, HDF5 and the filesystem.
NCORE="${SLURM_CPUS_ON_NODE:-$(nproc 2>/dev/null || echo 12)}"
RESERVE=6
ERRSIM_MAXCPU_AUTO=$(( NCORE > RESERVE + 4 ? NCORE - RESERVE : NCORE ))
export ERRSIM_MAXCPU="${ERRSIM_MAXCPU:-$ERRSIM_MAXCPU_AUTO}"

# --- concurrency hygiene -----------------------------------------------------
# 1. Per-task TMPDIR.  Paired scenes require seeding the global numpy stream
#    identically in every task, which also makes graspYAML generate IDENTICAL
#    "unique" temp-YAML filenames across tasks.  Node-local, per-task TMPDIR keeps
#    those names in separate directories so they cannot collide, and keeps GRASP's
#    working dirs off the shared filesystem.
export ERRSIM_TASK_TMPDIR="${SLURM_TMPDIR:-/tmp}/errsim_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$ERRSIM_TASK_TMPDIR"

# 2. HDF5 file locking.  Both the geometry .nc4 and the calibration .h5 are opened
#    READ-ONLY by every task at once; HDF5's lock files fail on some parallel
#    filesystems (Lustre/GPFS) under that pattern.
export HDF5_USE_FILE_LOCKING=FALSE

date
hostname
echo "--- err_sim task ${SLURM_ARRAY_TASK_ID} ---"

PYTHON="${PYTHON:-python}"

# --- locate the repository ---------------------------------------------------
# SLURM COPIES this script into a spool directory before running it, so
# ${BASH_SOURCE[0]} is /var/spool/slurmd/... under sbatch -- NOT where the script
# lives.  Deriving the repo from it therefore works locally and fails on the
# cluster with:
#     python: can't open file '/var/spool/slurmd/err_sim/run_task.py'
# Resolve in order of decreasing reliability, and verify before using:
#   1. ERRSIM_REPO         -- explicit, survives any submission directory
#   2. SLURM_SUBMIT_DIR    -- where sbatch was run from (and its parent)
#   3. dirname of this script -- correct only when NOT spooled, i.e. running locally
REPO=""
for cand in "${ERRSIM_REPO:-}" \
            "${SLURM_SUBMIT_DIR:-}" \
            "${SLURM_SUBMIT_DIR:-}/GSFC-Retrieval-Simulators" \
            "${SLURM_SUBMIT_DIR:-}/.." \
            "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"; do
    if [[ -n "$cand" && -f "$cand/err_sim/run_task.py" ]]; then
        REPO="$(cd "$cand" && pwd)"
        break
    fi
done
if [[ -z "$REPO" ]]; then
    echo "FATAL: cannot locate the GSFC-Retrieval-Simulators checkout." >&2
    echo "       Looked for err_sim/run_task.py under ERRSIM_REPO, SLURM_SUBMIT_DIR" >&2
    echo "       (='${SLURM_SUBMIT_DIR:-unset}') and this script's directory." >&2
    echo "       Submit from the repo root, or: sbatch --export=ALL,ERRSIM_REPO=/path/to/repo ..." >&2
    exit 1
fi
cd "$REPO" || exit 1

# Echo the resolved configuration.  The 2026-09-23 run could not be identified after
# the fact because none of this was recorded; run_task.py now also writes it into
# each task's provenance JSON.
echo "repo:       $REPO"
echo "scene:      $ERRSIM_CONCASE   (smoke over OCEAN; 'Land' suffixes are inert)"
echo "AOD draw:   $ERRSIM_TAU_FACTOR"
echo "bck yaml:   $ERRSIM_BCK_YAML"
echo "instrument: $ERRSIM_INSTRUMENT"
echo "cores:      $ERRSIM_MAXCPU of $NCORE on node"
echo "pixels:     ${ERRSIM_NPIX:-all (526)}"

# No second argument -> every valid geometry pixel (526).  Do NOT pass a count here:
# a positional value pins the run to that many pixels, and the previous campaign
# silently retrieved 30 per task because the default had not been made explicit.
"$PYTHON" -u err_sim/run_task.py "${SLURM_ARRAY_TASK_ID}"
rc=$?

# Clean up node-local scratch so repeated array runs do not fill it.
rm -rf "$ERRSIM_TASK_TMPDIR"

date
echo "--- task ${SLURM_ARRAY_TASK_ID} exit ${rc} ---"
exit $rc
