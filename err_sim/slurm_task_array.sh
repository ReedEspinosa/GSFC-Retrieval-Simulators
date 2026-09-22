#!/usr/local/bin/bash
#SBATCH --job-name=errsimTask
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH -o errsim.%A-%a.out
#SBATCH -e errsim.%A-%a.err
#SBATCH --account=s3324
#SBATCH --cpus-per-task=10
#SBATCH --array=0-10
# =============================================================================
# One err_sim task per array index: ONE instrument (one pool entry per wavelength
# channel) with ONE calibration event, over the full geometry.
#
#   array range  0-249  -- a 1000-entry instrument pool supplies 1000/4 = 250 tasks.
#                          Rerun the calibration sim with more instruments to go past
#                          that; run_task.py refuses rather than wrapping around.
#   %50               -- at most 50 tasks resident at once, so the array does not need
#                          250 nodes; it drains in 5 waves.
#   --cpus-per-task   -- run_experiment picks MAX_CPU up from SLURM_CPUS_PER_TASK, so
#                          this single directive sets the parallelism.  graspDB runs
#                          one single-threaded GRASP process per inversion chunk, so
#                          cores == concurrent chunks.  At 526 pixels and 40 cores the
#                          chunk size is ceil(526/40)=14 (under MAX_T=25, itself under
#                          the build's _KITIME=30), giving 38 chunks -- one wave.
#
# Every task uses IDENTICAL scenes (paired design), so the spread across tasks is
# attributable to calibration alone.  Instrument/calibration indices and the
# measurement-noise stream all derive from the array index, so any task replays
# exactly.
#
# Submit:   sbatch err_sim/slurm_task_array.sh
# Collect:  python err_sim/collect_tasks.py err_sim/tasks tasks_summary.csv
#
# LOGS.  -o is stdout, -e is stderr; %A is the array's master job id and %a the task
# index, so task 3 of job 12345 writes errsim.12345-3.out / .err.
#
# These paths are relative to the directory you ran `sbatch` FROM (SLURM sets the
# job's working directory to the submission directory), NOT to this script.  They are
# deliberately written into that directory rather than a log/ subdirectory: SLURM
# opens the output files BEFORE the job script runs, and it does NOT create missing
# parent directories, so `mkdir -p log` inside the script is far too late -- the job
# fails or the output is discarded before line 1 executes.
#
# To keep logs in a subdirectory, create it BEFORE submitting and pass it explicitly:
#     mkdir -p log && sbatch -o log/errsim.%A-%a.out -e log/errsim.%A-%a.err \
#         err_sim/slurm_task_array.sh
# (or use an absolute path in the directives).
# =============================================================================

# Descriptor headroom: graspDB holds one open pipe per forward pixel (see
# run_overnight.sh) -- the default 256 dies partway through a 526-pixel run.
ulimit -n 8192 2>/dev/null || echo "WARNING: could not raise ulimit -n"

# --- concurrency hygiene -----------------------------------------------------
# 1. Per-task TMPDIR.  Paired scenes require seeding the global numpy stream
#    identically in every task, which also makes graspYAML generate IDENTICAL
#    "unique" temp-YAML filenames across tasks.  Node-local, per-task TMPDIR keeps
#    those names in separate directories so they cannot collide, and keeps GRASP's
#    working dirs off the shared filesystem.  run_task.py sets this itself if unset;
#    point it at real node-local scratch here.
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
echo "repo: $REPO"

# second arg omitted -> every valid geometry pixel
"$PYTHON" -u err_sim/run_task.py "${SLURM_ARRAY_TASK_ID}"
rc=$?

# Clean up node-local scratch so repeated array runs do not fill it.
rm -rf "$ERRSIM_TASK_TMPDIR"

date
echo "--- task ${SLURM_ARRAY_TASK_ID} exit ${rc} ---"
exit $rc
