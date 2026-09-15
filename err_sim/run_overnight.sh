#!/bin/bash
# =============================================================================
# run_overnight.sh -- full err_sim campaign over EVERY pixel in the orbital
# geometry file, for all three error-model paths, then summarise + plot.
#
#   Path 1 harperrsim     analytic calibration-uncertainty propagation (unbiased)
#   Path 2 harperrsimmc   Monte Carlo sensor noise + calibration bias
#   Path 3 harperrsimbck  control: injects exactly the noise its BCK YAML assumes
#
# Geometry: MAAP-GeometrySubSample_AOS_1330_LTAN_442km_alt_2023Aug12.nc4
#           660 entries -> 526 valid (0 < sza <= 70).  Real vza and per-view phi.
#
# Usage:
#     ./err_sim/run_overnight.sh                 # all 526 pixels, 3 paths
#     ./err_sim/run_overnight.sh 100             # cap at 100 pixels (a dry-run size)
#
# Safe to launch and walk away:
#   * each path runs independently; one failing does NOT abort the others
#   * every path writes its own timestamped log
#   * the summary + plot run over whatever paths actually produced a pickle
#   * results land in a timestamped run directory, so a rerun never overwrites
#
# In the morning, read:  <runDir>/SUMMARY.txt     and  <runDir>/path_comparison_*.png
# =============================================================================

set -uo pipefail          # NOT -e: a failed path must not kill the whole campaign

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PYTHON="${PYTHON:-$HOME/miniforge3/envs/grasp_sim/bin/python}"
NPIX="${1:-all}"                       # 'all' or an integer
PATHS=(harperrsim harperrsimmc harperrsimbck)

STAMP="$(date +%Y-%m-%dT%H%M%S)"
RUNDIR="$HERE/runs/$STAMP"
mkdir -p "$RUNDIR"
MAIN_LOG="$RUNDIR/run.log"

# Everything below is tee'd to the console AND to run.log
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "==========================================================================="
echo " err_sim overnight campaign"
echo " started : $(date)"
echo " host    : $(hostname -s)"
echo " python  : $PYTHON"
echo " pixels  : $NPIX"
echo " paths   : ${PATHS[*]}"
echo " rundir  : $RUNDIR"
echo "==========================================================================="

if [[ ! -x "$PYTHON" ]]; then
    echo "FATAL: python not found at $PYTHON (override with PYTHON=/path/to/python)"
    exit 1
fi

# --- file-descriptor headroom -------------------------------------------------
# graspDB.processData opens a Popen(stdout=PIPE) per graspRun and only closes
# those pipes after ALL of them have finished, so the forward stage holds one
# open pipe per pixel.  At 526 pixels that blows straight through macOS's default
# soft limit of 256 with 'OSError: [Errno 24] Too many open files'.  Raise it here
# rather than relying on whatever the launching shell happened to have.
NEED_FDS=8192
CUR_FDS=$(ulimit -n)
if [[ "$CUR_FDS" != "unlimited" && "$CUR_FDS" -lt "$NEED_FDS" ]]; then
    ulimit -n "$NEED_FDS" 2>/dev/null \
        && echo "file descriptors: raised $CUR_FDS -> $(ulimit -n)" \
        || echo "WARNING: could not raise ulimit -n above $CUR_FDS; runs >~200 pixels may fail"
else
    echo "file descriptors: $CUR_FDS (ample)"
fi

# Fail fast on a broken GRASP rather than discovering it 3 hours in.
export HERE
GRASP_BIN="$($PYTHON - <<'PY'
import re, os
src = open(os.path.join(os.environ['HERE'], 'run_experiment.py')).read()
print(re.search(r"^DIR_GRASP\s*=\s*'([^']+)'", src, re.M).group(1))
PY
)"
if [[ ! -x "$GRASP_BIN" ]]; then
    echo "FATAL: GRASP binary not executable: $GRASP_BIN"
    exit 1
fi
echo "GRASP: $GRASP_BIN"
"$GRASP_BIN" 2>&1 | sed -n '1,5p' | sed 's/^/       /'
echo

# --- stale-result guard -------------------------------------------------------
# summarize_paths.py reads err_sim/experiment_*.pkl by name.  If a path fails, a
# pickle left over from an EARLIER run would be summarised silently and look like a
# successful (but wrong-sized) result.  Move any pre-existing ones aside first, so
# a failed path yields a missing pickle and an honest "MISSING" in the summary.
STASH="$RUNDIR/_stale_pickles_from_previous_runs"
shopt -s nullglob
stale=( "$HERE"/experiment_*.pkl "$HERE"/experiment_*.png "$HERE"/path_comparison_*.png )
if (( ${#stale[@]} )); then
    mkdir -p "$STASH"
    echo "moving ${#stale[@]} pre-existing artefact(s) out of the way -> $(basename "$STASH")/"
    for f in "${stale[@]}"; do mv "$f" "$STASH/" 2>/dev/null; done
fi
shopt -u nullglob
echo

t0=$(date +%s)
declare -a STATUS=()
NOK=0

for inst in "${PATHS[@]}"; do
    echo "---------------------------------------------------------------------------"
    echo ">>> $inst   started $(date +%H:%M:%S)"
    echo "---------------------------------------------------------------------------"
    tp0=$(date +%s)
    ERRSIM_INSTRUMENT="$inst" ERRSIM_NPIX="$NPIX" \
        "$PYTHON" -u "$HERE/run_experiment.py" > "$RUNDIR/$inst.log" 2>&1
    rc=$?
    tp1=$(date +%s)
    mins=$(( (tp1 - tp0) / 60 ))
    if [[ $rc -eq 0 ]]; then
        echo "    OK    ($mins min)  -> $inst.log"
        STATUS+=("$inst OK ${mins}min")
        NOK=$((NOK+1))
    else
        echo "    FAILED (exit $rc, $mins min) -- see $RUNDIR/$inst.log"
        echo "    last traceback line:"
        grep -E "Error|Exception" "$RUNDIR/$inst.log" | tail -2 | sed 's/^/      /'
        STATUS+=("$inst FAILED(exit $rc) ${mins}min")
    fi
    # Keep this path's artefacts before the next path overwrites the shared names
    for f in "$HERE/experiment_$inst.pkl" "$HERE/experiment_${inst}"_*.png; do
        [[ -e "$f" ]] && cp -p "$f" "$RUNDIR/" 2>/dev/null
    done
done

echo
echo "---------------------------------------------------------------------------"
echo ">>> summary + comparison plot"
echo "---------------------------------------------------------------------------"
if (( NOK == 0 )); then
    echo "    SKIPPED: every path failed, so there is nothing to summarise."
    echo "    (Summarising anyway would report stale pickles from an earlier run"
    echo "     as if they were this run's results.)"
    {
      echo "CAMPAIGN FAILED -- all ${#PATHS[@]} paths errored; no results were produced."
      echo
      for s in "${STATUS[@]}"; do echo "  $s"; done
      echo
      echo "See the per-path logs in this directory for the traceback."
    } > "$RUNDIR/SUMMARY.txt"
else
    "$PYTHON" "$HERE/summarize_paths.py" "$RUNDIR/SUMMARY.txt" > "$RUNDIR/summarize.log" 2>&1 \
        && echo "    summary OK" || echo "    summary FAILED -- see $RUNDIR/summarize.log"
    for wi in 0 1 2 3; do
        "$PYTHON" "$HERE/plot_path_comparison.py" "$wi" >> "$RUNDIR/summarize.log" 2>&1
    done
    cp -p "$HERE"/path_comparison_*.png "$RUNDIR/" 2>/dev/null && echo "    plots OK"

fi

t1=$(date +%s)
echo
echo "==========================================================================="
echo " FINISHED $(date)   total $(( (t1 - t0) / 60 )) min"
for s in "${STATUS[@]}"; do echo "   $s"; done
echo
echo " Read in the morning:"
echo "   $RUNDIR/SUMMARY.txt"
echo "   $RUNDIR/path_comparison_549nm.png   (and 441/669/873)"
echo "==========================================================================="

# Print the summary so it is visible at the tail of run.log too
[[ -f "$RUNDIR/SUMMARY.txt" ]] && { echo; cat "$RUNDIR/SUMMARY.txt"; }
