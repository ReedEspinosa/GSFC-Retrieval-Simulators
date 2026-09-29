#!/usr/local/bin/bash
# =============================================================================
# submit_all.sh -- run the cluster campaigns ONE AT A TIME.
#
#     nohup ./err_sim/submit_all.sh > submit_all.log 2>&1 &   # the usual way
#     ./err_sim/submit_all.sh --dry-run                       # print, submit nothing
#     ./err_sim/submit_all.sh dust_ocean                      # only this scene
#     ./err_sim/submit_all.sh --no-wait                       # fire everything at once
#
# WHY SERIAL.  Submitting the whole grid at once hits QOSMaxSubmitJobPerUserLimit --
# that limit counts ARRAY TASKS, and one campaign is 250 of them, so roughly two
# campaigns exhaust the quota.  More importantly it was simply slower in practice:
# one campaign at a time cleared the queue far quicker than four competing for
# priority against each other and burning the user's fairshare.
#
# So this script submits ONE campaign, waits for that job id to leave the queue
# (polling every --poll seconds, 5 min by default), then submits the next.  It runs
# for hours; start it under nohup and walk away.  Only the job it submitted is
# polled, so unrelated jobs of yours neither block it nor are waited on.
#
# IDEAL CONTROLS ARE NOT SUBMITTED HERE.  A Path 3 control is a single 526-pixel task
# taking under half an hour, which is not worth a cluster queue slot; run_ideals_local.sh
# does all of them locally overnight.
#
# Each campaign writes to its OWN directory.  Task filenames depend only on
# (index, architecture), and run_task.py deletes stale files before starting, so two
# campaigns sharing a directory would DESTROY the older one rather than merely mix it.
# =============================================================================
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
source err_sim/scenes.sh
mkdir -p log

DRY=0
WAIT=1
POLL=300
ARRAY_RANGE="0-249%84"
# 30 minutes, not 2 hours.  --time is PER ARRAY TASK, and a task takes 8-13 min
# (measured: smoke 32 CPU-s/px, dust 51).  A 2 h whole-node request is nearly
# unbackfillable -- SLURM will not slot it into a 40 min gap ahead of a higher
# priority reservation -- whereas 30 min fits almost anywhere, with a 2.3x margin
# over the slowest scene.
WALLTIME="00:30:00"
WANT=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY=1 ;;
        --no-wait)    WAIT=0 ;;
        --poll)       POLL="$2"; shift ;;
        --time)       WALLTIME="$2"; shift ;;
        --array)      ARRAY_RANGE="$2"; shift ;;
        -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)           echo "unknown flag: $1" >&2; exit 2 ;;
        *)            WANT+=("$1") ;;
    esac
    shift
done

select_scenes "${WANT[@]+"${WANT[@]}"}" || exit 2

# --- build the job list before submitting anything ---------------------------
JOBS=()
for scene in "${SELECTED[@]}"; do
    IFS='|' read -r sName sCase sYaml <<< "$scene"
    [[ -f "ACCP_ArchitectureAndCanonicalCases/$sYaml" ]] || {
        echo "MISSING settings file for $sName: $sYaml" >&2; exit 1; }
    for scheme in "${SCHEMES[@]}"; do
        IFS='|' read -r pName pMode <<< "$scheme"
        yOut=$(scheme_yaml "$sYaml" "$pName" "$pMode")
        JOBS+=("${sName}_${pName}_campaign|$sCase|$yOut")
    done
done

echo "$(date)  ${#JOBS[@]} campaign(s) to run, one at a time"
echo "  array $ARRAY_RANGE   walltime $WALLTIME per task   poll every ${POLL}s"
echo

# --- wait for one job id to leave the queue ----------------------------------
wait_for_job() {                       # $1 = job id
    local jid="$1" n elapsed=0
    while true; do
        # -h no header, -r expands array tasks; counts only THIS job
        n=$(squeue -j "$jid" -h -r 2>/dev/null | wc -l | tr -d ' ')
        [[ -z "$n" ]] && n=0
        if [[ "$n" -eq 0 ]]; then
            echo "    $(date +%H:%M:%S)  job $jid left the queue after ${elapsed}s"
            return 0
        fi
        printf '    %s  job %s: %s task(s) still queued/running (%dm elapsed)\n' \
               "$(date +%H:%M:%S)" "$jid" "$n" $((elapsed / 60))
        sleep "$POLL"
        elapsed=$((elapsed + POLL))
    done
}

n=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r tag sCase yOut <<< "$job"
    n=$((n + 1))
    out="$PWD/err_sim/tasks_${tag}"
    cmd=(sbatch --array="$ARRAY_RANGE" --time="$WALLTIME"
         -J "es_${tag}"
         -o "log/${tag}.%A-%a.out" -e "log/${tag}.%A-%a.err"
         --export="ALL,ERRSIM_CONCASE=$sCase,ERRSIM_BCK_YAML=$yOut,ERRSIM_INSTRUMENT=$CAMPAIGN_ARCH,ERRSIM_TASK_OUT=$out"
         err_sim/slurm_task_array.sh)

    echo "[$n/${#JOBS[@]}] $tag"
    if [[ $DRY -eq 1 ]]; then
        printf '    %s\n\n' "${cmd[*]}"
        continue
    fi

    mkdir -p "$out"
    # sbatch prints "Submitted batch job NNNN"; keep the id so only this job is polled
    submitOut=$("${cmd[@]}" 2>&1) || {
        echo "    sbatch FAILED: $submitOut" >&2
        echo "    stopping; rerun naming the remaining scenes once the queue frees" >&2
        exit 1; }
    jid=$(grep -oE '[0-9]+$' <<< "$submitOut" | tail -1)
    echo "    $(date +%H:%M:%S)  submitted job $jid -> $out"

    if [[ $WAIT -eq 1 && $n -lt ${#JOBS[@]} ]]; then
        wait_for_job "$jid"
    fi
    echo
done

echo "$(date)  all ${#JOBS[@]} campaign(s) submitted"
[[ $WAIT -eq 1 ]] && echo "(the last one may still be running; check squeue)"
echo
echo "Ideal controls run locally, not here:  ./err_sim/run_ideals_local.sh"
echo "Collect:  python err_sim/res_analysis/collect_tasks.py err_sim/tasks_<tag>"
echo "Compare:  python err_sim/res_analysis/box_whisk.py err_sim/tasks_A:labelA ..."
