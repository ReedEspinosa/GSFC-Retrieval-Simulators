#!/usr/local/bin/bash
# =============================================================================
# run_ideals_local.sh -- run every Path 3 "ideal" control locally, back to back.
#
#     nohup ./err_sim/run_ideals_local.sh > ideals.log 2>&1 &
#     ./err_sim/run_ideals_local.sh --dry-run
#     ./err_sim/run_ideals_local.sh dust_ocean          # one scene only
#
# WHY LOCAL.  A Path 3 control is a SINGLE 526-pixel task -- it touches no calibration
# product, so there is no instrument-to-instrument variation to average over and one
# task is the whole answer.  Measured locally on 12 cores: ~27 min for smoke.  That is
# not worth a cluster queue slot, and on the cluster it would compete with the
# campaigns for the same submit quota.
#
# Expect roughly 4-5 hours for all eight (4 scenes x 2 schemes), scaling with the
# per-scene cost measured at 48 pixels:
#     smoke_ocean 32.4 CPU-s/px   smoke_desert 31.4   dust_ocean 51.1   dust_desert 39.2
# Dust is slower because its coarse mode is spheroids, whose kernels cost ~1.5x per
# iteration; that is a per-iteration cost, not extra iterations.
#
# Runs under caffeinate so an idle Mac does not suspend it mid-run.  caffeinate does
# NOT prevent lid-close sleep -- leave the lid open.
# =============================================================================
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
source err_sim/scenes.sh

DRY=0
NPIX="${ERRSIM_NPIX:-}"          # empty -> every valid geometry pixel (526)
MAXCPU="${ERRSIM_MAXCPU:-12}"
WANT=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY=1 ;;
        --npix)       NPIX="$2"; shift ;;
        --maxcpu)     MAXCPU="$2"; shift ;;
        -h|--help)    sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)           echo "unknown flag: $1" >&2; exit 2 ;;
        *)            WANT+=("$1") ;;
    esac
    shift
done

select_scenes "${WANT[@]+"${WANT[@]}"}" || exit 2

# Descriptor headroom: graspDB holds one open pipe per forward pixel, and the default
# 256 dies partway through a 526-pixel run.
ulimit -n 8192 2>/dev/null || echo "WARNING: could not raise ulimit -n"

JOBS=()
for scene in "${SELECTED[@]}"; do
    IFS='|' read -r sName sCase sYaml <<< "$scene"
    [[ -f "ACCP_ArchitectureAndCanonicalCases/$sYaml" ]] || {
        echo "MISSING settings file for $sName: $sYaml" >&2; exit 1; }
    for scheme in "${SCHEMES[@]}"; do
        IFS='|' read -r pName pMode <<< "$scheme"
        yOut=$(scheme_yaml "$sYaml" "$pName" "$pMode")
        JOBS+=("${sName}_${pName}_ideal|$sCase|$yOut")
    done
done

echo "$(date)  ${#JOBS[@]} ideal control(s), ${MAXCPU} cores, pixels: ${NPIX:-all (526)}"
echo

runOne() {                       # $1 = tag, $2 = case, $3 = yaml
    local tag="$1" sCase="$2" yOut="$3"
    local out="$PWD/err_sim/tasks_${tag}"
    echo "  case    $sCase"
    echo "  yaml    $yOut"
    echo "  output  $out"
    [[ $DRY -eq 1 ]] && { echo "  (dry run, not executed)"; return 0; }
    mkdir -p "$out"
    local t0=$SECONDS
    # caffeinate wraps ONLY the python call: keeping it here rather than around an
    # exported shell function avoids losing the environment in a nested bash -c.
    [[ -n "$NPIX" ]] && export ERRSIM_NPIX="$NPIX"
    ERRSIM_INSTRUMENT="$IDEAL_ARCH" \
    ERRSIM_BCK_YAML="$yOut" \
    ERRSIM_CONCASE="$sCase" \
    ERRSIM_TAU_FACTOR=randLogNrm0.3 \
    ERRSIM_MAXCPU="$MAXCPU" \
    ERRSIM_TASK_OUT="$out" \
    caffeinate -i python -u err_sim/run_task.py 0 2>&1 \
        | grep -E "pixels processed|task 0 done|Traceback|Error|error:" \
        | sed 's/^/    /'
    local rc=${PIPESTATUS[0]}
    printf '  %s  finished in %d min (exit %d)\n' "$(date +%H:%M:%S)" \
           $(((SECONDS - t0) / 60)) "$rc"
    return $rc
}

n=0
failed=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r tag sCase yOut <<< "$job"
    n=$((n + 1))
    echo "[$n/${#JOBS[@]}] $tag   $(date +%H:%M:%S)"
    if runOne "$tag" "$sCase" "$yOut"; then
        :
    else
        echo "  FAILED -- continuing with the rest" >&2
        failed+=("$tag")
    fi
    echo
done

echo "$(date)  done"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "Compare against the campaigns, e.g.:"
echo "    python err_sim/res_analysis/box_whisk.py \\"
echo "        err_sim/tasks_smoke_ocean_iqu_ideal:'ideal I,Q,U' \\"
echo "        err_sim/tasks_smoke_ocean_iqu_campaign:'I,Q,U' \\"
echo "        err_sim/tasks_smoke_ocean_dolp_ideal:'ideal I,DoLP' \\"
echo "        err_sim/tasks_smoke_ocean_dolp_campaign:'I,DoLP'"
