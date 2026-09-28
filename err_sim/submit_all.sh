#!/usr/local/bin/bash
# =============================================================================
# submit_all.sh -- queue every (scene x retrieval scheme) campaign in one go.
#
#     ./err_sim/submit_all.sh            # submit everything
#     ./err_sim/submit_all.sh --dry-run  # print the sbatch lines, submit nothing
#     ./err_sim/submit_all.sh smoke_ocean dust_ocean    # only these scenes
#
# There is ONE batch script (slurm_task_array.sh) and it is fully environment
# driven, so a scene is just a set of variables.  Per-scene copies of that script
# would be 200 lines of duplicated comments drifting out of sync; instead every
# configuration lives in the SCENES table below and is passed via --export.
#
# SLURM decides the order.  Submit them all and walk away.
#
# Each campaign gets its OWN output directory.  Task filenames are built from
# (index, architecture) alone, so two campaigns sharing a directory would land on
# identical names -- and run_task.py deletes a stale file before starting, so the
# older campaign would be destroyed rather than merely mixed.
# =============================================================================
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
mkdir -p log

DRY=0
WANT=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY=1 ;;
        -*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *) WANT+=("$arg") ;;
    esac
done

# --- scene table --------------------------------------------------------------
# name | canonical case | settings YAML
#
# Every YAML here has been checked with err_sim/check_apriori.py: a retrieved
# parameter whose truth lies OUTSIDE its a-priori range is pinned at the bound and
# cannot respond to calibration, so the campaign would measure the settings file
# rather than the retrieval.  Re-run that check if you touch a YAML or add a scene.
#
# NOTE the case names are substring-matched and 'Land' is NOT a recognised token --
# only 'Desert' and 'Vegetation' select a land surface.  See canonicalCaseMap.
SCENES=(
    "smoke_ocean|smokeVariable|settings_BCK_POLAR_2modes_errsim_smoke_v2.yml"
    "smoke_desert|smokeVariableDesert|settings_BCK_POLAR_2modes_errsim_smoke_desert.yml"
    "dust_ocean|dustVariable|settings_BCK_POLAR_2modes_errsim_dust.yml"
    "dust_desert|dustVariableDesert|settings_BCK_POLAR_2modes_errsim_dust_desert.yml"
)

# --- retrieval schemes --------------------------------------------------------
# The I+DoLP scheme is the SAME settings file with one line changed
# (measurement_fitting.polarization), applied by sed at submit time so the scene
# YAMLs do not have to be maintained in duplicate.
SCHEMES=("iqu|relative_polarization_components" "dolp|degree_of_polarization")

# Path 2 (Monte Carlo) carries calibration bias -- the campaign.
# Path 3 injects exactly the noise the settings file assumes -- the ideal control,
# which needs only ONE task since it touches no calibration product.
ARRAY_ARCH=harperrsimmc
IDEAL_ARCH=harperrsimbck
ARRAY_RANGE="0-249%84"

submitted=0
for scene in "${SCENES[@]}"; do
    IFS='|' read -r sName sCase sYaml <<< "$scene"
    if [[ ${#WANT[@]} -gt 0 ]]; then
        keep=0
        for w in "${WANT[@]}"; do [[ "$w" == "$sName" ]] && keep=1; done
        [[ $keep -eq 1 ]] || continue
    fi
    [[ -f "ACCP_ArchitectureAndCanonicalCases/$sYaml" ]] || {
        echo "MISSING settings file for $sName: $sYaml" >&2; exit 1; }

    for scheme in "${SCHEMES[@]}"; do
        IFS='|' read -r pName pMode <<< "$scheme"

        # Derive the scheme's YAML once per (scene, scheme).  Identical content is
        # regenerated on every submit, which keeps it in step with the base file.
        yOut="$sYaml"
        if [[ "$pName" != "iqu" ]]; then
            yOut="${sYaml%.yml}_${pName}.yml"
            sed "s/polarization: relative_polarization_components/polarization: $pMode/" \
                "ACCP_ArchitectureAndCanonicalCases/$sYaml" \
                > "ACCP_ArchitectureAndCanonicalCases/$yOut"
        fi

        for kind in campaign ideal; do
            if [[ "$kind" == campaign ]]; then
                arch=$ARRAY_ARCH; arrayOpt="--array=$ARRAY_RANGE"
            else
                arch=$IDEAL_ARCH; arrayOpt="--array=0-0"
            fi
            tag="${sName}_${pName}_${kind}"
            out="$PWD/err_sim/tasks_${tag}"

            cmd=(sbatch $arrayOpt
                 -J "es_${tag}"
                 -o "log/${tag}.%A-%a.out" -e "log/${tag}.%A-%a.err"
                 --export="ALL,ERRSIM_CONCASE=$sCase,ERRSIM_BCK_YAML=$yOut,ERRSIM_INSTRUMENT=$arch,ERRSIM_TASK_OUT=$out"
                 err_sim/slurm_task_array.sh)

            if [[ $DRY -eq 1 ]]; then
                printf '%s\n\n' "${cmd[*]}"
            else
                mkdir -p "$out"
                "${cmd[@]}" || { echo "sbatch FAILED for $tag" >&2; exit 1; }
            fi
            submitted=$((submitted + 1))
        done
    done
done

echo
if [[ $DRY -eq 1 ]]; then
    echo "dry run: $submitted job(s) would be submitted"
else
    echo "submitted $submitted job(s); watch with  squeue -u \$USER"
fi
echo
echo "Collect afterwards, one per campaign directory:"
echo "    python err_sim/res_analysis/collect_tasks.py err_sim/tasks_<tag>"
echo "Compare any set of them in a single figure:"
echo "    python err_sim/res_analysis/box_whisk.py err_sim/tasks_A:labelA err_sim/tasks_B:labelB ..."
