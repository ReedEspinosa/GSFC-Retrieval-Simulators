#!/usr/local/bin/bash
# =============================================================================
# scenes.sh -- the campaign grid, in one place.
#
# Sourced by submit_all.sh (cluster campaigns) and run_ideals_local.sh (ideal
# controls run locally).  Keeping the table here means the two cannot drift apart;
# add a scene once and both pick it up.
# =============================================================================

# name | canonical case | settings YAML
#
# Every YAML has been verified with err_sim/check_apriori.py: a retrieved parameter
# whose truth lies OUTSIDE its a-priori range is pinned at the bound and cannot respond
# to calibration, so the campaign would measure the settings file rather than the
# retrieval.  Re-run that check if you touch a YAML or add a scene.
#
# Case names are substring-matched and 'Land' is NOT a recognised token -- only
# 'Desert' and 'Vegetation' select a land surface.  See canonicalCaseMap.
SCENES=(
    "smoke_ocean|smokeVariable|settings_BCK_POLAR_2modes_errsim_smoke_v2.yml"
    "smoke_desert|smokeVariableDesert|settings_BCK_POLAR_2modes_errsim_smoke_desert.yml"
    "dust_ocean|dustVariable|settings_BCK_POLAR_2modes_errsim_dust.yml"
    "dust_desert|dustVariableDesert|settings_BCK_POLAR_2modes_errsim_dust_desert.yml"
)

# The I+DoLP scheme is the SAME settings file with one line changed
# (measurement_fitting.polarization), applied by sed at run time so the scene YAMLs
# are not maintained in duplicate.
SCHEMES=("iqu|relative_polarization_components" "dolp|degree_of_polarization")

# Path 2 (Monte Carlo) carries calibration bias -- the campaign, 250 instruments.
# Path 3 injects exactly the noise the settings file assumes -- the ideal control,
# which needs only ONE task because it touches no calibration product.
CAMPAIGN_ARCH=harperrsimmc
IDEAL_ARCH=harperrsimbck


# Write the DoLP variant of a scene's settings file and echo the name to use.
# Regenerated every run so it stays in step with the base file.
scheme_yaml() {           # $1 = base yaml, $2 = scheme name, $3 = polarization value
    local base="$1" pName="$2" pMode="$3"
    if [[ "$pName" == "iqu" ]]; then
        echo "$base"
        return
    fi
    local out="${base%.yml}_${pName}.yml"
    sed "s/polarization: relative_polarization_components/polarization: $pMode/" \
        "ACCP_ArchitectureAndCanonicalCases/$base" \
        > "ACCP_ArchitectureAndCanonicalCases/$out"
    echo "$out"
}


# Filter SCENES against names given on the command line; empty list means all.
# Populates the global SELECTED array.
select_scenes() {
    SELECTED=()
    local scene sName rest keep w
    for scene in "${SCENES[@]}"; do
        sName="${scene%%|*}"
        if [[ $# -eq 0 ]]; then
            SELECTED+=("$scene")
            continue
        fi
        keep=0
        for w in "$@"; do [[ "$w" == "$sName" ]] && keep=1; done
        [[ $keep -eq 1 ]] && SELECTED+=("$scene")
    done
    if [[ ${#SELECTED[@]} -eq 0 ]]; then
        echo "no scenes matched: $*" >&2
        echo "known scenes: $(printf '%s ' "${SCENES[@]%%|*}")" >&2
        return 1
    fi
    return 0
}
