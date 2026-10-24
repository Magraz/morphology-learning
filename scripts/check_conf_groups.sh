# Sourced helper shared by train_batches.sh (local) and
# scripts/hpc/experiment_batch.sh (SLURM).
#
#   check_conf_groups <conf_dir> <group/name>...
#
# Checks that every Hydra config group file <conf_dir>/<group/name>.yaml exists
# before anything is launched. A misspelled name would otherwise start jobs that
# fail at config composition (and, on the cluster, the watchdog resubmits every
# job that did not finish COMPLETED, indefinitely). Prints the missing files and
# returns 1 if any are absent.
check_conf_groups() {
    local conf_dir=$1
    shift
    local missing=()
    local entry
    for entry in "$@"; do
        [[ -f "${conf_dir}/${entry}.yaml" ]] || missing+=("${entry}.yaml")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "error: config files not found under ${conf_dir}:" >&2
        printf '  %s\n' "${missing[@]}" >&2
        return 1
    fi
}
