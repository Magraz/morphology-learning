# Sourced helper shared by experiment_batch.sh (one job per trial) and
# experiment_batch_node.sh (one job for all trials).
#
#   echo "${LEDGER_HEADER}" > <ledger>
#   submit_and_record <ledger> <run id> <run script> \
#       <batch> <experiment> <algorithm> <environment> <trial id>
#
# Submits one job by calling <run script> with the last five arguments. That is
# the signature every scripts/hpc/run_*.sh script shares, and the one
# watchdog_jobs_v2.sh resubmits a ledger row with. Prints the outcome and
# appends one row to <ledger>. Returns the run script's exit status.

LEDGER_HEADER="timestamp,run_id,batch_name,environment,algorithm,experiment,trial_id,job_name,job_id,submission_ok"

submit_and_record() {
    local ledger=$1 run_id=$2 run_script=$3
    local batch=$4 experiment=$5 algorithm=$6 environment=$7 trial=$8
    # The name every run script passes to `sbatch -J`; the watchdog matches it.
    local job_name="${trial}_${experiment}_${algorithm}_${batch}"
    local timestamp output status job_id ok

    timestamp="$(date +%Y-%m-%dT%H:%M:%S%z)"
    output="$(bash "${run_script}" "${batch}" "${experiment}" "${algorithm}" "${environment}" "${trial}" 2>&1)"
    status=$?

    job_id="$(printf '%s\n' "${output}" | sed -n -E 's/.*Submitted batch job ([0-9]+).*/\1/p' | tail -n 1)"
    if [[ -z "${job_id}" ]]; then
        # Handles sbatch --parsable output if used later.
        job_id="$(printf '%s\n' "${output}" | sed -n -E 's/^([0-9]+)(;.*)?$/\1/p' | tail -n 1)"
    fi
    [[ -z "${job_id}" ]] && job_id="UNKNOWN"

    if [[ ${status} -eq 0 ]]; then
        ok=1
        echo "submitted ${job_name} -> ${job_id}"
    else
        ok=0
        echo "failed ${job_name} -> ${job_id}" >&2
        echo "${output}" >&2
    fi

    echo "${timestamp},${run_id},${batch},${environment},${algorithm},${experiment},${trial},${job_name},${job_id},${ok}" >> "${ledger}"
    return "${status}"
}
