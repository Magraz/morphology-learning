#!/bin/bash
# Multi-batch version of experiment_set.sh: submits every experiment in
# EXPERIMENT_NAMES x every trial in [TRIAL_START, TRIAL_END] for EACH batch in
# BATCH_NAMES. A one-element BATCH_NAMES behaves exactly like experiment_set.sh.
#
# Each batch gets its own ledger, job_tracking/<algorithm>_<batch>_latest.csv,
# because watchdog_jobs_v2.sh looks ledgers up by that identifier. A single
# watchdog job is then submitted that cycles over all of the batches.

BATCH_NAMES=(
# "mjx_1a_3o_111_1024_gs" "mjx_1a_3o_111_1024_gs_sparse" "mjx_1a_3o_111_1024_gs_scale2" "mjx_1a_3o_111_1024_gs_sparse_scale2"
"mjx_2a_4o_1122_1024_gs" "mjx_2a_4o_1122_1024_gs_sparse" "mjx_2a_4o_1122_1024_gs_scale2" "mjx_2a_4o_1122_1024_gs_sparse_scale2"
"mjx_6a_4o_1024_gs" "mjx_6a_4o_1024_gs_sparse" "mjx_6a_4o_1024_gs_scale2" "mjx_6a_4o_1024_gs_sparse_scale2"
)

EXPERIMENT_NAMES=(
# "simplified_feudal"
# "simplified_feudal_tanh_relative_input" "simplified_feudal_tanh_local_input"
# "simplified_feudal_tanh_relative_input_cf" "simplified_feudal_tanh_local_input_cf"
# "simplified_feudal_tanh_relative_input_dpp" "simplified_feudal_tanh_local_input_dpp"
"simplified_feudal_tanh_local_input_interventions" "simplified_feudal_tanh_relative_input_interventions"
"simplified_feudal_tanh_local_input_interventions_k2" "simplified_feudal_tanh_relative_input_interventions_k2"
"simplified_feudal_tanh_relative_input_interventions_redecide_all" "simplified_feudal_tanh_relative_input_interventions_redecide"
)
ALGORITHM=simplified_feudal_mappo_jax

# EXPERIMENT_NAMES=("mlp")
# ALGORITHM=mappo_jax

TRIAL_START=0
TRIAL_END=4
SBATCH_SCRIPT="run_trial_gpu.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)/conf"
TRACKING_DIR="${SCRIPT_DIR}/job_tracking"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ${#BATCH_NAMES[@]} -eq 0 ]]; then
    echo "error: BATCH_NAMES is empty" >&2
    exit 1
fi

# Check every Hydra config group exists before submitting anything.
source "${SCRIPT_DIR}/../check_conf_groups.sh"
if ! check_conf_groups "${CONF_DIR}" \
        "algorithm/${ALGORITHM}" \
        "${BATCH_NAMES[@]/#/env/}" \
        "${EXPERIMENT_NAMES[@]/#/model/}"; then
    echo "nothing was submitted" >&2
    exit 1
fi

mkdir -p "${TRACKING_DIR}"

BATCH_IDENTIFIERS=()
for BATCH_NAME in "${BATCH_NAMES[@]}"; do
    BATCH_IDENTIFIER="${ALGORITHM}_${BATCH_NAME}"
    BATCH_IDENTIFIERS+=("${BATCH_IDENTIFIER}")
    RUN_ID="${BATCH_IDENTIFIER}_${RUN_TIMESTAMP}"
    JOB_LEDGER="${TRACKING_DIR}/${RUN_ID}.csv"
    LATEST_LEDGER="${TRACKING_DIR}/${BATCH_IDENTIFIER}_latest.csv"

    echo "=== batch ${BATCH_NAME} ==="
    echo "timestamp,run_id,batch_name,environment,algorithm,experiment,trial_id,job_name,job_id,submission_ok" > "${JOB_LEDGER}"

    for EXPERIMENT in "${EXPERIMENT_NAMES[@]}"; do
        for TRIAL_ID in $(seq "${TRIAL_START}" "${TRIAL_END}"); do
            JOB_NAME="${TRIAL_ID}_${EXPERIMENT}_${BATCH_IDENTIFIER}"
            TIMESTAMP="$(date +%Y-%m-%dT%H:%M:%S%z)"

            SUBMISSION_OUTPUT="$(
                bash "${SCRIPT_DIR}/${SBATCH_SCRIPT}" \
                    "${BATCH_NAME}" \
                    "${EXPERIMENT}" \
                    "${ALGORITHM}" \
                    "${ENVIRONMENT}" \
                    "${TRIAL_ID}" 2>&1
            )"
            SUBMIT_EXIT=$?

            JOB_ID="$(printf '%s\n' "${SUBMISSION_OUTPUT}" | sed -n -E 's/.*Submitted batch job ([0-9]+).*/\1/p' | tail -n 1)"
            if [[ -z "${JOB_ID}" ]]; then
                # Handles sbatch --parsable output if used later.
                JOB_ID="$(printf '%s\n' "${SUBMISSION_OUTPUT}" | sed -n -E 's/^([0-9]+)(;.*)?$/\1/p' | tail -n 1)"
            fi

            if [[ -z "${JOB_ID}" ]]; then
                JOB_ID="UNKNOWN"
            fi

            if [[ ${SUBMIT_EXIT} -eq 0 ]]; then
                SUBMISSION_OK=1
                echo "submitted ${JOB_NAME} -> ${JOB_ID}"
            else
                SUBMISSION_OK=0
                echo "failed ${JOB_NAME} -> ${JOB_ID}" >&2
                echo "${SUBMISSION_OUTPUT}" >&2
            fi

            echo "${TIMESTAMP},${RUN_ID},${BATCH_NAME},${ENVIRONMENT},${ALGORITHM},${EXPERIMENT},${TRIAL_ID},${JOB_NAME},${JOB_ID},${SUBMISSION_OK}" >> "${JOB_LEDGER}"
        done
    done

    cp "${JOB_LEDGER}" "${LATEST_LEDGER}"
    echo "job ledger written to: ${JOB_LEDGER}"
    echo "latest ledger updated: ${LATEST_LEDGER}"
done

echo "submitting one watchdog for identifiers: ${BATCH_IDENTIFIERS[*]}"
bash "${SCRIPT_DIR}/run_watchdog.sh" "${BATCH_IDENTIFIERS[@]}"
