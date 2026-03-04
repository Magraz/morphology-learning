#!/bin/bash
ENVIRONMENT=multi_box_push
BATCH_NAME=${ENVIRONMENT}_test
EXPERIMENT_NAMES=("mlp_shared" "hgnn_shared" "hgnn_shared_entropy")
ALGORITHM=mappo
TRIAL_START=0
TRIAL_END=9
SBATCH_SCRIPT="run_trial_gpu.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKING_DIR="${SCRIPT_DIR}/job_tracking"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${BATCH_NAME}_${RUN_TIMESTAMP}"
JOB_LEDGER="${TRACKING_DIR}/${RUN_ID}.csv"
LATEST_LEDGER="${TRACKING_DIR}/${BATCH_NAME}_latest.csv"

mkdir -p "${TRACKING_DIR}"
echo "timestamp,run_id,batch_name,environment,algorithm,experiment,trial_id,job_name,job_id,submission_ok" > "${JOB_LEDGER}"

for EXPERIMENT in "${EXPERIMENT_NAMES[@]}"; do
    for TRIAL_ID in $(seq "${TRIAL_START}" "${TRIAL_END}"); do
        JOB_NAME="${TRIAL_ID}_${EXPERIMENT}_${BATCH_NAME}"
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
