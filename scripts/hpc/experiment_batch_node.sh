#!/bin/bash
# Node version of experiment_batch.sh: trains every trial in BATCH_NAMES x
# EXPERIMENT_NAMES x [TRIAL_START, TRIAL_END] in ONE SLURM job on one dgxh
# (H100) node, instead of submitting one job per trial. The job
# (run_node_gpu.sh) runs train_batches.sh, which keeps TRIALS_PER_GPU trials
# running on each GPU. Use it for trials small enough that many share a GPU
# (e.g. rover). Node resources are set at the top of run_node_gpu.sh.
#
#   bash scripts/hpc/experiment_batch_node.sh
#
# It writes, under job_tracking/ (<id> = <algorithm>_node_<RUN_NAME>):
#   <id>_trials.tsv           the trial list the job reads (and a timestamped copy)
#   <id>_latest.csv           a one-row ledger (and a timestamped copy)
# then submits the job and a watchdog (run_watchdog.sh --run-script
# run_node_gpu.sh) that resubmits it every 10 minutes until it ends COMPLETED,
# i.e. until train_batches.sh exits 0 because every trial finished. A
# resubmitted job runs the whole list again; finished trials exit at once.
#
# Follow it from the login node with
#   bash scripts/train_batches_progress.sh      # the latest train_batches run
# and the job's own output in logs/node_<algorithm>_<RUN_NAME>.out.

# Names this run's job (0_node_<algorithm>_<RUN_NAME>), trial list and ledger.
# Letters, digits, '.', '_' and '-' only.
RUN_NAME=rover

BATCH_NAMES=(
"rover_2poi_c2" "rover_2poi_c8" "rover_4poi_c1234" "rover_6poi_c3_ring" "rover_6poi_c8-4-4-8-4-4"
)

EXPERIMENT_NAMES=(
"mappo_mlp" "rmappo"
)
ALGORITHM=rmappo_jax

TRIAL_START=0
TRIAL_END=9
SBATCH_SCRIPT="run_node_gpu.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)/conf"
TRACKING_DIR="${SCRIPT_DIR}/job_tracking"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
IDENTIFIER="${ALGORITHM}_node_${RUN_NAME}"
JOB_NAME="0_node_${ALGORITHM}_${RUN_NAME}"

source "${SCRIPT_DIR}/submit_and_record.sh"

if ! [[ "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: RUN_NAME may contain only letters, digits, '.', '_' and '-', got '${RUN_NAME}'" >&2
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

# A second job on the same list would train the same trials on another node,
# two processes writing one results directory, and a second watchdog would
# resubmit it too.
if command -v squeue > /dev/null; then
    ACTIVE="$(squeue -h -u "${USER}" -o "%j" | grep -F -x -e "${JOB_NAME}" -e "watchdog_${IDENTIFIER}")"
    if [[ -n "${ACTIVE}" ]]; then
        echo "error: already in the queue: $(echo ${ACTIVE}); scancel them or change RUN_NAME" >&2
        echo "nothing was submitted" >&2
        exit 1
    fi
fi

mkdir -p "${TRACKING_DIR}"
RUN_ID="${IDENTIFIER}_${RUN_TIMESTAMP}"

# The trial list, in the layout train_batches.sh reads (TRIALS_FILE).
TRIALS_RECORD="${TRACKING_DIR}/${RUN_ID}_trials.tsv"
TRIALS_FILE="${TRACKING_DIR}/${IDENTIFIER}_trials.tsv"
{
    printf 'batch\texperiment\ttrial\n'
    for BATCH_NAME in "${BATCH_NAMES[@]}"; do
        for EXPERIMENT in "${EXPERIMENT_NAMES[@]}"; do
            for TRIAL_ID in $(seq "${TRIAL_START}" "${TRIAL_END}"); do
                printf '%s\t%s\t%s\n' "${BATCH_NAME}" "${EXPERIMENT}" "${TRIAL_ID}"
            done
        done
    done
} > "${TRIALS_RECORD}"
N_TRIALS=$(( $(wc -l < "${TRIALS_RECORD}") - 1 ))
if (( N_TRIALS == 0 )); then
    echo "error: no trials (is a list empty, or TRIAL_START > TRIAL_END?)" >&2
    exit 1
fi
cp "${TRIALS_RECORD}" "${TRIALS_FILE}"
echo "${N_TRIALS} trials of ${ALGORITHM}: ${#BATCH_NAMES[@]} batches x" \
     "${#EXPERIMENT_NAMES[@]} experiments x trials ${TRIAL_START}-${TRIAL_END}"
echo "trial list: ${TRIALS_FILE}"

JOB_LEDGER="${TRACKING_DIR}/${RUN_ID}.csv"
LATEST_LEDGER="${TRACKING_DIR}/${IDENTIFIER}_latest.csv"
echo "${LEDGER_HEADER}" > "${JOB_LEDGER}"
if ! submit_and_record "${JOB_LEDGER}" "${RUN_ID}" "${SCRIPT_DIR}/${SBATCH_SCRIPT}" \
        "${RUN_NAME}" node "${ALGORITHM}" "" 0; then
    echo "the job was not submitted, so no watchdog was started" >&2
    exit 1
fi
cp "${JOB_LEDGER}" "${LATEST_LEDGER}"
echo "job ledger written to: ${JOB_LEDGER}"
echo "latest ledger updated: ${LATEST_LEDGER}"

echo "submitting the watchdog for identifier: ${IDENTIFIER}"
bash "${SCRIPT_DIR}/run_watchdog.sh" --run-script "${SCRIPT_DIR}/${SBATCH_SCRIPT}" "${IDENTIFIER}"
