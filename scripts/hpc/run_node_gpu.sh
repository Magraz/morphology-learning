#!/bin/bash
# Submits ONE SLURM job that trains a whole list of trials on one dgxh (H100)
# node by running train_batches.sh there: TRIALS_PER_GPU trials share each of
# the job's GPUS_PER_JOB GPUs, and the rest wait for a free slot. The node
# counterpart of run_trial_gpu.sh (one job per trial). experiment_batch_node.sh
# calls it; so does watchdog_jobs_v2.sh when it resubmits.
#
#   bash run_node_gpu.sh <run name> node <algorithm> <environment> 0
#
# The arguments are the columns of a ledger row, the same five every run_*.sh
# script takes, so the watchdog can resubmit the job unchanged. <run name>
# picks the trial list job_tracking/<algorithm>_node_<run name>_trials.tsv that
# experiment_batch_node.sh writes. "node" and 0 only fill the ledger's
# experiment and trial columns, giving the job name 0_node_<algorithm>_<run
# name>. <environment> is unused, as in the other run scripts.
#
# Every trial runs with checkpoint=true (train.sh). After a preemption SLURM
# requeues the job (--requeue); after a timeout, or when a trial failed, the
# job does not end COMPLETED and the watchdog resubmits it. Either way
# train_batches.sh runs the whole list again: finished trials load their
# checkpoint and exit (under a minute each for rover), the others resume.
#
# Sizing, measured 2026-10-28 on a 16-rover rmappo trial (rover_2poi_c8, the
# largest rover config), 5 trials sharing an RTX 4080 SUPER: 2.6 GB of GPU
# memory, 2.4 GB of RAM and about one CPU core per trial, 1.2 s per update
# (1e7 steps = 1563 updates). An 80 GB H100 holds about 30 such trials by
# memory; their speed when 16 share one H100 is not measured.

RUN_NAME=$1
EXPERIMENT_NAME=$2
ALGORITHM=$3
ENVIRONMENT=$4
TRIAL_ID=$5

# Resources of the one job.
GPUS_PER_JOB=1
TRIALS_PER_GPU=16          # train_batches.sh N_PARALLEL
CPUS_PER_JOB=20            # one per trial, plus a few for compiling
MEM_PER_JOB=96G            # going over it kills the job, i.e. every trial
TIME_PER_JOB=24:00:00
NODELIST="dgxh-[1-4]"
PROGRESS_INTERVAL=600      # seconds between progress reports in the .out file

SCRATCH=/nfs/stak/users/agrazvam/hpc-share/tmp
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRIALS_FILE="${SCRIPT_DIR}/job_tracking/${ALGORITHM}_node_${RUN_NAME}_trials.tsv"
JOB_NAME="${TRIAL_ID}_${EXPERIMENT_NAME}_${ALGORITHM}_${RUN_NAME}"

if [[ $# -ne 5 ]]; then
    echo "usage: $(basename "$0") <run name> node <algorithm> <environment> <trial id>" >&2
    exit 1
fi
# A per-trial ledger resubmitted through this script would start a node job
# named after one trial.
if [[ "${EXPERIMENT_NAME}" != node ]]; then
    echo "error: the experiment column must be 'node', got '${EXPERIMENT_NAME}';" \
         "per-trial jobs are submitted with run_trial_gpu.sh" >&2
    exit 1
fi
if [[ ! -f "${TRIALS_FILE}" ]]; then
    echo "error: trial list not found: ${TRIALS_FILE}" >&2
    exit 1
fi

sbatch <<EOT
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -A kt-lab
#SBATCH --partition=preempt
#SBATCH -o ./logs/node_${ALGORITHM}_${RUN_NAME}.out
#SBATCH -e ./logs/node_${ALGORITHM}_${RUN_NAME}.err
#SBATCH -c ${CPUS_PER_JOB}
#SBATCH --cpu-freq=high
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --time=${TIME_PER_JOB}
#SBATCH --nodelist=${NODELIST}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPUS_PER_JOB}
#SBATCH --requeue

export TMPDIR="${SCRATCH}"

# Job info, Python module, and a check that a GPU is visible.
source "${REPO_DIR}/scripts/hpc/gpu_job_preamble.sh"

# Sync the environment and check JAX sees a GPU once for the whole job. The
# trials' own 'uv run' calls then skip the sync (UV_NO_SYNC) instead of all
# checking the shared .venv on NFS at once.
cd "${REPO_DIR}"
uv run python3 -c "import jax; print('JAX devices:', jax.devices()); raise SystemExit(0 if jax.default_backend() == 'gpu' else 1)" || {
    echo "ERROR: JAX does not see a GPU"
    exit 1
}
export UV_NO_SYNC=1

# SLURM lists the job's GPUs in CUDA_VISIBLE_DEVICES; train_batches.sh gives
# each trial one of them.
GPUS="\${CUDA_VISIBLE_DEVICES//,/ }"
if [[ -z "\${GPUS}" ]]; then
    echo "warning: CUDA_VISIBLE_DEVICES is not set; every trial uses the default GPU"
fi

TRIALS_FILE="${TRIALS_FILE}" ALGORITHM="${ALGORITHM}" GPUS="\${GPUS}" \\
    N_PARALLEL=${TRIALS_PER_GPU} PROGRESS_INTERVAL=${PROGRESS_INTERVAL} \\
    bash "${REPO_DIR}/train_batches.sh"
EOT
