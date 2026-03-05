#!/bin/bash

set -u
set -o pipefail

BATCH_NAMES=("$@")
if [[ ${#BATCH_NAMES[@]} -eq 0 ]]; then
    echo "usage: $(basename "$0") <batch_name> [batch_name ...]" >&2
    exit 1
fi

SCRATCH=/nfs/stak/users/agrazvam/hpc-share/tmp
EXPERIMENT_SCRIPT=/nfs/stak/users/agrazvam/hpc-share/morphology-learning/scripts/hpc/watchdog_loop.py
LOG_STEM="${BATCH_NAMES[0]}"
BATCH_NAMES_ESCAPED="$(printf '%q ' "${BATCH_NAMES[@]}")"

sbatch <<EOT
#!/bin/bash
#SBATCH -J watchdog_${LOG_STEM}                      # name of job
#SBATCH -A kt-lab	                                                         # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH --partition=share                                                  # name of partition or queue
#SBATCH -o ./logs/watchdog_${LOG_STEM}.out                                 # name of output file for this submission script
#SBATCH -e ./logs/watchdog_${LOG_STEM}.err                                 # name of error file for this submission script
#SBATCH -c 1                                                                 # number of cores/threads per task (default 1)
#SBATCH --cpu-freq=high
#SBATCH --mem=4G                                                            # request gigabytes memory (per node, default depends on node)
#SBATCH --time=72:00:00                                                      # time needed for job (1 day)
#SBATCH --nodes=1
#SBATCH --requeue

# gather basic information, can be useful for troubleshooting
hostname
echo \$SLURM_JOBID
showjob \$SLURM_JOBID

# Set temporary directory for the job, so that is doesn't use the default /tmp that gets filled up which causes issues during training
export TMPDIR="$SCRATCH"

module purge
module load python/3.11
module load slurm   # or your cluster's slurm client module name

which squeue
which sacct

which python
python --version

DISPLAY=":0" uv run python3 $EXPERIMENT_SCRIPT $BATCH_NAMES_ESCAPED
EOT
