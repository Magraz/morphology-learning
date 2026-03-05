#!/bin/bash
BATCH_NAME=$1
EXPERIMENT_NAME=$2
ALGORITHM=$3
ENVIRONMENT=$4
TRIAL_ID=$5

SCRATCH=/nfs/stak/users/agrazvam/hpc-share/tmp
EXPERIMENT_SCRIPT=/nfs/stak/users/agrazvam/hpc-share/morphology-learning/run_trial.py

sbatch <<EOT
#!/bin/bash
#SBATCH -J ${TRIAL_ID}_${EXPERIMENT_NAME}_${BATCH_NAME}                      # name of job
#SBATCH -A kt-lab	                                                         # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH --partition=preempt                                                  # name of partition or queue
#SBATCH -o ./logs/${BATCH_NAME}_${EXPERIMENT_NAME}_${TRIAL_ID}.out           # name of output file for this submission script
#SBATCH -e ./logs/${BATCH_NAME}_${EXPERIMENT_NAME}_${TRIAL_ID}.err           # name of error file for this submission script
#SBATCH -c 4                                                                 # number of cores/threads per task (default 1)
#SBATCH --cpu-freq=high
#SBATCH --mem=12G                                                            # request gigabytes memory (per node, default depends on node)
#SBATCH --time=72:00:00                                                      # time needed for job (1 day)
#SBATCH --nodelist=dgxh-[1-4],cn-w-1,cn-t-1,cn-r-[1-6],cn-s-[1-2],cn-s-[4-5],cn-gpu[10-12],optimus,sail-gpu0,dgx2-[1-5],ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                                                         # number of GPUs to request (default 0)
#SBATCH --requeue

# gather basic information, can be useful for troubleshooting
hostname
echo \$SLURM_JOBID
showjob \$SLURM_JOBID

# Set temporary directory for the job, so that is doesn't use the default /tmp that gets filled up which causes issues during training
export TMPDIR="$SCRATCH"

module purge
module load python/3.11   # same Python used to create venv

which python
python --version
echo "VIRTUAL_ENV=\$VIRTUAL_ENV"

# Check if CUDA/GPU is available
echo "Checking CUDA availability..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found - CUDA may not be available"
    exit 1
fi

if ! nvidia-smi -L; then
    echo "ERROR: No GPU visible to this job allocation"
    exit 1
fi

# Verify CUDA is accessible in the same uv runtime used for training.
uv run python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA build: {torch.version.cuda}'); print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA not available in the uv runtime"
    exit 1
}

DISPLAY=":0" uv run python3 $EXPERIMENT_SCRIPT --batch $BATCH_NAME --name $EXPERIMENT_NAME --algorithm $ALGORITHM --environment $ENVIRONMENT --trial_id $TRIAL_ID --checkpoint
EOT
