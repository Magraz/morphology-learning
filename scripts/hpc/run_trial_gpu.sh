#!/bin/bash
BATCH_NAME=$1
EXPERIMENT_NAME=$2
ALGORITHM=$3
ENVIRONMENT=$4
TRIAL_ID=$5
RUNTIME=uv
CONDA_ENV_NAME=hypergraphs

SCRATCH=/nfs/stak/users/agrazvam/hpc-share/tmp
EXPERIMENT_SCRIPT=/nfs/stak/users/agrazvam/hpc-share/morphology-learning/train.py

sbatch <<EOT
#!/bin/bash
#SBATCH -J ${TRIAL_ID}_${EXPERIMENT_NAME}_${BATCH_NAME}
#SBATCH -A kt-lab
#SBATCH --partition=preempt
#SBATCH -o ./logs/${BATCH_NAME}_${EXPERIMENT_NAME}_${TRIAL_ID}.out
#SBATCH -e ./logs/${BATCH_NAME}_${EXPERIMENT_NAME}_${TRIAL_ID}.err
#SBATCH -c 8
#SBATCH --cpu-freq=high
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --nodelist=dgxh-[1-4],cn-w-[1-2],cn-t-1,cn-r-[1-6],cn-s-[1-5],cn-gpu[5-7],cn-gpu[10-12],optimus,sail-gpu0,dgx2-[1-2],dgx2-[4-5],cn-x-[1-2],cn-m-[1-2]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue

hostname
echo \$SLURM_JOBID
showjob \$SLURM_JOBID

export TMPDIR="$SCRATCH"

module purge
module load python/3.11

which python
python --version

echo "Checking CUDA availability..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found - CUDA may not be available"
    exit 1
fi

if ! nvidia-smi -L; then
    echo "ERROR: No GPU visible to this job allocation"
    exit 1
fi

if [[ "$RUNTIME" == "uv" ]]; then
    echo "Using uv runtime"

    uv run python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA build: {torch.version.cuda}'); print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); exit(0 if torch.cuda.is_available() else 1)" || {
        echo "ERROR: CUDA not available in the uv runtime"
        exit 1
    }

    uv run python3 "$EXPERIMENT_SCRIPT" \
        env="$BATCH_NAME" \
        model="$EXPERIMENT_NAME" \
        algorithm="$ALGORITHM" \
        trial_id="$TRIAL_ID" \
        checkpoint=true

elif [[ "$RUNTIME" == "conda" ]]; then

    module load gcc/14.3

    echo "Using conda runtime"

    if [[ -z "$CONDA_ENV_NAME" ]]; then
        echo "ERROR: conda runtime selected but no conda env name was provided"
        exit 1
    fi

    source ~/.bashrc
    conda activate "$CONDA_ENV_NAME" || {
        echo "ERROR: failed to activate conda env '$CONDA_ENV_NAME'"
        exit 1
    }

    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA build: {torch.version.cuda}'); print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); exit(0 if torch.cuda.is_available() else 1)" || {
        echo "ERROR: CUDA not available in the conda runtime"
        exit 1
    }

    python "$EXPERIMENT_SCRIPT" \
        --batch "$BATCH_NAME" \
        --name "$EXPERIMENT_NAME" \
        --algorithm "$ALGORITHM" \
        --environment "$ENVIRONMENT" \
        --trial_id "$TRIAL_ID" \
        --checkpoint

else
    echo "ERROR: unknown runtime '$RUNTIME'. Expected 'uv' or 'conda'."
    exit 1
fi
EOT
