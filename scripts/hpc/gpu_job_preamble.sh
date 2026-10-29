# Sourced at the start of every GPU job script that run_trial_gpu.sh,
# run_trial_gpu_wildwood.sh and run_node_gpu.sh submit. It prints where the job
# runs, loads Python, and ends the job with exit 1 if no GPU is visible. The
# job script sets TMPDIR before sourcing it.

hostname
echo "${SLURM_JOBID}"
showjob "${SLURM_JOBID}"

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
