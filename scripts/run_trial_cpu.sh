#!/bin/bash
BATCH_NAME=$1
EXPERIMENT_NAME=$2
ALGORITHM=$3
ENVIRONMENT=$4
TRIAL_ID=$5

SCRATCH=/nfs/stak/users/agrazvam/hpc-share/tmp
EXPERIMENT_SCRIPT=/nfs/stak/users/agrazvam/hpc-share/morphology-learning/src/run_trial.py

sbatch <<EOT
#!/bin/bash
#SBATCH -J ${TRIAL_ID}_${EXPERIMENT_NAME}_${BATCH_NAME}                      # name of job
#SBATCH -A kt-lab	                                                         # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH --partition=preempt                                                  # name of partition or queue
#SBATCH -o ./logs/${BATCH_NAME}_${EXPERIMENT_NAME}_${TRIAL_ID}.out           # name of output file for this submission script
#SBATCH -e ./logs/${BATCH_NAME}_${EXPERIMENT_NAME}_${TRIAL_ID}.err           # name of error file for this submission script
#SBATCH -c 4                                                                 # number of cores/threads per task (default 1)
#SBATCH --cpu-freq=high
#SBATCH --mem=24G                                                            # request gigabytes memory (per node, default depends on node)
#SBATCH --time=72:00:00                                                      # time needed for job (1 day)
#SBATCH --nodelist=cn-s-[1-5],cn-t-1,cn-v-[1-8],cn-u-[1-2]
#SBATCH --nodes=1
#SBATCH --requeue

# Gather basic information, can be useful for troubleshooting
hostname
echo $SLURM_JOBID
showjob $SLURM_JOBID

# Set temporary directory for the job, so that is doesn't use the default /tmp that gets filled up which causes issues during training
export TMPDIR="$SCRATCH"

DISPLAY=":0" python3 $EXPERIMENT_SCRIPT --batch $BATCH_NAME --name $EXPERIMENT_NAME --algorithm $ALGORITHM --environment $ENVIRONMENT --trial_id $TRIAL_ID --checkpoint
EOT