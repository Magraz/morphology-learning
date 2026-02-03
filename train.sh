#parallel bash train.sh salp_navigate_24a ppo salp_navigate frech ::: gcn gat graph_transformer gcn_full gat_full graph_transformer_full transformer_full mlp
#parallel bash train.sh salp_navigate_24a ppo salp_navigate :: $(seq 0 11) ::: gcn gat graph_transformer gcn_full gat_full graph_transformer_full transformer_full mlp

BATCH=${1}
ALGORITHM=${2}
ENVIRONMENT=${3}
TRIAL_ID=${4}
EXP_NAME=${5}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 ${SCRIPT_DIR}/run_trial.py --batch ${BATCH} --algorithm ${ALGORITHM} --environment ${ENVIRONMENT} --trial_id ${TRIAL_ID} --name ${EXP_NAME} --checkpoint

echo "Finished trial ${BATCH}_${EXP_NAME}_${TRIAL_ID}"