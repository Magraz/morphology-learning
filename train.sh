#parallel bash train.sh salp_navigate_24a ppo salp_navigate frech ::: gcn gat graph_transformer gcn_full gat_full graph_transformer_full transformer_full mlp
#parallel bash train.sh salp_navigate_24a ppo salp_navigate ::: $(seq 0 11) ::: gcn gat graph_transformer gcn_full gat_full graph_transformer_full transformer_full mlp

BATCH=${1}
ALGORITHM=${2}
ENVIRONMENT=${3}   # vestigial: env identity now lives in conf/env/${BATCH}
TRIAL_ID=${4}
EXP_NAME=${5}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Hydra entry point. BATCH -> env group, EXP_NAME -> model group. The batch must
# be migrated into conf/ (conf/env/${BATCH}.yaml + conf/model/${EXP_NAME}.yaml).
uv run python3 ${SCRIPT_DIR}/train.py \
    env=${BATCH} \
    model=${EXP_NAME} \
    algorithm=${ALGORITHM} \
    trial_id=${TRIAL_ID} \
    checkpoint=true

echo "Finished trial ${BATCH}_${EXP_NAME}_${TRIAL_ID}"
