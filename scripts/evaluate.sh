BATCH=${1}
ALGORITHM=${2}
ENVIRONMENT=${3}   # vestigial: env identity now lives in conf/env/${BATCH}
TRIAL_ID=${4}
EXP_NAME=${5}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Hydra entry point. BATCH -> env group, EXP_NAME -> model group. The batch must
# be migrated into conf/ (conf/env/${BATCH}.yaml + conf/model/${EXP_NAME}.yaml).
python3 ${SCRIPT_DIR}/../train.py \
    env=${BATCH} \
    model=${EXP_NAME} \
    algorithm=${ALGORITHM} \
    trial_id=${TRIAL_ID} \
    evaluate=true

echo "Finished evaluating ${BATCH}_${EXP_NAME}_${TRIAL_ID}"
