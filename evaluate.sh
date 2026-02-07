BATCH=${1}
ALGORITHM=${2}
ENVIRONMENT=${3}
TRIAL_ID=${4}
EXP_NAME=${5}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 ${SCRIPT_DIR}/run_trial.py --batch ${BATCH} --algorithm ${ALGORITHM} --environment ${ENVIRONMENT} --trial_id ${TRIAL_ID} --name ${EXP_NAME} --evaluate

echo "Finished evaluating ${BATCH}_${EXP_NAME}_${TRIAL_ID}"