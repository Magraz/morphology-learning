#!/bin/bash
ENVIRONMENT=multi_box_push
BATCH_NAME=${ENVIRONMENT}_test
EXPERIMENT_NAMES=("mlp_shared")
ALGORITHM=mappo

for EXPERIMENT in "${EXPERIMENT_NAMES[@]}"; do
    for TRIAL_ID in $(seq 0 4); do
        bash run_trial_gpu.sh $BATCH_NAME $EXPERIMENT $ALGORITHM $ENVIRONMENT $TRIAL_ID
    done
done
