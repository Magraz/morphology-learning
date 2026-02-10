#!/bin/bash
ENVIRONMENT=salp_navigate
BATCH_NAME=${ENVIRONMENT}_8a_ver_0
EXPERIMENT_NAMES=("mlp" "gcn" "gat" "graph_transformer" "gcn_full" "gat_full" "graph_transformer_full")
# EXPERIMENT_NAMES=("gat_full")
ALGORITHM=ppo
TRIAL_ID=1

for EXPERIMENT in "${EXPERIMENT_NAMES[@]}"; do
    bash run_trial_cpu_gpu.sh $BATCH_NAME $EXPERIMENT $ALGORITHM $ENVIRONMENT $TRIAL_ID
done
