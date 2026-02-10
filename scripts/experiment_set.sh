#!/bin/bash
ENVIRONMENT=salp_navigate
BATCH_NAME=${ENVIRONMENT}_16a_eval
# EXPERIMENT_NAMES=("gcn" "gat" "graph_transformer" "gcn_full" "gat_full" "graph_transformer_full"  "transformer_full")
# EXPERIMENT_NAMES=("gcn" "graph_transformer" "graph_transformer_full")
EXPERIMENT_NAMES=("gcn" "gat" "graph_transformer" "gcn_full" "gat_full" "graph_transformer_full")
# EXPERIMENT_NAMES=("mlp")
ALGORITHM=ppo

for EXPERIMENT in "${EXPERIMENT_NAMES[@]}"; do
    for TRIAL_ID in $(seq 0 4); do
        bash run_trial_cpu_evaluate.sh $BATCH_NAME $EXPERIMENT $ALGORITHM $ENVIRONMENT $TRIAL_ID
    done
done
