#!/bin/bash
# Local counterpart of scripts/hpc/experiment_batch.sh: runs every trial in
# BATCH_NAMES x EXPERIMENT_NAMES x [TRIAL_START, TRIAL_END] on this machine, one
# train.sh call per trial.
#
#   bash train_batches.sh                  # one trial at a time
#   N_PARALLEL=2 bash train_batches.sh     # two trials at once, sharing the GPU
#   N_PARALLEL=2 CPU_SLOTS="4 6" bash train_batches.sh   # and two on the CPU
#   DRY_RUN=1 bash train_batches.sh        # list the trials, run nothing
#   PROGRESS_INTERVAL=60 bash train_batches.sh   # progress.log every minute (0: off)
#   GPUS="0 1" N_PARALLEL=8 bash train_batches.sh  # 8 trials on each of GPUs 0, 1
#   TRIALS_FILE=trials.tsv ALGORITHM=rmappo_jax bash train_batches.sh
#
# GPUS spreads the GPU slots over several GPUs: N_PARALLEL slots per listed GPU,
# each trial run with CUDA_VISIBLE_DEVICES set to its slot's GPU. The entries
# are GPU ids as CUDA_VISIBLE_DEVICES names them. Without GPUS every GPU slot
# uses the default GPU, as before. scripts/hpc/run_node_gpu.sh sets GPUS from
# its SLURM job's own CUDA_VISIBLE_DEVICES.
#
# TRIALS_FILE replaces the BATCH_NAMES / EXPERIMENT_NAMES / TRIAL_START-END
# lists below with an explicit list: a tab-separated file whose first line is
# the header "batch<TAB>experiment<TAB>trial", then one trial per line (the layout of
# the trials.tsv this script writes into its log directory). ALGORITHM must
# then be set in the environment as well. scripts/hpc/run_node_gpu.sh uses it.
#
# train.sh passes checkpoint=true, so re-running this script resumes unfinished
# trials. A finished trial loads its checkpoint, runs no updates and exits 0 (it
# still pays the JAX compile first).
#
# N_PARALLEL changes only how many trials run side by side. The MJX env groups
# set a literal n_envs, so it never changes a trial's batch or its results.
#
# CPU_SLOTS adds trials that run on the CPU. Each space-separated entry is one
# slot: the logical CPUs its trial may use, comma-separated, numbered as
# `taskset -c` and `lscpu -e=CPU,CORE` number them. "4,6 8,10" is two slots of
# two CPUs each. A CPU trial runs under JAX_PLATFORMS=cpu and is pinned to its
# CPUs with taskset. Only the JAX algorithms change backend; the torch ones run
# on `device` from conf/config.yaml (cpu) in every slot. A free GPU slot is
# used before a free CPU slot, and N_PARALLEL=0 runs on the CPU slots only.
# Measured on mjx_3a_1o_1024_gs (2026-10-24): a CPU trial on one performance
# core took 4.6 s per update for simplified_feudal_tanh_relative_input and
# 8.7 s for its _interventions_adaptive_skip arm, against 6.9 and 6.4 s for a
# GPU trial sharing the GPU with one other. CPU trials did not slow the GPU
# trials. Efficiency cores were about 2x slower than performance cores. The
# CPU backend changes only floating-point rounding, not the batch.
#
# Once a run starts the script prints nothing on the terminal; only errors that
# stop it before any trial runs, and the DRY_RUN listing, are printed. Everything
# goes to the run's log directory, logs/train_batches/<timestamp>/:
#   <batch>__<experiment>__<trial>.log   each trial's output
#   train_batches.log   this script's messages (trial started / finished,
#                       warnings, the final summary table)
#   progress.log        a progress snapshot every PROGRESS_INTERVAL seconds
#   summary.tsv         each finished trial's status and slot
# A failed trial does not stop the others, and the script exits 1 if any trial
# failed. A trial that is already being trained by another process is skipped.
# Ctrl-C stops every running trial.
#
# Every PROGRESS_INTERVAL seconds (default 300) progress.log gets how far each
# running trial is, e.g.
#
#   [16:30:05] progress of 25 trials: 4 running, 10 done, 11 queued
#     21.4% [####----------------] 2h05m left     gpu      <batch> / <experiment> / <trial>
#
# The percentage is the trainer's own "Steps: <done>/<total>" line, and the time
# left comes from the speed over the trial's last 100 such lines. To see every
# trial (done, failed and queued too) at any time, run
#   bash scripts/train_batches_progress.sh [logs/train_batches/<timestamp>]
# which reads the run's log directory (default: the latest run).

# N_PARALLEL=2 CPU_SLOTS="4 6 8" bash train_batches.sh

# With TRIALS_FILE, ALGORITHM comes from the environment. Keep it before the
# assignment below overwrites it.
ALGORITHM_FROM_ENV=${ALGORITHM:-}

BATCH_NAMES=(
"rover_2poi_c8"
)

EXPERIMENT_NAMES=(
"mappo_mlp"
"rmappo"
# "mlp"
# "simplified_feudal_tanh_relative_input"
# "simplified_feudal_tanh_relative_input_interventions_bandit" "simplified_feudal_tanh_relative_input_interventions_bandit_no_zero_penalty"
# "simplified_feudal_tanh_relative_input_interventions_adaptive_random" "simplified_feudal_tanh_relative_input_interventions_adaptive_skip"
)
ALGORITHM=rmappo_jax

# ALGORITHM=mappo_jax

TRIAL_START=4
TRIAL_END=9

# These can be set from the environment, as in the usage lines above.
N_PARALLEL=${N_PARALLEL:-1}
CPU_SLOTS=${CPU_SLOTS:-}
GPUS=${GPUS:-}
TRIALS_FILE=${TRIALS_FILE:-}
DRY_RUN=${DRY_RUN:-0}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-300}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_DIR="${SCRIPT_DIR}/conf"
source "${SCRIPT_DIR}/scripts/train_batches_progress.sh"

# Reads TRIALS_FILE into TRIALS ("<batch> <experiment> <trial>" entries), and
# BATCH_NAMES / EXPERIMENT_NAMES (each name once, in order of appearance) for the
# config check. Returns 1 on a missing file, a wrong header or a malformed line.
read_trials_file() {
    local line batch experiment trial extra number=1
    local -A seen_batch=() seen_experiment=()
    if [[ ! -f "${TRIALS_FILE}" ]]; then
        echo "error: TRIALS_FILE '${TRIALS_FILE}' not found" >&2
        return 1
    fi
    IFS= read -r line < "${TRIALS_FILE}"
    if [[ "${line%$'\r'}" != $'batch\texperiment\ttrial' ]]; then
        echo "error: the first line of ${TRIALS_FILE} must be the header" \
             "'batch<TAB>experiment<TAB>trial'" >&2
        return 1
    fi
    BATCH_NAMES=()
    EXPERIMENT_NAMES=()
    while IFS=$'\t' read -r batch experiment trial extra; do
        number=$(( number + 1 ))
        trial=${trial%$'\r'}
        [[ -z "${batch}${experiment}${trial}${extra}" ]] && continue
        if [[ -n "${extra}" || ! "${batch}" =~ ^[^[:space:]]+$ \
                || ! "${experiment}" =~ ^[^[:space:]]+$ || ! "${trial}" =~ ^[^[:space:]]+$ ]]; then
            echo "error: line ${number} of ${TRIALS_FILE} is not three tab-separated" \
                 "names without spaces" >&2
            return 1
        fi
        TRIALS+=("${batch} ${experiment} ${trial}")
        [[ -z "${seen_batch[${batch}]}" ]] && BATCH_NAMES+=("${batch}")
        [[ -z "${seen_experiment[${experiment}]}" ]] && EXPERIMENT_NAMES+=("${experiment}")
        seen_batch[${batch}]=1
        seen_experiment[${experiment}]=1
    done < <(tail -n +2 "${TRIALS_FILE}")
}

TRIALS=()
if [[ -n "${TRIALS_FILE}" ]]; then
    if [[ -z "${ALGORITHM_FROM_ENV}" ]]; then
        echo "error: TRIALS_FILE needs ALGORITHM set in the environment" >&2
        exit 1
    fi
    ALGORITHM=${ALGORITHM_FROM_ENV}
    if ! read_trials_file; then
        echo "nothing was run" >&2
        exit 1
    fi
    if [[ ${#TRIALS[@]} -eq 0 ]]; then
        echo "error: no trials in ${TRIALS_FILE}" >&2
        exit 1
    fi
else
    for BATCH_NAME in "${BATCH_NAMES[@]}"; do
        for EXPERIMENT in "${EXPERIMENT_NAMES[@]}"; do
            for TRIAL_ID in $(seq "${TRIAL_START}" "${TRIAL_END}"); do
                TRIALS+=("${BATCH_NAME} ${EXPERIMENT} ${TRIAL_ID}")
            done
        done
    done
    if [[ ${#TRIALS[@]} -eq 0 ]]; then
        echo "error: no trials (is a list empty, or TRIAL_START > TRIAL_END?)" >&2
        exit 1
    fi
fi

# Check every Hydra config group exists before running anything.
source "${SCRIPT_DIR}/scripts/check_conf_groups.sh"
if ! check_conf_groups "${CONF_DIR}" \
        "algorithm/${ALGORITHM}" \
        "${BATCH_NAMES[@]/#/env/}" \
        "${EXPERIMENT_NAMES[@]/#/model/}"; then
    echo "nothing was run" >&2
    exit 1
fi

if ! [[ "${N_PARALLEL}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "error: N_PARALLEL must be a non-negative integer, got '${N_PARALLEL}'" >&2
    exit 1
fi
if ! [[ "${PROGRESS_INTERVAL}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "error: PROGRESS_INTERVAL must be a non-negative integer (seconds), got '${PROGRESS_INTERVAL}'" >&2
    exit 1
fi

# "16-31" or "0-3,8" -> one CPU number per line.
expand_cpu_list() {
    local part
    local -a parts
    IFS=, read -ra parts <<< "$1"
    for part in "${parts[@]}"; do
        if [[ "${part}" == *-* ]]; then
            seq "${part%-*}" "${part#*-}"
        else
            echo "${part}"
        fi
    done
}

# Checks CPU_SLOT_LIST. Returns 1 on a malformed slot, a CPU this script may not
# run on, or a CPU listed twice. Warns about efficiency cores.
check_cpu_slots() {
    local slot cpu slow
    local -A used=() efficiency=()
    # Intel hybrid CPUs list their efficiency cores here.
    if [[ -r /sys/devices/cpu_atom/cpus ]]; then
        for cpu in $(expand_cpu_list "$(< /sys/devices/cpu_atom/cpus)"); do
            efficiency[${cpu}]=1
        done
    fi
    for slot in "${CPU_SLOT_LIST[@]}"; do
        if ! [[ "${slot}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            echo "error: CPU slot '${slot}' is not a comma-separated list of CPU numbers" >&2
            return 1
        fi
        if ! taskset -c "${slot}" true 2> /dev/null; then
            echo "error: CPU slot '${slot}' names a CPU this script cannot run on" \
                 "(allowed: $(taskset -cp $$ | sed 's/.*: //'))" >&2
            return 1
        fi
        slow=()
        for cpu in ${slot//,/ }; do
            if [[ -n "${used[${cpu}]}" ]]; then
                echo "error: CPU ${cpu} is in more than one CPU slot" >&2
                return 1
            fi
            used[${cpu}]=1
            [[ -n "${efficiency[${cpu}]}" ]] && slow+=("${cpu}")
        done
        if [[ ${#slow[@]} -gt 0 ]]; then
            echo "warning: CPU slot '${slot}' uses efficiency cores (${slow[*]}); trials" \
                 "ran about 2x slower on these than on performance cores" >&2
        fi
    done
}

read -ra CPU_SLOT_LIST <<< "${CPU_SLOTS}"
if ! check_cpu_slots; then
    echo "nothing was run" >&2
    exit 1
fi

read -ra GPU_LIST <<< "${GPUS}"
declare -A SEEN_GPU=()
for gpu in "${GPU_LIST[@]}"; do
    if [[ "${gpu}" == *,* ]]; then
        echo "error: GPUS is space-separated, got '${GPUS}'" >&2
        exit 1
    fi
    if [[ -n "${SEEN_GPU[${gpu}]}" ]]; then
        echo "error: GPU ${gpu} is listed twice in GPUS" >&2
        exit 1
    fi
    SEEN_GPU[${gpu}]=1
done
if (( ${#GPU_LIST[@]} > 0 && N_PARALLEL == 0 )); then
    echo "error: GPUS is set but N_PARALLEL=0 gives it no slots" >&2
    exit 1
fi

# Slot i runs its trials on SLOT_DEVICE[i]: "gpu" (the default GPU), "gpu:<id>",
# or "cpu:<CPU list>". GPU slots come first, so they are used first. With GPUS
# they alternate between the GPUs, so the first trials spread over all of them.
SLOT_DEVICE=()
for (( i = 0; i < N_PARALLEL; i++ )); do
    if (( ${#GPU_LIST[@]} > 0 )); then
        SLOT_DEVICE+=("${GPU_LIST[@]/#/gpu:}")
    else
        SLOT_DEVICE+=(gpu)
    fi
done
for slot in "${CPU_SLOT_LIST[@]}"; do
    SLOT_DEVICE+=("cpu:${slot}")
done
if [[ ${#SLOT_DEVICE[@]} -eq 0 ]]; then
    echo "error: no slots (N_PARALLEL=0 needs CPU_SLOTS)" >&2
    exit 1
fi

if (( ${#GPU_LIST[@]} > 0 )); then
    GPU_SLOTS_TEXT="${N_PARALLEL} on each of GPUs ${GPU_LIST[*]}"
else
    GPU_SLOTS_TEXT="${N_PARALLEL} on the GPU"
fi

# From here on a run prints nothing on the terminal: everything this script
# prints goes to train_batches.log in the run's log directory. A dry run prints
# it instead, as listing the run is all a dry run is for.
if [[ "${DRY_RUN}" != "1" ]]; then
    RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_DIR="${SCRIPT_DIR}/logs/train_batches/${RUN_TIMESTAMP}"
    mkdir -p "${LOG_DIR}" || exit 1
    exec > "${LOG_DIR}/train_batches.log" 2>&1
fi

if [[ -n "${TRIALS_FILE}" ]]; then
    echo "${#TRIALS[@]} trials of ${ALGORITHM} from ${TRIALS_FILE} (${#BATCH_NAMES[@]} batches," \
         "${#EXPERIMENT_NAMES[@]} experiments), ${#SLOT_DEVICE[@]} at a time" \
         "(${GPU_SLOTS_TEXT}, ${#CPU_SLOT_LIST[@]} on the CPU)"
else
    echo "${#TRIALS[@]} trials of ${ALGORITHM}: ${#BATCH_NAMES[@]} batches x" \
         "${#EXPERIMENT_NAMES[@]} experiments x trials ${TRIAL_START}-${TRIAL_END}," \
         "${#SLOT_DEVICE[@]} at a time (${GPU_SLOTS_TEXT}, ${#CPU_SLOT_LIST[@]} on the CPU)"
fi
echo "slots: ${SLOT_DEVICE[*]}"

# Resident memory per trial on mjx_3a_1o_1024_gs: 3.0-3.7 GB for CPU trials in
# short runs, up to 5.2 GB for a GPU trial after 4 hours. Running out of RAM
# would kill trials, so warn when the trials that run at once may not fit.
# Inside a SLURM job the limit is the job's memory request (--mem, in MiB), not
# what the node has free; going over it kills the whole job.
N_CONCURRENT=$(( ${#SLOT_DEVICE[@]} < ${#TRIALS[@]} ? ${#SLOT_DEVICE[@]} : ${#TRIALS[@]} ))
AVAILABLE_MIB=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
if [[ "${SLURM_MEM_PER_NODE}" =~ ^[1-9][0-9]*$ ]] && (( SLURM_MEM_PER_NODE < AVAILABLE_MIB )); then
    AVAILABLE_MIB=${SLURM_MEM_PER_NODE}
fi
if [[ -n "${AVAILABLE_MIB}" ]] && (( N_CONCURRENT * 4500 > AVAILABLE_MIB )); then
    echo "warning: ${N_CONCURRENT} trials at once may need about" \
         "$(( N_CONCURRENT * 4500 )) MiB of RAM, and ${AVAILABLE_MIB} MiB is available" >&2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "each trial runs in the first free slot, GPU slots first:"
    for TRIAL in "${TRIALS[@]}"; do
        read -r BATCH_NAME EXPERIMENT TRIAL_ID <<< "${TRIAL}"
        echo "bash train.sh ${BATCH_NAME} ${ALGORITHM} '' ${TRIAL_ID} ${EXPERIMENT}"
    done
    exit 0
fi

# JAX reserves 75% of GPU memory in every process at startup, so a second trial
# on the same GPU would run out of memory. N_PARALLEL is the number of trials
# per GPU, with or without GPUS. Unless a JAX memory setting is
# already in the environment, each trial allocates only what it uses. A fixed
# share per trial (XLA_PYTHON_CLIENT_MEM_FRACTION) does not work: the CUDA
# graphs XLA builds need memory outside that share, and bandit trials ran out of
# it at 0.28 of the GPU each.
if (( N_PARALLEL > 1 )) && \
        [[ -z "${XLA_PYTHON_CLIENT_MEM_FRACTION}${XLA_PYTHON_CLIENT_PREALLOCATE}${XLA_PYTHON_CLIENT_ALLOCATOR}" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    echo "trials allocate GPU memory on demand (XLA_PYTHON_CLIENT_PREALLOCATE=false)"
fi

# Other processes holding GPU memory (e.g. trials left over from an earlier run)
# make new trials fail with out-of-memory errors, so name them up front.
if (( N_PARALLEL > 0 )) && command -v nvidia-smi > /dev/null; then
    while IFS=', ' read -r GPU_PID GPU_MIB; do
        if (( GPU_MIB >= 500 )); then
            echo "warning: pid ${GPU_PID} already holds ${GPU_MIB} MiB of GPU memory:" \
                 "$(ps -o args= -p "${GPU_PID}" | cut -c1-150)" >&2
        fi
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
fi

# Python buffers output that goes to a file or pipe and writes it only on exit,
# so a log would show no progress while its trial runs, and a stopped trial
# would lose its last lines.
export PYTHONUNBUFFERED=1

SUMMARY="${LOG_DIR}/summary.tsv"
printf 'status\tbatch\texperiment\ttrial\tdevice\tminutes\n' > "${SUMMARY}"
# scripts/train_batches_progress.sh reads the trial list, and this script's
# pid, host and SLURM job to tell a running run from one that was stopped. The
# job file holds the job id and SLURM_RESTART_COUNT: a requeued job keeps its id
# but starts a new run in a new log directory.
{
    printf 'batch\texperiment\ttrial\n'
    printf '%s\n' "${TRIALS[@]}" | tr ' ' '\t'
} > "${LOG_DIR}/trials.tsv"
echo $$ > "${LOG_DIR}/train_batches.pid"
hostname > "${LOG_DIR}/train_batches.host"
if [[ -n "${SLURM_JOB_ID}" ]]; then
    echo "${SLURM_JOB_ID} ${SLURM_RESTART_COUNT:-0}" > "${LOG_DIR}/slurm_job"
fi

run_trial() {
    local batch=$1 experiment=$2 trial=$3 device=$4
    local log="${LOG_DIR}/${batch}__${experiment}__${trial}.log"
    local start status result
    local -a launch=()
    # A CPU slot keeps JAX off the GPU and pins the trial to the slot's CPUs. A
    # "gpu:<id>" slot (GPUS) shows the trial only its own GPU.
    if [[ "${device}" == cpu:* ]]; then
        launch=(env JAX_PLATFORMS=cpu taskset -c "${device#cpu:}")
    elif [[ "${device}" == gpu:* ]]; then
        launch=(env CUDA_VISIBLE_DEVICES="${device#gpu:}")
    fi
    start=$(date +%s)
    # The progress report reads the device of a running trial from this line.
    echo "train_batches.sh: started $(date '+%F %T'), device ${device}" > "${log}"
    "${launch[@]}" bash "${SCRIPT_DIR}/train.sh" "${batch}" "${ALGORITHM}" "" "${trial}" "${experiment}" \
        >> "${log}" 2>&1
    status=$?
    result=OK
    (( status == 0 )) || result="FAILED(${status})"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${result}" "${batch}" "${experiment}" "${trial}" "${device}" \
        "$(( ($(date +%s) - start) / 60 ))" >> "${SUMMARY}"
    echo "[$(date +%H:%M:%S)] ${result}  ${batch} / ${experiment} / ${trial} (${device})"
    return "${status}"
}

stop_all() {
    # Ignore further signals, so a second Ctrl-C cannot end this script before
    # every trial is stopped (the trials would keep running and hold the GPU).
    trap '' INT TERM HUP QUIT
    echo "interrupted: stopping running trials" >&2
    local pgids pgid alive deadline=$((SECONDS + 10))
    pgids=$(jobs -p)
    for pgid in ${pgids}; do
        kill -TERM -- "-${pgid}" 2>/dev/null
    done
    # Give the trials 10 s to exit, then kill whatever is left of each group.
    while (( SECONDS < deadline )); do
        alive=0
        for pgid in ${pgids}; do
            kill -0 -- "-${pgid}" 2>/dev/null && alive=1
        done
        (( alive )) || break
        sleep 0.5
    done
    for pgid in ${pgids}; do
        kill -KILL -- "-${pgid}" 2>/dev/null && echo "force-killed process group ${pgid}" >&2
    done
    wait
    exit 130
}

# True if another process (an earlier run of this script, or a train.sh started
# by hand) is already training this trial. Starting it again would put two
# processes on one results directory.
already_running() {
    pgrep -f "train\.py env=$1 model=$2 algorithm=${ALGORITHM} trial_id=$3 " > /dev/null
}

# SLOT_PID[i] is the job running in slot i (empty before its first trial).
SLOT_PID=()

# True while slot $1's job is among the running jobs.
slot_busy() {
    [[ -n "${SLOT_PID[$1]}" && " $(jobs -rp | tr '\n' ' ') " == *" ${SLOT_PID[$1]} "* ]]
}

# Sets SLOT to the first free slot, or to "" if every slot is busy.
find_free_slot() {
    local i
    SLOT=""
    for i in "${!SLOT_DEVICE[@]}"; do
        if ! slot_busy "${i}"; then
            SLOT=${i}
            return
        fi
    done
}

# Appends the running trials' progress to progress.log every PROGRESS_INTERVAL
# seconds, until it is sent TERM. Sleeping in the background and waiting for it lets the TERM
# trap run at once and exit quietly; a killed foreground sleep would print
# "Terminated".
progress_monitor() {
    local nap
    trap 'kill "${nap}" 2> /dev/null; exit 0' TERM
    while :; do
        sleep "${PROGRESS_INTERVAL}" & nap=$!
        wait "${nap}" || exit 0
        print_progress "${LOG_DIR}" running
    done
}

# Waits 2 seconds. Sleeping in the background and waiting for it lets a signal
# run the trap at once, as in the monitor.
nap() {
    sleep 2 &
    wait $!
}

# Job control puts each trial in its own process group, so stop_all can stop
# the whole tree (bash -> uv -> python) and not just the subshell.
set -m
trap stop_all INT TERM HUP QUIT

# The monitor is a job too, so stop_all stops it with the trials.
MONITOR_PID=""
if (( PROGRESS_INTERVAL > 0 )); then
    progress_monitor < /dev/null >> "${LOG_DIR}/progress.log" 2>&1 &
    MONITOR_PID=$!
fi

for TRIAL in "${TRIALS[@]}"; do
    read -r BATCH_NAME EXPERIMENT TRIAL_ID <<< "${TRIAL}"
    # Check for a free slot every 2 s. Not `wait -n` (see below): it could
    # leave a slot idle until another trial ended.
    find_free_slot
    while [[ -z "${SLOT}" ]]; do
        nap
        find_free_slot
    done
    if already_running "${BATCH_NAME}" "${EXPERIMENT}" "${TRIAL_ID}"; then
        printf 'SKIPPED(running)\t%s\t%s\t%s\t-\t0\n' "${BATCH_NAME}" "${EXPERIMENT}" "${TRIAL_ID}" >> "${SUMMARY}"
        echo "[$(date +%H:%M:%S)] skip    ${BATCH_NAME} / ${EXPERIMENT} / ${TRIAL_ID} (already running elsewhere)"
        continue
    fi
    echo "[$(date +%H:%M:%S)] start   ${BATCH_NAME} / ${EXPERIMENT} / ${TRIAL_ID} (${SLOT_DEVICE[SLOT]})"
    run_trial "${BATCH_NAME}" "${EXPERIMENT}" "${TRIAL_ID}" "${SLOT_DEVICE[SLOT]}" < /dev/null &
    SLOT_PID[SLOT]=$!
done
# Wait for the last trials, each by its process id; `wait <pid>` returns at
# once for a job that has already ended. A plain `wait` would also wait for the
# monitor, which runs until it is stopped. Until 2026-10-28 this loop used
# `wait -n`, which before bash 5.3 skips a job whose end the shell has already
# reported and waits for the next job to end instead. When that next job was
# the monitor, the script never exited, and the monitor kept printing
# "<N> done" every PROGRESS_INTERVAL.
for pid in "${SLOT_PID[@]}"; do
    wait "${pid}" 2> /dev/null
done
if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2> /dev/null
    wait "${MONITOR_PID}"
fi

echo
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
N_FAILED=$(awk -F'\t' 'NR > 1 && $1 ~ /^FAILED/' "${SUMMARY}" | wc -l)
N_SKIPPED=$(awk -F'\t' 'NR > 1 && $1 ~ /^SKIPPED/' "${SUMMARY}" | wc -l)
echo "$(( ${#TRIALS[@]} - N_FAILED - N_SKIPPED )) of ${#TRIALS[@]} trials finished," \
     "${N_FAILED} failed, ${N_SKIPPED} skipped (already running elsewhere)"
(( N_FAILED == 0 ))
