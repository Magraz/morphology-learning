#!/bin/bash
# Progress of every trial of one train_batches.sh run, read from that run's log
# directory (logs/train_batches/<timestamp>/). train_batches.sh sources this
# file and prints the running trials every PROGRESS_INTERVAL seconds. Run it
# directly to see every trial at any time, e.g. from a second terminal:
#
#   bash scripts/train_batches_progress.sh               # the latest run
#   bash scripts/train_batches_progress.sh logs/train_batches/<timestamp>
#   watch -n 60 bash scripts/train_batches_progress.sh   # redraw every minute
#
# One row per trial:
#
#   21.4% [####----------------] 2h05m left     gpu      <batch> / <experiment> / <trial>
#
# The percentage is the last "Steps: <done>/<total>" line the trial's trainer
# printed: mappo, mappo_vanilla, ippo and the JAX stacks print one per update.
# dcg prints no total, so its trials show "-". A resumed trial's percentage
# includes the steps trained before the resume. The time left is the remaining
# steps divided by the speed over the last PROGRESS_SPEED_LINES of those lines,
# not over the whole run, because the speed changes as a policy trains (one
# interventions trial went from 1.4 s to 6.3 s per update in its first hour).
#
# States: running, done, FAILED(<exit status>), skipped (already being trained
# by another process), queued, and, once the run has ended without finishing a
# trial (Ctrl-C), stopped (it had started) or not run.
#
# Needs the run's trials.tsv, summary.tsv and train_batches.pid, which
# train_batches.sh writes since 2026-10-24. Since 2026-10-28 it also writes
# train_batches.host and, inside a SLURM job, slurm_job, so that a run on a
# compute node (scripts/hpc/run_node_gpu.sh) can be followed from the login node
# (see run_is_alive below).

PROGRESS_SPEED_LINES=100
PROGRESS_BAR_WIDTH=20

# Prints "<steps done> <total steps> <steps per second>" from the last progress
# line of trainer log $1, or nothing if there is none yet. The speed is 0 until
# there are two lines.
log_progress() {
    grep -a '^Steps: [0-9]*/[0-9]* ' "$1" 2> /dev/null \
        | tail -n "${PROGRESS_SPEED_LINES}" \
        | awk '{
            split($2, steps, "/")
            seconds = ""
            for (i = 3; i < NF; i++) if ($i == "Time:") seconds = $(i + 1) + 0
            if (NR == 1) { first_steps = steps[1] + 0; first_seconds = seconds }
            last_steps = steps[1] + 0; total = steps[2] + 0; last_seconds = seconds
        }
        END {
            if (NR == 0) exit
            speed = 0
            if (NR > 1 && first_seconds != "" && last_seconds > first_seconds)
                speed = (last_steps - first_steps) / (last_seconds - first_seconds)
            printf "%.0f %.0f %.3f\n", last_steps, total, speed
        }'
}

# Prints one row of the progress table. Arguments: state, device, trial name,
# then log_progress's three numbers (empty if there are none).
progress_row() {
    awk -v state="$1" -v device="$2" -v name="$3" \
        -v steps="$4" -v total="$5" -v speed="$6" -v width="${PROGRESS_BAR_WIDTH}" 'BEGIN {
        percent = -1
        if (total > 0) percent = 100 * steps / total
        if (state == "done") percent = 100
        filled = (percent < 0) ? 0 : int(width * percent / 100)
        bar = ""
        for (i = 0; i < width; i++) bar = bar (i < filled ? "#" : "-")
        # A running trial with no progress line yet (still compiling, or dcg)
        # keeps the plain "running".
        if (state == "running" && speed > 0) {
            left = int((total - steps) / speed)
            if (left >= 3600) state = sprintf("%dh%02dm left", int(left / 3600), int(left % 3600 / 60))
            else if (left >= 60) state = sprintf("%dm left", int(left / 60))
            else state = "<1m left"
        }
        # Truncated, not rounded, so 99.96% does not read as 100.0%.
        printf "  %6s [%s] %-15s %-8s %s\n", (percent < 0 ? "-" : sprintf("%.1f%%", int(percent * 10) / 10)),
            bar, state, device, name
    }'
}

# Returns 0 if the train_batches.sh run logged in directory $1 is still running,
# 1 if it is not, and 2 if this host cannot tell. On the host the run is on, the
# test is its pid. From another host, e.g. a SLURM login node looking at a run
# on a compute node, it is the run's SLURM job: it must be RUNNING and not
# requeued since the run started (a requeue keeps the job id but starts a new
# run in a new log directory). Runs from before 2026-10-28 have no host file
# and are checked by pid.
run_is_alive() {
    local dir=$1 host pid job restarts info state count
    host=$(cat "${dir}/train_batches.host" 2> /dev/null)
    if [[ -z "${host}" || "${host}" == "$(hostname)" ]]; then
        pid=$(cat "${dir}/train_batches.pid" 2> /dev/null)
        [[ -n "${pid}" ]] && ps -o args= -p "${pid}" 2> /dev/null | grep -q 'train_batches\.sh'
        return
    fi
    if [[ ! -f "${dir}/slurm_job" ]] || ! command -v scontrol > /dev/null; then
        return 2
    fi
    read -r job restarts < "${dir}/slurm_job"
    # A job SLURM no longer knows (finished long enough ago) is not running.
    info=$(scontrol show job -o "${job}" 2> /dev/null) || return 1
    [[ "${info}" =~ (^|[[:space:]])JobState=([A-Z_]+) ]] && state=${BASH_REMATCH[2]}
    [[ "${info}" =~ (^|[[:space:]])Restarts=([0-9]+) ]] && count=${BASH_REMATCH[2]}
    [[ "${state}" == RUNNING && "${count}" == "${restarts}" ]]
}

# Prints the progress of the run logged in directory $1: a line counting the
# trials in each state, then one row per trial, or with $2 = "running" one row
# per running trial.
print_progress() {
    local dir=$1 scope=${2:-all}
    local alive=0 note="" line batch experiment trial result device key log state class
    local -A result_of=() device_of=() count=()
    local -a rows=() progress
    if [[ ! -f "${dir}/trials.tsv" ]]; then
        echo "error: ${dir}/trials.tsv not found; train_batches.sh writes it since 2026-10-24" >&2
        return 1
    fi
    # Once the run has ended, a trial with no summary row will not finish.
    run_is_alive "${dir}"
    case $? in
        0) alive=1 ;;
        2) note=" (the run is on $(cat "${dir}/train_batches.host"); this host cannot tell whether it is still running)" ;;
    esac
    while IFS=$'\t' read -r result batch experiment trial device _; do
        result_of["${batch}/${experiment}/${trial}"]=${result}
        device_of["${batch}/${experiment}/${trial}"]=${device}
    done < <(tail -n +2 "${dir}/summary.tsv" 2> /dev/null)

    while IFS=$'\t' read -r batch experiment trial; do
        key="${batch}/${experiment}/${trial}"
        log="${dir}/${batch}__${experiment}__${trial}.log"
        result=${result_of[${key}]}
        case "${result}" in
            OK) state=done ;;
            FAILED*) state=${result} ;;
            SKIPPED*) state=skipped ;;
            *)
                if [[ -f "${log}" ]]; then
                    (( alive )) && state=running || state=stopped
                else
                    (( alive )) && state=queued || state="not run"
                fi
                ;;
        esac
        class=${state%%(*}
        count["${class}"]=$(( ${count["${class}"]:-0} + 1 ))
        [[ "${scope}" == running && "${state}" != running ]] && continue
        device=${device_of[${key}]}
        # run_trial writes the device on the first line of a trial's log.
        if [[ -z "${device}" && -f "${log}" ]]; then
            device=$(sed -n '1s/^train_batches\.sh: .*, device //p' "${log}")
        fi
        read -ra progress <<< "$(log_progress "${log}")"
        rows+=("$(progress_row "${state}" "${device:--}" "${batch} / ${experiment} / ${trial}" "${progress[@]}")")
    done < <(tail -n +2 "${dir}/trials.tsv")

    line=""
    for class in running done FAILED skipped stopped queued "not run"; do
        [[ -n "${count["${class}"]}" ]] && line+="${line:+, }${count["${class}"]} ${class,,}"
    done
    printf '%s\n' "[$(date +%H:%M:%S)] progress of $(( $(wc -l < "${dir}/trials.tsv") - 1 )) trials: ${line}${note}" \
        "${rows[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    RUN_DIR=${1:-$(ls -d "$(dirname "$0")"/../logs/train_batches/*/ 2> /dev/null | sort | tail -n 1)}
    if [[ -z "${RUN_DIR}" ]]; then
        echo "error: no run under logs/train_batches/; pass a run's log directory" >&2
        exit 1
    fi
    RUN_DIR=$(cd "${RUN_DIR}" && pwd) || exit 1
    echo "${RUN_DIR}"
    print_progress "${RUN_DIR}" all
fi
