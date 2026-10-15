#!/bin/bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKING_DIR="${SCRIPT_DIR}/job_tracking"
DEFAULT_RUN_SCRIPT="${SCRIPT_DIR}/run_trial_gpu.sh"

LEDGER_PATH=""
BATCH_IDENTIFIER_FILTER=""
RUN_SCRIPT="${DEFAULT_RUN_SCRIPT}"
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [algorithm_batch_name] [--dry-run] [--run-script PATH]

Checks each expected job in a ledger and resubmits any job that is not
currently present in squeue for this user, unless sacct shows it finished
with state COMPLETED: either the ledger's job ID, or any job with the same
job name submitted since the ledger was written (i.e. a resubmission).
Exits without resubmitting anything if a sacct query fails.

Options:
  algorithm_batch_name
                     Optional identifier in the form <algorithm>_<batch_name>.
                     If provided, uses
                     ${TRACKING_DIR}/<algorithm>_<batch_name>_latest.csv.
                     If omitted, newest *_latest.csv in ${TRACKING_DIR} is used.
  --dry-run          Show what would be resubmitted without submitting.
  --run-script PATH  Submission script to call (default: ${DEFAULT_RUN_SCRIPT}).
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --run-script)
            if [[ $# -lt 2 ]]; then
                echo "error: --run-script requires a path" >&2
                usage
                exit 1
            fi
            RUN_SCRIPT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -n "${BATCH_IDENTIFIER_FILTER}" ]]; then
                echo "error: unexpected argument '$1'" >&2
                usage
                exit 1
            fi
            BATCH_IDENTIFIER_FILTER="$1"
            shift
            ;;
    esac
done

if [[ ! -d "${TRACKING_DIR}" ]]; then
    echo "error: tracking directory not found: ${TRACKING_DIR}" >&2
    exit 1
fi

if [[ -z "${BATCH_IDENTIFIER_FILTER}" ]]; then
    mapfile -t LATEST_LEDGERS < <(ls -1t "${TRACKING_DIR}"/*_latest.csv 2>/dev/null)
    if [[ ${#LATEST_LEDGERS[@]} -eq 0 ]]; then
        echo "error: no *_latest.csv found in ${TRACKING_DIR}" >&2
        exit 1
    fi
    LEDGER_PATH="${LATEST_LEDGERS[0]}"
else
    LEDGER_PATH="${TRACKING_DIR}/${BATCH_IDENTIFIER_FILTER}_latest.csv"
    if [[ ! -f "${LEDGER_PATH}" ]]; then
        echo "error: no latest ledger found for identifier '${BATCH_IDENTIFIER_FILTER}': ${LEDGER_PATH}" >&2
        exit 1
    fi
fi

if [[ ! -f "${LEDGER_PATH}" ]]; then
    echo "error: ledger not found: ${LEDGER_PATH}" >&2
    exit 1
fi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "error: run script not found: ${RUN_SCRIPT}" >&2
    exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
    echo "error: squeue command not found in PATH" >&2
    exit 1
fi
if ! command -v sacct >/dev/null 2>&1; then
    echo "error: sacct command not found in PATH" >&2
    exit 1
fi

mkdir -p "${TRACKING_DIR}"
WATCHDOG_LOG="${TRACKING_DIR}/watchdog_actions.csv"
if [[ ! -s "${WATCHDOG_LOG}" ]]; then
    echo "timestamp,job_name,batch_name,experiment,trial_id,status,old_job_id,new_job_id" > "${WATCHDOG_LOG}"
fi

# The name every launcher passes to `sbatch -J`; resubmissions reuse it.
expected_job_name() {
    local trial_id="$1" experiment="$2" algorithm="$3" batch_name="$4"
    echo "${trial_id}_${experiment}_${algorithm}_${batch_name}"
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

declare -A LEDGER_IDS
declare -A LEDGER_NAMES
declare -A COMPLETED_IDS
declare -A COMPLETED_NAMES
declare -A ACTIVE_IDS
declare -A ACTIVE_NAMES
LEDGER_START=""

while IFS='|' read -r ACTIVE_ID ACTIVE_NAME; do
    [[ -n "${ACTIVE_ID}" ]] && ACTIVE_IDS["${ACTIVE_ID}"]=1
    [[ -n "${ACTIVE_NAME}" ]] && ACTIVE_NAMES["${ACTIVE_NAME}"]=1
done < <(squeue -h -u "${USER}" -o "%A|%j")

while IFS=',' read -r TIMESTAMP _RUN_ID BATCH_NAME _ENVIRONMENT ALGORITHM EXPERIMENT TRIAL_ID LEDGER_JOB_NAME JOB_ID _SUBMISSION_OK; do
    JOB_ID="${JOB_ID%$'\r'}"
    [[ "${JOB_ID}" =~ ^[0-9]+$ ]] && LEDGER_IDS["${JOB_ID}"]=1

    [[ -z "${BATCH_NAME}" || "${BATCH_NAME}" == "batch_name" ]] && continue
    LEDGER_NAMES["$(expected_job_name "${TRIAL_ID}" "${EXPERIMENT}" "${ALGORITHM}" "${BATCH_NAME}")"]=1
    [[ -n "${LEDGER_JOB_NAME}" ]] && LEDGER_NAMES["${LEDGER_JOB_NAME}"]=1

    # Ledger timestamps are local time with a %z offset; sacct -S takes local
    # time without one. Same-format strings compare chronologically.
    if [[ "${TIMESTAMP}" =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}) ]]; then
        if [[ -z "${LEDGER_START}" || "${BASH_REMATCH[1]}" < "${LEDGER_START}" ]]; then
            LEDGER_START="${BASH_REMATCH[1]}"
        fi
    fi
done < "${LEDGER_PATH}"

# A failed sacct query would leave the completed sets empty and resubmit every
# finished job, so abort the cycle instead.
if [[ ${#LEDGER_IDS[@]} -gt 0 ]]; then
    if ! ACCOUNTING="$(sacct -n -P -X -j "$(join_by_comma "${!LEDGER_IDS[@]}")" -o JobIDRaw,State)"; then
        echo "error: sacct query by job id failed; not resubmitting anything this cycle" >&2
        exit 1
    fi
    while IFS='|' read -r ACCOUNT_JOB_ID ACCOUNT_STATE; do
        ACCOUNT_JOB_ID="${ACCOUNT_JOB_ID%$'\r'}"
        ACCOUNT_STATE="${ACCOUNT_STATE%$'\r'}"
        [[ -z "${ACCOUNT_JOB_ID}" || -z "${ACCOUNT_STATE}" ]] && continue
        if [[ "${ACCOUNT_STATE}" == COMPLETED* ]]; then
            COMPLETED_IDS["${ACCOUNT_JOB_ID}"]=1
        fi
    done <<< "${ACCOUNTING}"
fi

# The ledger keeps the ID of the first submission only, so a job the watchdog
# resubmitted can complete under an ID the ledger never sees. Match completed
# jobs by name as well. Jobs from before this ledger was written are excluded,
# so a relaunched batch is not satisfied by an older run with the same names.
if [[ ${#LEDGER_NAMES[@]} -gt 0 && -z "${LEDGER_START}" ]]; then
    echo "warning: no parseable timestamp in ${LEDGER_PATH}; checking completion by ledger job id only" >&2
elif [[ ${#LEDGER_NAMES[@]} -gt 0 ]]; then
    if ! ACCOUNTING="$(sacct -n -P -X -u "${USER}" -S "${LEDGER_START}" \
            --name="$(join_by_comma "${!LEDGER_NAMES[@]}")" -o JobIDRaw,JobName,State)"; then
        echo "error: sacct query by job name failed; not resubmitting anything this cycle" >&2
        exit 1
    fi
    while IFS='|' read -r ACCOUNT_JOB_ID ACCOUNT_NAME ACCOUNT_STATE; do
        ACCOUNT_STATE="${ACCOUNT_STATE%$'\r'}"
        [[ "${ACCOUNT_JOB_ID}" =~ ^[0-9]+$ && -n "${ACCOUNT_NAME}" ]] || continue
        [[ "${ACCOUNT_STATE}" == COMPLETED* ]] || continue
        # Keep the most recent completed job for each name.
        PREVIOUS_ID="${COMPLETED_NAMES[${ACCOUNT_NAME}]:-0}"
        if (( ACCOUNT_JOB_ID > PREVIOUS_ID )); then
            COMPLETED_NAMES["${ACCOUNT_NAME}"]="${ACCOUNT_JOB_ID}"
        fi
    done <<< "${ACCOUNTING}"
fi

TOTAL=0
RUNNING=0
COMPLETED=0
RESUBMITTED=0
WOULD_RESUBMIT=0
FAILED=0

while IFS=',' read -r _TIMESTAMP _RUN_ID BATCH_NAME ENVIRONMENT ALGORITHM EXPERIMENT TRIAL_ID LEDGER_JOB_NAME JOB_ID _SUBMISSION_OK; do
    JOB_ID="${JOB_ID%$'\r'}"
    LEDGER_JOB_NAME="${LEDGER_JOB_NAME%$'\r'}"
    TRIAL_ID="${TRIAL_ID%$'\r'}"
    EXPERIMENT="${EXPERIMENT%$'\r'}"
    BATCH_NAME="${BATCH_NAME%$'\r'}"
    ENVIRONMENT="${ENVIRONMENT%$'\r'}"
    ALGORITHM="${ALGORITHM%$'\r'}"

    [[ -z "${BATCH_NAME}" ]] && continue
    if [[ "${BATCH_NAME}" == "batch_name" ]]; then
        continue
    fi

    JOB_NAME="$(expected_job_name "${TRIAL_ID}" "${EXPERIMENT}" "${ALGORITHM}" "${BATCH_NAME}")"

    TOTAL=$((TOTAL + 1))

    IS_ACTIVE=0
    if [[ -n "${JOB_ID}" && "${JOB_ID}" != "UNKNOWN" && -n "${ACTIVE_IDS[${JOB_ID}]+x}" ]]; then
        IS_ACTIVE=1
    fi
    if [[ ${IS_ACTIVE} -eq 0 && -n "${ACTIVE_NAMES[${JOB_NAME}]+x}" ]]; then
        IS_ACTIVE=1
    fi
    # Avoid duplicating an in-flight job created before algorithm-prefixed names
    # were introduced. New submissions always use JOB_NAME above.
    if [[ ${IS_ACTIVE} -eq 0 && -n "${LEDGER_JOB_NAME}" && -n "${ACTIVE_NAMES[${LEDGER_JOB_NAME}]+x}" ]]; then
        IS_ACTIVE=1
    fi

    LOG_TS="$(date +%Y-%m-%dT%H:%M:%S%z)"
    if [[ ${IS_ACTIVE} -eq 1 ]]; then
        RUNNING=$((RUNNING + 1))
        echo "running ${JOB_NAME} (${JOB_ID})"
        echo "${LOG_TS},${JOB_NAME},${BATCH_NAME},${EXPERIMENT},${TRIAL_ID},running,${JOB_ID}," >> "${WATCHDOG_LOG}"
        continue
    fi

    # The job that completed it: the ledger's own ID, or a resubmission found by name.
    COMPLETED_BY=""
    if [[ -n "${JOB_ID}" && "${JOB_ID}" != "UNKNOWN" && -n "${COMPLETED_IDS[${JOB_ID}]+x}" ]]; then
        COMPLETED_BY="${JOB_ID}"
    elif [[ -n "${COMPLETED_NAMES[${JOB_NAME}]+x}" ]]; then
        COMPLETED_BY="${COMPLETED_NAMES[${JOB_NAME}]}"
    elif [[ -n "${LEDGER_JOB_NAME}" && -n "${COMPLETED_NAMES[${LEDGER_JOB_NAME}]+x}" ]]; then
        COMPLETED_BY="${COMPLETED_NAMES[${LEDGER_JOB_NAME}]}"
    fi
    if [[ -n "${COMPLETED_BY}" ]]; then
        COMPLETED=$((COMPLETED + 1))
        echo "completed ${JOB_NAME} (${COMPLETED_BY})"
        echo "${LOG_TS},${JOB_NAME},${BATCH_NAME},${EXPERIMENT},${TRIAL_ID},completed,${JOB_ID},${COMPLETED_BY}" >> "${WATCHDOG_LOG}"
        continue
    fi

    if [[ ${DRY_RUN} -eq 1 ]]; then
        WOULD_RESUBMIT=$((WOULD_RESUBMIT + 1))
        echo "would resubmit ${JOB_NAME} (${JOB_ID})"
        echo "${LOG_TS},${JOB_NAME},${BATCH_NAME},${EXPERIMENT},${TRIAL_ID},would_resubmit,${JOB_ID}," >> "${WATCHDOG_LOG}"
        continue
    fi

    SUBMISSION_OUTPUT="$(
        bash "${RUN_SCRIPT}" \
            "${BATCH_NAME}" \
            "${EXPERIMENT}" \
            "${ALGORITHM}" \
            "${ENVIRONMENT}" \
            "${TRIAL_ID}" 2>&1
    )"
    SUBMIT_EXIT=$?

    NEW_JOB_ID="$(printf '%s\n' "${SUBMISSION_OUTPUT}" | sed -n -E 's/.*Submitted batch job ([0-9]+).*/\1/p' | tail -n 1)"
    if [[ -z "${NEW_JOB_ID}" ]]; then
        NEW_JOB_ID="$(printf '%s\n' "${SUBMISSION_OUTPUT}" | sed -n -E 's/^([0-9]+)(;.*)?$/\1/p' | tail -n 1)"
    fi

    if [[ ${SUBMIT_EXIT} -eq 0 ]]; then
        RESUBMITTED=$((RESUBMITTED + 1))
        [[ -n "${NEW_JOB_ID}" ]] && ACTIVE_IDS["${NEW_JOB_ID}"]=1
        ACTIVE_NAMES["${JOB_NAME}"]=1
        echo "resubmitted ${JOB_NAME} -> ${NEW_JOB_ID:-UNKNOWN}"
        echo "${LOG_TS},${JOB_NAME},${BATCH_NAME},${EXPERIMENT},${TRIAL_ID},resubmitted,${JOB_ID},${NEW_JOB_ID:-UNKNOWN}" >> "${WATCHDOG_LOG}"
    else
        FAILED=$((FAILED + 1))
        echo "failed to resubmit ${JOB_NAME}" >&2
        echo "${SUBMISSION_OUTPUT}" >&2
        echo "${LOG_TS},${JOB_NAME},${BATCH_NAME},${EXPERIMENT},${TRIAL_ID},resubmit_failed,${JOB_ID},UNKNOWN" >> "${WATCHDOG_LOG}"
    fi
done < "${LEDGER_PATH}"

echo
echo "watchdog summary:"
echo "  ledger:      ${LEDGER_PATH}"
echo "  actions log: ${WATCHDOG_LOG}"
echo "  total jobs:  ${TOTAL}"
echo "  running:     ${RUNNING}"
echo "  completed:   ${COMPLETED}"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  would_resubmit: ${WOULD_RESUBMIT}"
else
    echo "  resubmitted: ${RESUBMITTED}"
fi
echo "  failed:      ${FAILED}"
