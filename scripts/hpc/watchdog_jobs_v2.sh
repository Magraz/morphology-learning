#!/bin/bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKING_DIR="${SCRIPT_DIR}/job_tracking"
DEFAULT_RUN_SCRIPT="${SCRIPT_DIR}/run_trial_gpu.sh"

LEDGER_PATH=""
BATCH_NAME_FILTER=""
RUN_SCRIPT="${DEFAULT_RUN_SCRIPT}"
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [batch_name] [--dry-run] [--run-script PATH]

Checks each expected job in a ledger and resubmits any job that is not
currently present in squeue for this user, unless the ledger job already
finished with state COMPLETED in sacct.

Options:
  batch_name         Optional batch name. If provided, uses
                     ${TRACKING_DIR}/<batch_name>_latest.csv.
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
            if [[ -n "${BATCH_NAME_FILTER}" ]]; then
                echo "error: unexpected argument '$1'" >&2
                usage
                exit 1
            fi
            BATCH_NAME_FILTER="$1"
            shift
            ;;
    esac
done

if [[ ! -d "${TRACKING_DIR}" ]]; then
    echo "error: tracking directory not found: ${TRACKING_DIR}" >&2
    exit 1
fi

if [[ -z "${BATCH_NAME_FILTER}" ]]; then
    mapfile -t LATEST_LEDGERS < <(ls -1t "${TRACKING_DIR}"/*_latest.csv 2>/dev/null)
    if [[ ${#LATEST_LEDGERS[@]} -eq 0 ]]; then
        echo "error: no *_latest.csv found in ${TRACKING_DIR}" >&2
        exit 1
    fi
    LEDGER_PATH="${LATEST_LEDGERS[0]}"
else
    LEDGER_PATH="${TRACKING_DIR}/${BATCH_NAME_FILTER}_latest.csv"
    if [[ ! -f "${LEDGER_PATH}" ]]; then
        echo "error: no latest ledger found for batch '${BATCH_NAME_FILTER}': ${LEDGER_PATH}" >&2
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
CHECK_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WATCHDOG_LOG="${TRACKING_DIR}/watchdog_${CHECK_TIMESTAMP}.csv"
echo "timestamp,job_name,batch_name,experiment,trial_id,status,old_job_id,new_job_id" > "${WATCHDOG_LOG}"

declare -A LEDGER_IDS
declare -A COMPLETED_IDS
declare -A ACTIVE_IDS
declare -A ACTIVE_NAMES

while IFS='|' read -r ACTIVE_ID ACTIVE_NAME; do
    [[ -n "${ACTIVE_ID}" ]] && ACTIVE_IDS["${ACTIVE_ID}"]=1
    [[ -n "${ACTIVE_NAME}" ]] && ACTIVE_NAMES["${ACTIVE_NAME}"]=1
done < <(squeue -h -u "${USER}" -o "%A|%j")

while IFS=',' read -r _TIMESTAMP _RUN_ID BATCH_NAME _ENVIRONMENT _ALGORITHM _EXPERIMENT _TRIAL_ID _JOB_NAME JOB_ID _SUBMISSION_OK; do
    JOB_ID="${JOB_ID%$'\r'}"
    [[ "${JOB_ID}" =~ ^[0-9]+$ ]] && LEDGER_IDS["${JOB_ID}"]=1
done < "${LEDGER_PATH}"

LEDGER_JOB_IDS=("${!LEDGER_IDS[@]}")
if [[ ${#LEDGER_JOB_IDS[@]} -gt 0 ]]; then
    OLD_IFS="${IFS}"
    IFS=,
    JOB_ID_LIST="${LEDGER_JOB_IDS[*]}"
    IFS="${OLD_IFS}"

    while IFS='|' read -r ACCOUNT_JOB_ID ACCOUNT_STATE; do
        ACCOUNT_JOB_ID="${ACCOUNT_JOB_ID%$'\r'}"
        ACCOUNT_STATE="${ACCOUNT_STATE%$'\r'}"
        [[ -z "${ACCOUNT_JOB_ID}" || -z "${ACCOUNT_STATE}" ]] && continue
        if [[ "${ACCOUNT_STATE}" == COMPLETED* ]]; then
            COMPLETED_IDS["${ACCOUNT_JOB_ID}"]=1
        fi
    done < <(sacct -n -P -X -j "${JOB_ID_LIST}" -o JobIDRaw,State 2>/dev/null)
fi

TOTAL=0
RUNNING=0
COMPLETED=0
RESUBMITTED=0
WOULD_RESUBMIT=0
FAILED=0

while IFS=',' read -r _TIMESTAMP _RUN_ID BATCH_NAME ENVIRONMENT ALGORITHM EXPERIMENT TRIAL_ID JOB_NAME JOB_ID _SUBMISSION_OK; do
    JOB_ID="${JOB_ID%$'\r'}"
    JOB_NAME="${JOB_NAME%$'\r'}"
    TRIAL_ID="${TRIAL_ID%$'\r'}"
    EXPERIMENT="${EXPERIMENT%$'\r'}"
    BATCH_NAME="${BATCH_NAME%$'\r'}"
    ENVIRONMENT="${ENVIRONMENT%$'\r'}"
    ALGORITHM="${ALGORITHM%$'\r'}"

    [[ -z "${BATCH_NAME}" ]] && continue
    if [[ "${BATCH_NAME}" == "batch_name" ]]; then
        continue
    fi

    if [[ -z "${JOB_NAME}" ]]; then
        JOB_NAME="${TRIAL_ID}_${EXPERIMENT}_${BATCH_NAME}"
    fi

    TOTAL=$((TOTAL + 1))

    IS_ACTIVE=0
    if [[ -n "${JOB_ID}" && "${JOB_ID}" != "UNKNOWN" && -n "${ACTIVE_IDS[${JOB_ID}]+x}" ]]; then
        IS_ACTIVE=1
    fi
    if [[ ${IS_ACTIVE} -eq 0 && -n "${ACTIVE_NAMES[${JOB_NAME}]+x}" ]]; then
        IS_ACTIVE=1
    fi

    LOG_TS="$(date +%Y-%m-%dT%H:%M:%S%z)"
    if [[ ${IS_ACTIVE} -eq 1 ]]; then
        RUNNING=$((RUNNING + 1))
        echo "running ${JOB_NAME} (${JOB_ID})"
        echo "${LOG_TS},${JOB_NAME},${BATCH_NAME},${EXPERIMENT},${TRIAL_ID},running,${JOB_ID}," >> "${WATCHDOG_LOG}"
        continue
    fi

    if [[ -n "${JOB_ID}" && "${JOB_ID}" != "UNKNOWN" && -n "${COMPLETED_IDS[${JOB_ID}]+x}" ]]; then
        COMPLETED=$((COMPLETED + 1))
        echo "completed ${JOB_NAME} (${JOB_ID})"
        echo "${LOG_TS},${JOB_NAME},${BATCH_NAME},${EXPERIMENT},${TRIAL_ID},completed,${JOB_ID}," >> "${WATCHDOG_LOG}"
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
