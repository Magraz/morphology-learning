# `watchdog_jobs_v2.sh` usage

## What it does

`scripts/hpc/watchdog_jobs_v2.sh` checks expected jobs from a ledger file and:

- leaves jobs alone if they are currently active in `squeue` for your user
- skips jobs that already finished with `COMPLETED` in `sacct`
- resubmits missing jobs through a submission script (default: `run_trial_gpu.sh`)

It also writes an action log CSV under `scripts/hpc/job_tracking/`.

## Prerequisites

- Run on a SLURM environment where `squeue`, `sacct`, and `sbatch` are available.
- Ensure `scripts/hpc/job_tracking/` exists.
- Ensure a ledger exists at `scripts/hpc/job_tracking/<batch_name>_latest.csv`, or ensure at least one `*_latest.csv` exists in `scripts/hpc/job_tracking/`.

`experiment_set.sh` produces these ledgers automatically.

## Command syntax

```bash
bash scripts/hpc/watchdog_jobs_v2.sh [batch_name] [--dry-run] [--run-script PATH]
```

Options:

- `batch_name`: Optional batch name. Uses `job_tracking/<batch_name>_latest.csv`.
- Omit `batch_name`: Uses the newest `*_latest.csv` by modification time.
- `--dry-run`: Show what would be resubmitted, but do not submit anything.
- `--run-script PATH`: Override submission script (default `scripts/hpc/run_trial_gpu.sh`).
- `-h` / `--help`: Print help.

## Typical workflow

1. Preview actions safely:

```bash
bash scripts/hpc/watchdog_jobs_v2.sh my_batch --dry-run
```

2. If the preview looks correct, resubmit missing jobs:

```bash
bash scripts/hpc/watchdog_jobs_v2.sh my_batch
```

3. Use a different launcher if needed:

```bash
bash scripts/hpc/watchdog_jobs_v2.sh my_batch --run-script scripts/hpc/run_trial_cpu.sh
```

## Output you should expect

The script prints per-job actions like:

- `running <job_name> (<job_id>)`
- `completed <job_name> (<job_id>)`
- `would resubmit <job_name> (<job_id>)` (dry run)
- `resubmitted <job_name> -> <new_job_id>`

At the end it prints a summary with:

- ledger used
- actions log path
- total jobs
- running
- completed
- would_resubmit (dry-run) or resubmitted
- failed

## Watchdog action log

Each run writes:

`scripts/hpc/job_tracking/watchdog_<YYYYMMDD_HHMMSS>.csv`

Columns:

- `timestamp`
- `job_name`
- `batch_name`
- `experiment`
- `trial_id`
- `status` (`running`, `completed`, `would_resubmit`, `resubmitted`, `resubmit_failed`)
- `old_job_id`
- `new_job_id`
