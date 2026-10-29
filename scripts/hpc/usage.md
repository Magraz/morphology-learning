# `watchdog_jobs_v2.sh` usage

## What it does

`scripts/hpc/watchdog_jobs_v2.sh` checks expected jobs from a ledger file and:

- leaves jobs alone if they are currently active in `squeue` for your user
- skips jobs that already finished with `COMPLETED` in `sacct`
- resubmits missing jobs through a submission script (default: `run_trial_gpu.sh`)

A job counts as finished when `sacct` reports `COMPLETED` for either:

- the job ID recorded in the ledger, or
- any job with the same job name (`<trial>_<model>_<algorithm>_<batch>`)
  submitted since the ledger's earliest timestamp.

The name match is needed because the ledger is written once, at launch, and
keeps the original job ID. A job that hit `TIMEOUT` and was resubmitted
finishes under a new ID. Before the name match existed, such a job was
resubmitted every cycle forever: each rerun loaded the finished checkpoint,
did nothing and exited 0. The time bound means relaunching a batch with a new
ledger is not satisfied by an older run with the same names.

`COMPLETED` only means the job script exited 0, i.e. `train.py` exited 0. Any
other final state is resubmitted, including a config error that fails on every
attempt.

If either `sacct` query fails (for example the accounting database is down),
the watchdog exits with code 1 and resubmits nothing that cycle. Otherwise
every finished job would look unfinished and be resubmitted.

It also writes an action log CSV under `scripts/hpc/job_tracking/`.

## Prerequisites

- Run on a SLURM environment where `squeue`, `sacct`, and `sbatch` are available.
- Ensure `scripts/hpc/job_tracking/` exists.
- Ensure a ledger exists at `scripts/hpc/job_tracking/<algorithm>_<batch_name>_latest.csv`, or ensure at least one `*_latest.csv` exists in `scripts/hpc/job_tracking/`.

`experiment_set.sh` produces these ledgers automatically.

## Command syntax

```bash
bash scripts/hpc/watchdog_jobs_v2.sh [algorithm_batch_name] [--dry-run] [--run-script PATH]
```

Options:

- `algorithm_batch_name`: Optional identifier in the form `<algorithm>_<batch_name>`. Uses `job_tracking/<algorithm>_<batch_name>_latest.csv`.
- Omit `algorithm_batch_name`: Uses the newest `*_latest.csv` by modification time.
- `--dry-run`: Show what would be resubmitted, but do not submit anything.
- `--run-script PATH`: Override submission script (default `scripts/hpc/run_trial_gpu.sh`).
- `-h` / `--help`: Print help.

## Typical workflow

1. Preview actions safely:

```bash
bash scripts/hpc/watchdog_jobs_v2.sh mappo_jax_my_batch --dry-run
```

2. If the preview looks correct, resubmit missing jobs:

```bash
bash scripts/hpc/watchdog_jobs_v2.sh mappo_jax_my_batch
```

3. Use a different launcher if needed:

```bash
bash scripts/hpc/watchdog_jobs_v2.sh mappo_jax_my_batch --run-script scripts/hpc/run_trial_cpu.sh
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

## Node jobs (all trials in one job)

`experiment_batch_node.sh` trains a whole trial list in one job on a dgxh node
(`run_node_gpu.sh`, which runs `train_batches.sh` there). Its ledger,
`job_tracking/<algorithm>_node_<run name>_latest.csv`, has one row: the job
`0_node_<algorithm>_<run name>`. The watchdog treats that row like any other,
but must resubmit it with the node script:

```bash
bash scripts/hpc/watchdog_jobs_v2.sh rmappo_jax_node_rover --dry-run \
    --run-script scripts/hpc/run_node_gpu.sh
```

`experiment_batch_node.sh` starts its watchdog that way
(`run_watchdog.sh --run-script ... <identifier>`). The job counts as
`COMPLETED` only when every trial on the list has finished. A resubmitted job
runs the whole list again; finished trials load their checkpoint and exit.

## Watchdog action log

All watchdog cycles append to:

`scripts/hpc/job_tracking/watchdog_actions.csv`

Each row carries its own timestamp, so the full action history remains available
without creating a new CSV for every watchdog cycle.

Columns:

- `timestamp`
- `job_name`
- `batch_name`
- `experiment`
- `trial_id`
- `status` (`running`, `completed`, `would_resubmit`, `resubmitted`, `resubmit_failed`)
- `old_job_id` (the ledger's job ID)
- `new_job_id` (for `resubmitted`, the new job's ID; for `completed`, the ID
  of the job that completed, which differs from `old_job_id` when a
  resubmission finished it)
