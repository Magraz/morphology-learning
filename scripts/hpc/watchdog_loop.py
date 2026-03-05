#!/usr/bin/env python3
"""Run watchdog_jobs_v2.sh on a fixed interval for one or more batches."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STOP_REQUESTED = False


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


def handle_signal(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f"Received signal {signum}; shutting down after current step.")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_watchdog = script_dir / "watchdog_jobs_v2.sh"

    parser = argparse.ArgumentParser(
        description=(
            "Run watchdog_jobs_v2.sh repeatedly for the given batch names. "
            "This is intended to run as a long-lived SLURM job."
        )
    )
    parser.add_argument(
        "batch_names",
        nargs="+",
        help="One or more batch names. Example: multi_box_push_12a_6o",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=30.0,
        help="Minutes between watchdog cycles (default: 30).",
    )
    parser.add_argument(
        "--watchdog-script",
        default=str(default_watchdog),
        help="Path to watchdog_jobs_v2.sh (default: scripts/hpc/watchdog_jobs_v2.sh).",
    )
    parser.add_argument(
        "--run-script",
        default=None,
        help="Optional run script passed through to watchdog with --run-script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to watchdog_jobs_v2.sh.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Optional cycle limit for testing. 0 means run forever.",
    )
    return parser.parse_args()


def sleep_with_interrupt(total_seconds: float) -> None:
    deadline = time.monotonic() + total_seconds
    while not STOP_REQUESTED:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(5.0, remaining))


def run_watchdog_for_batch(
    watchdog_script: Path, batch_name: str, run_script: str | None, dry_run: bool
) -> int:
    cmd = ["bash", str(watchdog_script), batch_name]
    if dry_run:
        cmd.append("--dry-run")
    if run_script:
        cmd.extend(["--run-script", run_script])

    log(f"Running batch '{batch_name}': {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    log(f"Batch '{batch_name}' finished with exit code {completed.returncode}")
    return completed.returncode


def main() -> int:
    args = parse_args()

    if args.interval_minutes <= 0:
        print("--interval-minutes must be > 0", file=sys.stderr)
        return 2

    watchdog_script = Path(args.watchdog_script).expanduser().resolve()
    if not watchdog_script.is_file():
        print(f"watchdog script not found: {watchdog_script}", file=sys.stderr)
        return 2

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    interval_seconds = args.interval_minutes * 60.0
    cycle = 0

    log(
        "Starting watchdog loop "
        f"(interval={args.interval_minutes} min, batches={args.batch_names})"
    )
    while not STOP_REQUESTED:
        cycle += 1
        log(f"Cycle {cycle} started")

        had_failure = False
        for batch_name in args.batch_names:
            if STOP_REQUESTED:
                break
            rc = run_watchdog_for_batch(
                watchdog_script=watchdog_script,
                batch_name=batch_name,
                run_script=args.run_script,
                dry_run=args.dry_run,
            )
            if rc != 0:
                had_failure = True

        if STOP_REQUESTED:
            break

        if had_failure:
            log("Cycle completed with at least one watchdog failure.")
        else:
            log("Cycle completed successfully.")

        if args.max_cycles > 0 and cycle >= args.max_cycles:
            log(f"Reached max cycles ({args.max_cycles}); exiting.")
            break

        log(f"Sleeping for {args.interval_minutes} minutes.")
        sleep_with_interrupt(interval_seconds)

    log("Watchdog loop stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
