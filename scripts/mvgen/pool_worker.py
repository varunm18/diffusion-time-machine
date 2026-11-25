"""Run tasks from a shared pool file, alongside any number of other workers (one per GPU).

``slurm/mvgen_gpu_queue.sbatch`` gives every job its own task list, so a list written for a job that is
still waiting in the Slurm queue waits with it. This worker instead claims tasks from ONE pool shared by
all workers: whichever GPU is free takes the next line. Start it as a task of a queue job
(``<repo>/scripts/mvgen/pool_worker.py --pool <file>``) or with ``srun --overlap`` inside a running job.

Pool file: one task per line, ``<script> <arguments>`` (run with this Python, from the current
directory, as the queue job does). Lines may be appended while workers run; ``STOP`` ends a worker once
every line before it is claimed. A line is claimed by creating ``<pool>.claims/<hash>`` (``mkdir`` is
atomic on NFS too), so each line runs once; ``<hash>/done`` or ``<hash>/failed`` records how it ended and
``<hash>/owner`` who ran it. A claim without ``done`` or ``failed`` after its worker died (time limit,
pre-emption) is stale: ``--release-stale`` removes such claims of dead Slurm jobs so the task runs again.
A file ``<pool>.stop.<Slurm job id>`` makes that job's workers end after their current task (to free a GPU).

    python scripts/mvgen/pool_worker.py --pool data/queue/prep/pool.txt
    python scripts/mvgen/pool_worker.py --pool data/queue/prep/pool.txt --status
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def read_pool(pool: Path) -> list[str]:
    lines = pool.read_text(encoding="utf-8").splitlines() if pool.exists() else []
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def claim_dir(pool: Path, task: str) -> Path:
    return pool.with_name(pool.name + ".claims") / hashlib.sha1(task.encode()).hexdigest()[:16]


def owner() -> str:
    return f"{socket.gethostname()} pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', '-')}"


def log(message: str) -> None:
    print(f"[pool] {datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def run(pool: Path, idle_min: float, log_dir: Path) -> None:
    log(f"worker {owner()} on {pool}")
    idle_since = time.time()
    stop_file = pool.with_name(f"{pool.name}.stop.{os.environ.get('SLURM_JOB_ID', '-')}")
    while True:
        if stop_file.exists():          # ask the worker of one job to end after its current task
            log(f"{stop_file.name} found: stopping")
            return
        tasks = read_pool(pool)
        claimed = None
        for task in tasks:
            if task == "STOP":
                log("STOP reached")
                return
            folder = claim_dir(pool, task)
            folder.parent.mkdir(parents=True, exist_ok=True)
            try:
                folder.mkdir()
            except FileExistsError:
                continue
            (folder / "owner").write_text(owner() + "\n" + task + "\n")
            claimed = (task, folder)
            break
        if claimed is None:
            if time.time() - idle_since > idle_min * 60:
                log(f"nothing new for {idle_min:g} minutes")
                return
            time.sleep(20)
            continue
        task, folder = claimed
        out = log_dir / f"pool_{folder.name}.out"
        log(f"started {task} (log {out})")
        start = time.time()
        with open(out, "w") as handle:
            code = subprocess.call([sys.executable, *shlex.split(task)], stdout=handle, stderr=subprocess.STDOUT)
        (folder / ("done" if code == 0 else "failed")).write_text(f"exit {code} after {time.time() - start:.0f} s\n")
        log(f"{'finished' if code == 0 else f'FAILED (exit {code})'} in {time.time() - start:.0f} s: {task}")
        idle_since = time.time()


def job_alive(job: str) -> bool:
    if job == "-":
        return True             # not a Slurm worker: cannot tell, keep the claim
    state = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True).stdout
    return bool(state.strip())


def status(pool: Path, release_stale: bool, quiet: bool = False) -> dict[str, int]:
    """Count the pool's tasks by state (and print them unless ``quiet``); a STOP line counts as ``stop``."""
    counts = {"waiting": 0, "running": 0, "done": 0, "failed": 0, "stale": 0, "stop": 0}
    for task in read_pool(pool):
        if task == "STOP":
            counts["stop"] = 1
            continue
        folder = claim_dir(pool, task)
        if not folder.exists():
            counts["waiting"] += 1
        elif (folder / "done").exists():
            counts["done"] += 1
        elif (folder / "failed").exists():
            counts["failed"] += 1
            if not quiet:
                print("failed:", task)
        else:
            text = (folder / "owner").read_text() if (folder / "owner").exists() else ""
            job = text.split("job=")[1].split()[0] if "job=" in text else "-"
            if job_alive(job):
                counts["running"] += 1
            else:
                counts["stale"] += 1
                if not quiet:
                    print("stale:", task)
                if release_stale:
                    for item in folder.iterdir():
                        item.unlink()
                    folder.rmdir()
    if not quiet:
        print(counts)
    return counts


def wait_drained(pool: Path, poll: float = 60) -> None:
    """Return once the pool ends with STOP and none of its tasks waits or runs (stale claims are released)."""
    while True:
        counts = status(pool, release_stale=True, quiet=True)
        if counts["stop"] and counts["waiting"] == 0 and counts["running"] == 0:
            log(f"pool drained: {counts}")
            return
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--idle-min", type=float, default=180, help="stop after this long without a new task")
    parser.add_argument("--log-dir", type=Path, default=REPO / "data" / "logs" / "pool")
    parser.add_argument("--status", action="store_true", help="count waiting, running, done and failed tasks")
    parser.add_argument("--release-stale", action="store_true", help="with --status: free claims of dead jobs")
    parser.add_argument("--wait", action="store_true",
                        help="only wait until the pool has a STOP line and no task waits or runs")
    args = parser.parse_args()
    pool = args.pool.resolve()
    if args.status:
        status(pool, args.release_stale)
        return
    if args.wait:
        wait_drained(pool)
        return
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run(pool, args.idle_min, args.log_dir)


if __name__ == "__main__":
    main()
