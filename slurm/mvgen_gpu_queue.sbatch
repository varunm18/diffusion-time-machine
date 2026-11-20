#!/bin/bash
# Hold one A6000 and run MVGenMaster scripts from a task list, one after another.
#
# GPU jobs on this cluster can wait hours in the queue, and the gates of the date-conditioning plan
# are many short runs whose settings depend on the run before. This job therefore reads its tasks
# from a file that may grow while the job runs:
#
#   TASKS=$PWD/data/queue/a.txt sbatch --time=06:00:00 --job-name=dtm_queue_a slurm/mvgen_gpu_queue.sbatch
#   echo "$PWD/scripts/mvgen/train.py --run-name gate1a_post ..." >> data/queue/a.txt
#
# Task file: one task per line, "<script> <arguments>", run as in mvgen_gpu.sbatch (inside
# external/MVGenMaster with its environment, so give absolute paths). Empty lines and lines that
# start with "#" are skipped. Only ever APPEND lines: the job counts the tasks it has run. A line
# "STOP" ends the job. Give each job its own task file.
#
# The job frees its GPU when the list has had nothing new for IDLE_MIN minutes (default 15). A task
# that fails is reported and the job goes on to the next one. Each task logs to
# data/logs/queue_<job>_<task number>.out; this job's own log lists the tasks and how they ended.
#
#SBATCH --job-name=dtm_queue
#SBATCH --partition=vulcan-jbhuang,vulcan-ampere
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=data/logs/queue_%j.out
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
TASKS="${TASKS:?set TASKS to the task file (absolute path)}"
IDLE_MIN="${IDLE_MIN:-15}"
POLL_SECONDS=20

cd "$REPO/external/MVGenMaster" || exit 1
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/external/MVGenMaster:${PYTHONPATH:-}"

echo "[queue] $(date) host=$(hostname) tasks=$TASKS idle_min=$IDLE_MIN"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

n_done=0
idle=0
while true; do
    mapfile -t tasks < <(grep -v -E '^[[:space:]]*(#|$)' "$TASKS" 2>/dev/null)
    if (( n_done < ${#tasks[@]} )); then
        task="${tasks[$n_done]}"
        n_done=$((n_done + 1))
        idle=0
        if [[ "$task" == "STOP" ]]; then
            echo "[queue] $(date) STOP"
            break
        fi
        log="$REPO/data/logs/queue_${SLURM_JOB_ID}_$(printf '%02d' "$n_done").out"
        echo "[queue] $(date) task $n_done started: $task"
        # shellcheck disable=SC2086  # the task line is split into script and arguments on purpose
        if .venv/bin/python $task > "$log" 2>&1; then
            echo "[queue] $(date) task $n_done finished"
        else
            echo "[queue] $(date) task $n_done FAILED (exit $?), see $log"
        fi
    elif (( idle >= IDLE_MIN * 60 )); then
        echo "[queue] $(date) nothing new for $IDLE_MIN minutes"
        break
    else
        sleep "$POLL_SECONDS"
        idle=$((idle + POLL_SECONDS))
    fi
done
echo "[queue] done $(date), $n_done task lines read"
