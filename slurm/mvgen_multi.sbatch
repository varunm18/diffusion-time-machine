#!/bin/bash
# Multi-scene training (scripts/mvgen/train_multi.py) on every GPU of one node, data parallel with
# torchrun, inside the MVGenMaster checkout and its environment. The run always continues from
# <RUN_DIR>/checkpoints/latest.pt when it exists (--resume), so the same command chains jobs past a
# time limit or a pre-emption. Pick the GPUs, partition and QOS on the command line:
#
#   RUN_DIR=$PWD/data/experiments/mvgen/multi/ms1 ARGS="--scenes-file $PWD/data/multiscene_plan/scenes_train.txt --look" \
#       sbatch --partition=vulcan-ampere --qos=vulcan-high-h200 --gres=gpu:h200-sxm:4 --cpus-per-task=32 \
#              --mem=256G --time=1-12:00:00 slurm/mvgen_multi.sbatch
#
#SBATCH --job-name=dtm_multi
#SBATCH --partition=vulcan-jbhuang,vulcan-ampere
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-high
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/multi_%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit this script with sbatch from the repository root}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the run folder (absolute path)}"
ARGS="${ARGS:-}"
cd "$REPO/external/MVGenMaster"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/external/MVGenMaster:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
GPUS=$(nvidia-smi -L | wc -l)
WORKERS=$(( (SLURM_CPUS_PER_TASK / GPUS) - 1 ))
(( WORKERS < 1 )) && WORKERS=1

echo "[multi] $(date) host=$(hostname) gpus=$GPUS loader workers per GPU=$WORKERS run=$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# shellcheck disable=SC2086  # ARGS is split into arguments on purpose
.venv/bin/torchrun --standalone --nproc_per_node "$GPUS" "$REPO/scripts/mvgen/train_multi.py" \
    --run-dir "$RUN_DIR" --resume --num-workers "$WORKERS" $ARGS
echo "[multi] done $(date)"
