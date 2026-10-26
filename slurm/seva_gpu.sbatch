#!/bin/bash
# Run a SEVA script on one A6000. The script and its arguments come from SCRIPT / ARGS.
#
#   ARGS="--dataset data/datasets/25512_model0 --samples 8" sbatch slurm/seva_gpu.sbatch
#   SCRIPT=scripts/other.py ARGS="..." sbatch slurm/seva_gpu.sbatch
#
#SBATCH --job-name=dtm_seva
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/seva_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# Weights are read (never written) from a shared Hugging Face hub cache.
export TIMEMACHINE_HF_HUB=${TIMEMACHINE_HF_HUB:-/fs/nexus-projects/wildnvs/liger/weights/hf/hub}
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
SCRIPT=${SCRIPT:-scripts/seva_baseline.py}
ARGS=${ARGS:-}

echo "[seva] $(date) host=$(hostname) script=$SCRIPT args=$ARGS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python "$SCRIPT" $ARGS
echo "[seva] done $(date)"
