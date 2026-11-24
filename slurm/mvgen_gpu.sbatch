#!/bin/bash
# Run an MVGenMaster script on one A6000, inside its checkout and its own environment
# (external/MVGenMaster/.venv; see docs/findings.md, 2026-11-19, for the setup).
# The script path is relative to external/MVGenMaster, or absolute. OUT is exported so
# scripts can write under the repo's data/ folder.
#
#   SCRIPT=run_mvgen.py ARGS="--input_path demo/test1 --output_path $PWD/data/experiments/mvgen/demo_test1" \
#       sbatch slurm/mvgen_gpu.sbatch
#
#SBATCH --job-name=dtm_mvgen
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/mvgen_%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
cd "$REPO/external/MVGenMaster"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/external/MVGenMaster:${PYTHONPATH:-}"
SCRIPT=${SCRIPT:-run_mvgen.py}
ARGS=${ARGS:-}

echo "[mvgen] $(date) host=$(hostname) script=$SCRIPT args=$ARGS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python "$SCRIPT" $ARGS
echo "[mvgen] done $(date)"
