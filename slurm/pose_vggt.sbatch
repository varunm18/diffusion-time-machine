#!/bin/bash
# Pose old photos of one scene with VGGT-Omega on one A6000 (see scripts/pose_with_vggt.py).
# MODE=eval re-poses old photos COLMAP already registered and measures the error;
# MODE=register poses the old photos COLMAP missed.
#
#   sbatch slurm/pose_vggt.sbatch                              # Notre-Dame model 0, eval
#   MODE=register sbatch slurm/pose_vggt.sbatch
#   SCENE=216687 MODEL=0 sbatch slurm/pose_vggt.sbatch         # Dam Square
#
#SBATCH --job-name=dtm_vggt
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/vggt_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# VGGT-Omega is used read-only from an existing checkout; never write bytecode into it.
export VGGT_OMEGA_ROOT=${VGGT_OMEGA_ROOT:-/vulcanscratch/ltahboub/vggt-omega}
export PYTHONDONTWRITEBYTECODE=1
SCENE=${SCENE:-25512}
MODEL=${MODEL:-0}
MODE=${MODE:-eval}

echo "[vggt] $(date) host=$(hostname) scene=$SCENE model=$MODEL mode=$MODE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable (driver too old for this torch build?)'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
.venv/bin/python scripts/pose_with_vggt.py --scene "$SCENE" --model "$MODEL" --mode "$MODE"
echo "[vggt] done $(date)"
