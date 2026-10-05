#!/bin/bash
# Dated-MegaScenes census at scale, as a CPU job (the submission nodes are not for long jobs).
# Processes the TOP_N scenes with the largest COLMAP reconstructions (metadata, poses, dates),
# then rebuilds the census tables. Cached per scene, so a rerun only does missing work.
#
#   sbatch slurm/census_cpu.sbatch              # default TOP_N=800
#   TOP_N=2000 sbatch slurm/census_cpu.sbatch
#
#SBATCH --job-name=dtm_census
#SBATCH --partition=vulcan-cpu
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=data/logs/census_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
TOP_N=${TOP_N:-800}

echo "[census] $(date) host=$(hostname) top=$TOP_N"
.venv/bin/python scripts/process_scenes.py --top "$TOP_N" --workers 16
.venv/bin/python scripts/census.py
echo "[census] done $(date)"
