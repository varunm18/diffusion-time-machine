#!/bin/bash
# Train on the Nexus *class* partition, which is what the capstone accounts can use.
# The repo's other job scripts target vulcan/scavenger and are rejected for us.
#
#   sbatch slurm/train_dated_seva_class.sbatch
#   squeue -u $USER          # watch it
#   tail -f logs/train-<jobid>.out
#
# The class QoS caps a job at 1 GPU / 4 CPUs / 32 GB / 3 days, so none of the
# resource lines below can be raised without the job being refused outright.

#SBATCH --job-name=dated-seva
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1     # 24 GB. rtxa4000 is 16 GB and too tight at 576px.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00           # well under the 3-day cap; raise once a run is proven
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.out

set -euo pipefail

# Home is capped at 30 GB, which a single checkpoint can exhaust, so weights,
# datasets and outputs all live in group storage. Override DATA_ROOT to move them.
DATA_ROOT="${DATA_ROOT:-/fs/class-projects/fall2026/$USER}"
export TIMEMACHINE_HF_HUB="${TIMEMACHINE_HF_HUB:-$DATA_ROOT/hf/hub}"
export HF_HOME="$DATA_ROOT/hf"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs "$DATA_ROOT/runs"

module purge
module load Python3/3.12.3
source .venv/bin/activate

echo "=== $(hostname) | job $SLURM_JOB_ID | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail here rather than three minutes into loading the weights. The smoke step is
# left out: this *is* the GPU, so a one-step rehearsal would only pay for itself
# twice, and a real failure shows up in the first logged step anyway.
python scripts/preflight.py \
    --dataset "$DATA_ROOT"/datasets/*_model0 \
    --image-root "$DATA_ROOT/images" \
    --no-smoke

python scripts/train_dated_seva.py \
    --dataset "$DATA_ROOT"/datasets/*_model0 \
    --out "$DATA_ROOT/runs/$SLURM_JOB_ID" \
    --date-injection film \
    --appearance-injection film \
    --appearance-remove-era \
    --resolution 576 \
    --n-targets 8 \
    --grad-checkpointing \
    --precision bf16

echo "=== finished $(date) ==="
