#!/bin/bash
#SBATCH --job-name=stl_tuning_check
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-9  # one dataset per array task

# Check whether gradient-tuning the assembled global explanation per class
# (objective="f1", stlrocket/tuning.py) improves held-out macro F1, across all
# 10 datasets with a completed until_weight=0 sweep at M=10000.
#
# Dataset list must stay in sync with run_explanations.sh / run_sweep_nfor_maxd_unt.sh.

DATASETS=(
    "BasicMotions"
    "Cricket"
    "DuckDuckGeese"
    "ERing"
    "Epilepsy"
    "HandMovementDirection"
    "Heartbeat"
    "RacketSports"
    "StandWalkJump"
    "UWaveGestureLibrary"
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

source /share/ai-lab/adsiega/STLKernel/venv/bin/activate
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
export MPLBACKEND=Agg

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET}"

# Each task writes its own row into a shared tuning_check_summary.csv; the
# script merges with whatever is already on disk, so concurrent array tasks
# accumulate rather than overwrite.
python -u "${PROJECT_DIR}/scripts/run_tuning_check.py" \
  --datasets       "$DATASET" \
  --results_table  "${PROJECT_DIR}/tables/results_long.csv" \
  --budget         10000 \
  --until_weight   0.0 \
  --output_dir     "${PROJECT_DIR}/results/tuning_check"
