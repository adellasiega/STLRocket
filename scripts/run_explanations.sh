#!/bin/bash
#SBATCH --job-name=stl_explanations
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-9  # one dataset per array task

# Build local + global STL explanations for each dataset's best stl_linear
# configuration (M=10000, until_weight=0, depth_max read per dataset from
# tables/results_long.csv) via scripts/run_explanations.py.
#
# The dataset list is the 10 with a completed until_weight=0 sweep at M=10000 --
# i.e. exactly the rows make_table.py writes to tables/results_long.csv. It must
# stay in sync with DATASETS in run_sweep_nfor_maxd_unt.sh.

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

# Each task writes its own <output_dir>/<dataset>/ subdirectory; summary.csv is
# merged with whatever is already on disk, so concurrent tasks accumulate rather
# than overwrite. Run plot_explanations.py once the array has drained.
python -u "${PROJECT_DIR}/scripts/run_explanations.py" \
  --datasets       "$DATASET" \
  --results_table  "${PROJECT_DIR}/tables/results_long.csv" \
  --budget         10000 \
  --until_weight   0.0 \
  --output_dir     "${PROJECT_DIR}/results/explanations"
