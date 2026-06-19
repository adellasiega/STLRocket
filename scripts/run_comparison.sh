#!/bin/bash
#SBATCH --job-name=stl_comparison
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-25  # 26 datasets; budgets/depths/seeds/methods loop inside one process

DATASETS=(
    "ArticularyWordRecognition"
    "AtrialFibrillation"
    "BasicMotions"
    "Cricket"
    "DuckDuckGeese"
    "EigenWorms"
    "Epilepsy"
    "EthanolConcentration"
    "ERing"
    "FaceDetection"
    "FingerMovements"
    "HandMovementDirection"
    "Handwriting"
    "Heartbeat"
    "Libras"
    "LSST"
    "MotorImagery"
    "NATOPS"
    "PenDigits"
    "PEMS-SF"
    "PhonemeSpectra"
    "RacketSports"
    "SelfRegulationSCP1"
    "SelfRegulationSCP2"
    "StandWalkJump"
    "UWaveGestureLibrary"
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

source /share/ai-lab/adsiega/STLKernel/venv/bin/activate
RESULTS_DIR="/share/ai-lab/adsiega/STLRocket/results"
export MPLBACKEND=Agg

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET}"

# Per-job wall-clock bound: configs = len(budgets) + len(budgets)*len(depths)
#   = 4 + 4*3 = 16, each capped at --config_budget (default 1800s = 30min) => ~8h.
# --time above keeps headroom over that bound.
python -u /share/ai-lab/adsiega/STLRocket/run_comparison.py \
  --dataset       "$DATASET" \
  --budgets       "10,100,1000,10000" \
  --depths        "1,2,3" \
  --n_run         10 \
  --config_budget 1800 \
  --cv            3 \
  --output_dir    "$RESULTS_DIR/comparison"
