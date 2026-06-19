#!/bin/bash
#SBATCH --job-name=stl_retrain_best
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-25  # one dataset per array task

# Retrain the STL linear model at the best (n_formulae, depth_max) configuration
# found by run_comparison.py for each dataset, then build local + global
# explanations (local_experiments/retrain_best.py).
#
# Each array task:
#   1. Maps its task id to a dataset.
#   2. Finds that dataset's most recent run_comparison.py output directory under
#      $COMPARISON_DIR (named "<dataset>_<timestamp>_<uid>").
#   3. Runs retrain_best.py against it; --dataset is omitted on purpose so the
#      dataset is read from the comparison config.json.

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
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
COMPARISON_DIR="${PROJECT_DIR}/results/comparison"
OUTPUT_DIR="${PROJECT_DIR}/results/explanations"
export MPLBACKEND=Agg

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET}"

# Most recent comparison run directory for this dataset.
RUN_DIR=$(ls -dt "${COMPARISON_DIR}/${DATASET}"_*/ 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: no comparison directory for ${DATASET} under ${COMPARISON_DIR}" >&2
    exit 1
fi
echo "  using comparison dir: ${RUN_DIR}"

python -u "${PROJECT_DIR}/local_experiments/retrain_best.py" \
  --comparison_dir "$RUN_DIR" \
  --output_dir     "${OUTPUT_DIR}/${DATASET}"
