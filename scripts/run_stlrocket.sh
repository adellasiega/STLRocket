#!/bin/bash
#SBATCH --job-name=stlrocket
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-25  # n_datasets x n_formulas

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

N_FORMULAS_LIST=(100)

N_DATASETS=${#DATASETS[@]}
N_FORMULAS_VALS=${#N_FORMULAS_LIST[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
FORMULAS_IDX=$(( SLURM_ARRAY_TASK_ID / N_DATASETS ))

DATASET=${DATASETS[$DATASET_IDX]}
N_FORMULAS=${N_FORMULAS_LIST[$FORMULAS_IDX]}

source /share/ai-lab/adsiega/STLKernel/venv/bin/activate
DATA_DIR="/share/ai-lab/adsiega/STELIS/Multivariate_arff"
RESULTS_DIR="/share/ai-lab/adsiega/STLRocket/results"
export MPLBACKEND=Agg

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} n_formulas=${N_FORMULAS}"

python /share/ai-lab/adsiega/STLRocket/run_experiment.py \
  --dataset     "$DATASET" \
  --n_formulas  "$N_FORMULAS" \
  --output_dir  "$RESULTS_DIR"
