#!/bin/bash
#SBATCH --job-name=stlrocket
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-52  # 26 datasets x n_formulas x depth_max x use_fourier

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

N_FORMULAS_LIST=(1000)
DEPTH_MAX_LIST=(2)
USE_FOURIER_LIST=(false true)

N_DATASETS=${#DATASETS[@]}
N_FORMULAS_VALS=${#N_FORMULAS_LIST[@]}
N_DEPTH_VALS=${#DEPTH_MAX_LIST[@]}
N_FOURIER_VALS=${#USE_FOURIER_LIST[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
FORMULAS_IDX=$(( (SLURM_ARRAY_TASK_ID / N_DATASETS) % N_FORMULAS_VALS ))
DEPTH_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_DATASETS * N_FORMULAS_VALS)) % N_DEPTH_VALS ))
FOURIER_IDX=$(( SLURM_ARRAY_TASK_ID / (N_DATASETS * N_FORMULAS_VALS * N_DEPTH_VALS) ))

DATASET=${DATASETS[$DATASET_IDX]}
N_FORMULAS=${N_FORMULAS_LIST[$FORMULAS_IDX]}
DEPTH_MAX=${DEPTH_MAX_LIST[$DEPTH_IDX]}
USE_FOURIER=${USE_FOURIER_LIST[$FOURIER_IDX]}

source /share/ai-lab/adsiega/STLKernel/venv/bin/activate
RESULTS_DIR="/share/ai-lab/adsiega/STLRocket/results"
export MPLBACKEND=Agg

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} n_formulas=${N_FORMULAS} depth_max=${DEPTH_MAX} use_fourier=${USE_FOURIER}"

python /share/ai-lab/adsiega/STLRocket/run_experiment.py \
  --dataset      "$DATASET" \
  --n_formulas   "$N_FORMULAS" \
  --depth_max    "$DEPTH_MAX" \
  --use_fourier  "$USE_FOURIER" \
  --output_dir   "$RESULTS_DIR" \
  --explain      false
