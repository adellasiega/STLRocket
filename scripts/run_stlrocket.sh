#!/bin/bash
#SBATCH --job-name=stlrocket
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-26  # n_datasets x n_formulas x threshold_corr

DATASETS=(
#    "ArticularyWordRecognition"
#    "AtrialFibrillation"
#    "BasicMotions"
    "Cricket"
#    "DuckDuckGeese"
#    "EigenWorms"
#    "Epilepsy"
#    "EthanolConcentration"
#    "ERing"
#    "FaceDetection"
#    "FingerMovements"
#    "HandMovementDirection"
#    "Handwriting"
#    "Heartbeat"
#    "Libras"
#    "LSST"
#    "MotorImagery"
#    "NATOPS"
#    "PenDigits"
#    "PEMS-SF"
#    "PhonemeSpectra"
#    "RacketSports"
#    "SelfRegulationSCP1"
#    "SelfRegulationSCP2"
#    "StandWalkJump"
#    "UWaveGestureLibrary"
)

N_FORMULAS_LIST=(1 2 4 8 16 32 64 128 256)
THRESHOLD_CORR_LIST=(0.4 0.8 1.0)

N_DATASETS=${#DATASETS[@]}
N_FORMULAS_VALS=${#N_FORMULAS_LIST[@]}
N_THRESHOLD_VALS=${#THRESHOLD_CORR_LIST[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
FORMULAS_IDX=$(( (SLURM_ARRAY_TASK_ID / N_DATASETS) % N_FORMULAS_VALS ))
THRESHOLD_IDX=$(( SLURM_ARRAY_TASK_ID / (N_DATASETS * N_FORMULAS_VALS) ))

DATASET=${DATASETS[$DATASET_IDX]}
N_FORMULAS=${N_FORMULAS_LIST[$FORMULAS_IDX]}
THRESHOLD_CORR=${THRESHOLD_CORR_LIST[$THRESHOLD_IDX]}

source /share/ai-lab/adsiega/STLKernel/venv/bin/activate
DATA_DIR="/share/ai-lab/adsiega/STELIS/Multivariate_arff"
RESULTS_DIR="/share/ai-lab/adsiega/STLRocket/results"
export MPLBACKEND=Agg

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} n_formulas=${N_FORMULAS} n_formulas_freq=${N_FORMULAS} threshold_corr=${THRESHOLD_CORR}"

python /share/ai-lab/adsiega/STLRocket/run_experiment.py \
  --dataset          "$DATASET" \
  --n_formulas       "$N_FORMULAS" \
  --n_formulas_freq  "$N_FORMULAS" \
  --threshold_corr   "$THRESHOLD_CORR" \
  --output_dir       "$RESULTS_DIR" \
  --explain          False
