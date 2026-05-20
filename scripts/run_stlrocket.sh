#!/bin/bash
#SBATCH --job-name=stlrocket
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-89  # 10 datasets x 3 n_formulas x 3 depth_max

DATASETS=(
  "ArticularyWordRecognition"
  "AtrialFibrillation"
  "BasicMotions"
  "Cricket"
  "ERing"
  "Epilepsy"
  "EthanolConcentration"
  "HandMovementDirection"
  "Handwriting"
  "Libras"
)

N_FORMULAS_LIST=(500 1000 3000)
DEPTH_MAX_LIST=(2 3 4)

N_DATASETS=${#DATASETS[@]}
N_FORMULAS_VALS=${#N_FORMULAS_LIST[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
FORMULAS_IDX=$(( (SLURM_ARRAY_TASK_ID / N_DATASETS) % N_FORMULAS_VALS ))
DEPTH_IDX=$(( SLURM_ARRAY_TASK_ID / (N_DATASETS * N_FORMULAS_VALS) ))

DATASET=${DATASETS[$DATASET_IDX]}
N_FORMULAS=${N_FORMULAS_LIST[$FORMULAS_IDX]}
DEPTH_MAX=${DEPTH_MAX_LIST[$DEPTH_IDX]}

source /share/ai-lab/adsiega/STLKernel/venv/bin/activate
DATA_DIR="/share/ai-lab/adsiega/STLRocket/Multivariate_arff"
RESULTS_DIR="/share/ai-lab/adsiega/STLRocket/results"
export MPLBACKEND=Agg

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} n_formulas=${N_FORMULAS} depth_max=${DEPTH_MAX}"

python /share/ai-lab/adsiega/STLRocket/run_experiment.py \
  --dataset     "$DATASET" \
  --n_formulas  "$N_FORMULAS" \
  --depth_max   "$DEPTH_MAX" \
  --n_run       10 \
  --cv          5 \
  --output_dir  "$RESULTS_DIR"
