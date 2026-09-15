#!/bin/bash
#SBATCH --job-name=stlrocket
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-623  # 26 datasets x 4 n_formulas x 3 depth_max x 2 until_weight

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

N_FORMULAS_LIST=(10 100 1000 10000)
DEPTH_MAX_LIST=(1 2 3)
UNTIL_WEIGHT_LIST=(0 1)

N_DATASETS=${#DATASETS[@]}
N_FORMULAS_VALS=${#N_FORMULAS_LIST[@]}
N_DEPTH_VALS=${#DEPTH_MAX_LIST[@]}
N_UNTIL_VALS=${#UNTIL_WEIGHT_LIST[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
FORMULAS_IDX=$(( (SLURM_ARRAY_TASK_ID / N_DATASETS) % N_FORMULAS_VALS ))
DEPTH_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_DATASETS * N_FORMULAS_VALS)) % N_DEPTH_VALS ))
UNTIL_IDX=$(( SLURM_ARRAY_TASK_ID / (N_DATASETS * N_FORMULAS_VALS * N_DEPTH_VALS) ))

DATASET=${DATASETS[$DATASET_IDX]}
N_FORMULAS=${N_FORMULAS_LIST[$FORMULAS_IDX]}
DEPTH_MAX=${DEPTH_MAX_LIST[$DEPTH_IDX]}
UNTIL_WEIGHT=${UNTIL_WEIGHT_LIST[$UNTIL_IDX]}

source /share/ai-lab/adsiega/STLKernel/venv/bin/activate
RESULTS_DIR="/share/ai-lab/adsiega/STLRocket/results"
mkdir -p "$RESULTS_DIR" /share/ai-lab/adsiega/STLRocket/logs/slurm
export MPLBACKEND=Agg

# BLAS/OpenMP and torch otherwise size their pools from the PHYSICAL node core
# count, not this job's cgroup, which oversubscribes --cpus-per-task badly when
# several array tasks land on one node. Must be exported before python starts.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} n_formulas=${N_FORMULAS} depth_max=${DEPTH_MAX} until_weight=${UNTIL_WEIGHT}"

# Keep the run-loop budget below the 12h SLURM wall above. The budget is checked
# before each seed, so leave enough headroom for one more seed to finish and be
# written; a task that overruns the wall is killed and loses its unwritten rows.
python /share/ai-lab/adsiega/STLRocket/scripts/run_stlinear_stltree.py \
  --dataset       "$DATASET" \
  --budgets       "$N_FORMULAS" \
  --depths        "$DEPTH_MAX" \
  --until_weight  "$UNTIL_WEIGHT" \
  --config_budget 28800 \
  --output_dir    "$RESULTS_DIR"
