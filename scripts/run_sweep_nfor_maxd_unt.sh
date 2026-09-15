#!/bin/bash
#SBATCH --job-name=stlrocket
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G --time=12:00:00
#SBATCH --partition=Main
#SBATCH --array=0-239  # 10 datasets x 4 n_formulas x 3 depth_max x 2 until_weight

# Fast subset: the 10 cheapest datasets by measured wall-clock from job 100751
# (d=1, uw=0 arm), summed over all four n_formulas budgets and extrapolated to
# 10 seeds. Together they are ~0.8 core-hours per (depth, until_weight) arm,
# versus ~128 for all 26. Commented-out entries are ordered by that same cost,
# so uncomment from the top down to trade runtime for coverage.
DATASETS=(
    "AtrialFibrillation"            #    1.5 min
    "StandWalkJump"                 #    2.2 min
    "ERing"                         #    2.3 min
    "BasicMotions"                  #    2.6 min
    "DuckDuckGeese"                 #    2.9 min
    "RacketSports"                  #    5.7 min
    "HandMovementDirection"         #    7.3 min
    "Heartbeat"                     #    7.7 min
    "UWaveGestureLibrary"           #    8.3 min
    "Epilepsy"                      #    8.4 min
    # ---- excluded below this line; cost climbs steeply ----
    # "Cricket"                     #    9.4 min
    # "PEMS-SF"                     #   10.1 min
    # "SelfRegulationSCP2"          #   10.6 min
    # "SelfRegulationSCP1"          #   11.2 min
    # "FingerMovements"             #   12.3 min
    # "NATOPS"                      #   12.5 min
    # "MotorImagery"                #   15.1 min
    # "Handwriting"                 #   17.0 min
    # "EigenWorms"                  #   17.4 min  OOM-killed at b=10000, 32G
    # "ArticularyWordRecognition"   #   42.7 min
    # "Libras"                      #   52.9 min
    # "EthanolConcentration"        #   80.7 min
    # "FaceDetection"               #  351.5 min
    # "PenDigits"                   #  946.9 min
    # "LSST"                        # 2828.8 min
    # "PhonemeSpectra"              # 3185.3 min
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
