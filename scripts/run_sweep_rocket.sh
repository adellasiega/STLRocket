#!/bin/bash
#SBATCH --job-name=rocket
#SBATCH --output=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.out
#SBATCH --error=/share/ai-lab/adsiega/STLRocket/logs/slurm/%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G --time=12:00:00
#SBATCH --partition=turing-long
#SBATCH --array=0-39  # 10 datasets x 4 n_kernels

# ROCKET baseline for comparison against STLRocket. Dataset list is kept in sync
# with run_sweep_nfor_maxd_unt.sh so the two sweeps cover the same ground; the
# STL-only depth_max and until_weight axes do not apply here, which is why this
# array is 40 tasks rather than 240.
DATASETS=(
    #"AtrialFibrillation"            #    1.5 min
    "StandWalkJump"                 #    2.2 min
    "ERing"                         #    2.3 min
    "BasicMotions"                  #    2.6 min
    "DuckDuckGeese"                 #    2.9 min
    "RacketSports"                  #    5.7 min
    "HandMovementDirection"         #    7.3 min
    "Heartbeat"                     #    7.7 min
    "UWaveGestureLibrary"           #    8.3 min
    "Epilepsy"                      #    8.4 min
    "Cricket"                       #    9.4 min
    # ---- excluded below this line; cost climbs steeply ----
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

# Matched to the STL formula-count budgets. ROCKET emits 2 features per kernel,
# so these are not equal column counts -- the comparable axis is the budget knob
# itself, plus the wall-clock each side spends to reach its accuracy.
N_KERNELS_LIST=(10 100 1000 10000)

N_DATASETS=${#DATASETS[@]}
N_KERNELS_VALS=${#N_KERNELS_LIST[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
KERNELS_IDX=$(( (SLURM_ARRAY_TASK_ID / N_DATASETS) % N_KERNELS_VALS ))

DATASET=${DATASETS[$DATASET_IDX]}
N_KERNELS=${N_KERNELS_LIST[$KERNELS_IDX]}

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

echo "Task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} n_kernels=${N_KERNELS}"

# Keep the run-loop budget below the 12h SLURM wall above. The budget is checked
# before each seed, so leave enough headroom for one more seed to finish and be
# written; a task that overruns the wall is killed and loses its unwritten rows.
python /share/ai-lab/adsiega/STLRocket/scripts/run_rocket.py \
  --dataset       "$DATASET" \
  --budgets       "$N_KERNELS" \
  --config_budget 28800 \
  --output_dir    "$RESULTS_DIR"
