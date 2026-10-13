#!/bin/bash
#SBATCH --job-name=stl_expl_abl
#SBATCH --output=logs/slurm/expl_ablation_%A_%a.out
#SBATCH --error=logs/slurm/expl_ablation_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G --time=2-00:00:00
#SBATCH --partition=Main

# Validation-only ablation of the explanation method (scripts/run_explanation_ablation.py)
# as a SLURM array: candidate formulae x greedy generalization pass, and the check of the
# simplification, on the same model.
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_explanation_ablation.sh              # submit the whole array
#   DRY_RUN=1 bash scripts/run_explanation_ablation.sh    # print every task's command, submit nothing
#
# Each array task runs one dataset for RUNS_PER_TASK consecutive runs (seed = BASE_SEED + run).
# Complete runs are skipped, so resubmitting the array only redoes the unfinished ones.
#
# After the array drains:
#   sacct -j <JOBID> -X --format=JobID,State,Elapsed,MaxRSS | grep -v COMPLETED

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/ablation/explanations"

# The datasets of the main ablation, without Handwriting (too few series per class to split)
# and the three at chance (AtrialFibrillation, StandWalkJump, SelfRegulationSCP2)
DATASETS=(
    ArticularyWordRecognition BasicMotions Cricket DuckDuckGeese EigenWorms Epilepsy
    EthanolConcentration ERing FaceDetection FingerMovements HandMovementDirection Heartbeat
    Libras LSST MotorImagery NATOPS PenDigits PEMS-SF PhonemeSpectra RacketSports
    SelfRegulationSCP1 UWaveGestureLibrary
)
N_RUNS=10
RUNS_PER_TASK=2

# Fixed
DEPTH=2
N_FORMULAS=10000
VAL_RATIO=0.25
CV=3
CUT_POINT=1.0
MAX_EXPLAINED=1000    # series explained per split (larger splits are subsampled)
MAX_SIMPLIFIED=300    # rules whose simplification is checked
BASE_SEED=0

MAX_ARRAY_SIZE=1001   # cluster's MaxArraySize: task ids must be < this
# ===========================================================================

N_D=${#DATASETS[@]}
N_GROUPS=$(( (N_RUNS + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
N_TASKS=$(( N_D * N_GROUPS ))

task_command() {  # $1 = task id -> prints the python command
    local id=$1
    local dataset=${DATASETS[$(( id % N_D ))]}
    local first=$(( (id / N_D) * RUNS_PER_TASK ))
    local last=$(( first + RUNS_PER_TASK - 1 < N_RUNS - 1 ? first + RUNS_PER_TASK - 1 : N_RUNS - 1 ))
    echo python -u scripts/run_explanation_ablation.py \
        --datasets "$dataset" --runs $(seq "$first" "$last") \
        --depth "$DEPTH" --n_formulas "$N_FORMULAS" --val_ratio "$VAL_RATIO" --cv "$CV" --cut_point "$CUT_POINT" \
        --max_explained "$MAX_EXPLAINED" --max_simplified "$MAX_SIMPLIFIED" \
        --base_seed "$BASE_SEED" --device cpu --out_dir "$OUT_DIR"
}

# ---- not inside SLURM: submit (or print) the array ----
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    if [[ -n "$DRY_RUN" ]]; then
        for (( i = 0; i < N_TASKS; i++ )); do echo "$i: $(task_command $i)"; done
        exit 0
    fi
    if (( N_TASKS > MAX_ARRAY_SIZE )); then
        echo "$N_TASKS tasks exceed MAX_ARRAY_SIZE=$MAX_ARRAY_SIZE: increase RUNS_PER_TASK" >&2
        exit 1
    fi
    mkdir -p logs/slurm "$OUT_DIR"
    sbatch --array=0-$(( N_TASKS - 1 )) "$0"
    exit $?
fi

# ---- inside an array task ----
read -r -a CMD <<< "$(task_command "$SLURM_ARRAY_TASK_ID")"

# BLAS/OpenMP and torch otherwise size their pools from the physical node core
# count, not this job's allocation, oversubscribing when tasks share a node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source "${VENV}/bin/activate"
cd "$PROJECT_DIR" || exit 1

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CMD[*]}"
"${CMD[@]}"
