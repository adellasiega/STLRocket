#!/bin/bash
#SBATCH --job-name=stl_ablation
#SBATCH --output=logs/slurm/ablation_%A_%a.out
#SBATCH --error=logs/slurm/ablation_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=2-00:00:00
#SBATCH --partition=Main

# Validation-only ablation (scripts/run_ablation.py) as a SLURM array.
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_ablation.sh                  # submit the whole array (size computed below)
#   DRY_RUN=1 bash scripts/run_ablation.sh        # print every task's command, submit nothing
#
# Each array task runs one (dataset, depth, until_weight) for RUNS_PER_TASK
# consecutive runs. N_FORMULAS is swept inside the task from a single feature build.
# seed = BASE_SEED + run, so a run has the same split and formulas in every task.
#
# After the array drains:
#   sacct -j <JOBID> -X --format=JobID,State,Elapsed,MaxRSS | grep -v COMPLETED
#   python scripts/plot_ablation.py results/ablation/full/*.csv

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/ablation/full"

# Swept across array tasks
DATASETS=(
    ArticularyWordRecognition AtrialFibrillation BasicMotions Cricket
    DuckDuckGeese EigenWorms Epilepsy EthanolConcentration ERing FaceDetection
    FingerMovements HandMovementDirection Handwriting Heartbeat Libras LSST
    MotorImagery NATOPS PenDigits PEMS-SF PhonemeSpectra RacketSports
    SelfRegulationSCP1 SelfRegulationSCP2 StandWalkJump UWaveGestureLibrary
)
DEPTHS=(1 2 3)
UNTIL_WEIGHTS=(0)
N_RUNS=10
RUNS_PER_TASK=2   # runs done sequentially in one task; raise it if the array exceeds the limit

# Swept inside each task (cheap: no extra feature builds)
N_FORMULAS=(100 1000 10000)

# Fixed
FIT_INTERCEPT=false
VAL_RATIO=0.25
CV=3
CUT_POINT=1.0
BASE_SEED=0

MAX_ARRAY_SIZE=1001   # cluster's MaxArraySize: task ids must be < this
# ===========================================================================

N_D=${#DATASETS[@]}
N_U=${#UNTIL_WEIGHTS[@]}
N_GROUPS=$(( (N_RUNS + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
N_TASKS=$(( N_D * N_GROUPS * N_U * ${#DEPTHS[@]} ))

task_command() {  # $1 = task id -> prints the python command
    local id=$1
    local dataset=${DATASETS[$(( id % N_D ))]}
    local group=$((               (id / N_D) % N_GROUPS ))
    local until=${UNTIL_WEIGHTS[$(( (id / (N_D * N_GROUPS)) % N_U ))]}
    local depth=${DEPTHS[$((        id / (N_D * N_GROUPS * N_U) ))]}
    local first=$(( group * RUNS_PER_TASK ))
    local last=$(( first + RUNS_PER_TASK - 1 < N_RUNS - 1 ? first + RUNS_PER_TASK - 1 : N_RUNS - 1 ))
    echo python -u scripts/run_ablation.py \
        --datasets "$dataset" --runs $(seq "$first" "$last") --depths "$depth" --until_weights "$until" \
        --n_formulas "${N_FORMULAS[@]}" --fit_intercept "$FIT_INTERCEPT" \
        --val_ratio "$VAL_RATIO" --cv "$CV" --cut_point "$CUT_POINT" \
        --base_seed "$BASE_SEED" --device cpu \
        --output "${OUT_DIR}/${dataset}_d${depth}_u${until}_r${first}-${last}.csv"
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
OUT=${CMD[-1]}

# BLAS/OpenMP and torch otherwise size their pools from the physical node core
# count, not this job's allocation, oversubscribing when tasks share a node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source "${VENV}/bin/activate"
cd "$PROJECT_DIR" || exit 1
rm -f "$OUT"  # run_ablation.py appends: a resubmitted task must not duplicate rows

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CMD[*]}"
"${CMD[@]}"
