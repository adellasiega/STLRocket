#!/bin/bash
#SBATCH --job-name=stl_compare
#SBATCH --output=logs/slurm/compare_%A_%a.out
#SBATCH --error=logs/slurm/compare_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=32G --time=2-00:00:00
#SBATCH --partition=Main

# Test-set comparison of classifiers (scripts/run_comparison.py) as a SLURM array. Default
# methods: MiniRocket, MultiRocket+Hydra, MiniRocket features + the lasso of STLRocket, a
# decision tree and a ridge classifier on the STL features. STLRocket itself and MrSEQL are
# available with METHODS="stlrocket ..." (STLRocket's accuracy also comes from
# run_explanations.sh, which trains the same model). No explanations.
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_comparison.sh              # submit the whole array
#   DRY_RUN=1 bash scripts/run_comparison.sh    # print every task's command, submit nothing
#   ONLY=ERing RUNS="0" bash scripts/run_comparison.sh            # smoke test: one dataset, one run
#   METHODS="mrseql" bash scripts/run_comparison.sh               # only some methods
#
# Each array task runs every method on one (dataset, run); results go to
# OUT_DIR/<dataset>/run<r>/<method>.json. Methods whose json exists are skipped, so
# resubmitting the array only redoes unfinished ones. seed = BASE_SEED + run.
#
# After the array drains:
#   sacct -j <JOBID> -X --format=JobID,State,Elapsed,MaxRSS | grep -v COMPLETED
#   ls OUT_DIR/*/run*/*.json | wc -l      # complete (dataset, run, method) triples

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/comparison_balanced"

DATASETS=(
    ArticularyWordRecognition AtrialFibrillation BasicMotions Cricket
    DuckDuckGeese EigenWorms Epilepsy EthanolConcentration ERing FaceDetection
    FingerMovements HandMovementDirection Handwriting Heartbeat Libras LSST
    MotorImagery NATOPS PenDigits PEMS-SF PhonemeSpectra RacketSports
    SelfRegulationSCP1 SelfRegulationSCP2 StandWalkJump UWaveGestureLibrary
)
RUNS=(${RUNS:-0 1 2 3 4 5 6 7 8 9})
[[ -n "$ONLY" ]] && DATASETS=($ONLY)

# Model
DEPTH=2
N_FORMULAS=10000
CV=3
CUT_POINT=1.0
BASE_SEED=0

METHODS=(${METHODS:-minirocket multirockethydra minirocket_lasso stl_tree stl_ridge})

MAX_ARRAY_SIZE=1001   # cluster's MaxArraySize: task ids must be < this
# ===========================================================================

N_D=${#DATASETS[@]}
N_R=${#RUNS[@]}
N_TASKS=$(( N_D * N_R ))

task_command() {  # $1 = task id -> prints the python command
    local id=$1
    local dataset=${DATASETS[$(( id % N_D ))]}
    local run=${RUNS[$(( id / N_D ))]}
    echo python -u scripts/run_comparison.py \
        --datasets "$dataset" --runs "$run" --methods "${METHODS[@]}" --depth "$DEPTH" \
        --n_formulas "$N_FORMULAS" --cv "$CV" --cut_point "$CUT_POINT" --base_seed "$BASE_SEED" \
        --device cpu --out_dir "$OUT_DIR"
}

# ---- not inside SLURM: submit (or print) the array ----
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    if [[ -n "$DRY_RUN" ]]; then
        for (( i = 0; i < N_TASKS; i++ )); do echo "$i: $(task_command $i)"; done
        exit 0
    fi
    if (( N_TASKS > MAX_ARRAY_SIZE )); then
        echo "$N_TASKS tasks exceed MAX_ARRAY_SIZE=$MAX_ARRAY_SIZE" >&2
        exit 1
    fi
    mkdir -p logs/slurm "$OUT_DIR"
    # The task list depends on ONLY, RUNS and METHODS: pass them to the tasks.
    sbatch --export=ALL,ONLY="$ONLY",RUNS="${RUNS[*]}",METHODS="${METHODS[*]}" --array=0-$(( N_TASKS - 1 )) "$0"
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
