#!/bin/bash
#SBATCH --job-name=stl_anchors
#SBATCH --output=logs/slurm/anchors_%A_%a.out
#SBATCH --error=logs/slurm/anchors_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G --time=2-00:00:00
#SBATCH --partition=Main

# STLRocket instance explanations against Anchors (scripts/run_anchors_comparison.py) as a
# SLURM array. Needs anchor-exp in the venv: pip install --no-deps anchor-exp lime
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_anchors_comparison.sh              # submit the whole array
#   DRY_RUN=1 bash scripts/run_anchors_comparison.sh    # print every task's command, submit nothing
#   ONLY=BasicMotions RUNS="0" bash scripts/run_anchors_comparison.sh   # smoke test
#
# Each array task runs one (dataset, run); results go to OUT_DIR/<dataset>_run<r>.*.
# A run whose .done.json exists is skipped, so resubmitting the array only redoes
# unfinished runs. seed = BASE_SEED + run (same model as scripts/run_explanations.sh).
# Anchors runs serially (about 1-5 s per series); anchor-exp preallocates 10^6 sample rows,
# about 1-3 GB with ~100 selected formulae, hence the memory request.
#
# After the array drains:
#   sacct -j <JOBID> -X --format=JobID,State,Elapsed,MaxRSS | grep -v COMPLETED
#   ls OUT_DIR/*.done.json | wc -l      # complete runs

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/anchors"

# The fast datasets (T < 1000) of the Until and correlation ablations
DATASETS=(
    ERing RacketSports BasicMotions Libras NATOPS
    FingerMovements DuckDuckGeese Epilepsy PEMS-SF UWaveGestureLibrary
)
RUNS=(${RUNS:-0 1 2 3 4})
[[ -n "$ONLY" ]] && DATASETS=($ONLY)

# Model
DEPTH=2
N_FORMULAS=10000
CV=3
CUT_POINT=1.0
BASE_SEED=0

# Explanations: ours with the defaults of stlrocket.instance_rules.RuleConfig, Anchors with
# the defaults of AnchorTabularExplainer.explain_instance; both at most N_MAX literals/predicates
N_MAX=3
CONFIDENCE=0.95
MIN_NEW_SHARE=0.05
ANCHOR_THRESHOLD=0.95
MAX_EXPLAINED=200     # series explained per split, the same for both methods

MAX_ARRAY_SIZE=1001   # cluster's MaxArraySize: task ids must be < this
# ===========================================================================

N_D=${#DATASETS[@]}
N_R=${#RUNS[@]}
N_TASKS=$(( N_D * N_R ))

task_command() {  # $1 = task id -> prints the python command
    local id=$1
    local dataset=${DATASETS[$(( id % N_D ))]}
    local run=${RUNS[$(( id / N_D ))]}
    echo python -u scripts/run_anchors_comparison.py \
        --datasets "$dataset" --runs "$run" --depth "$DEPTH" --n_formulas "$N_FORMULAS" \
        --cv "$CV" --cut_point "$CUT_POINT" --base_seed "$BASE_SEED" --device cpu \
        --n_max "$N_MAX" --confidence "$CONFIDENCE" --min_new_share "$MIN_NEW_SHARE" \
        --anchor_threshold "$ANCHOR_THRESHOLD" --max_explained "$MAX_EXPLAINED" \
        --out_dir "$OUT_DIR"
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
    # The task list depends on ONLY and RUNS: pass them to the tasks.
    sbatch --export=ALL,ONLY="$ONLY",RUNS="${RUNS[*]}" --array=0-$(( N_TASKS - 1 )) "$0"
    exit $?
fi

# ---- inside an array task ----
read -r -a CMD <<< "$(task_command "$SLURM_ARRAY_TASK_ID")"

# BLAS/OpenMP and torch otherwise size their pools from the physical node core
# count, not this job's allocation, oversubscribing when tasks share a node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source "${VENV}/bin/activate"
cd "$PROJECT_DIR" || exit 1

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CMD[*]}"
"${CMD[@]}"
