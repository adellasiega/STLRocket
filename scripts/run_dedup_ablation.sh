#!/bin/bash
#SBATCH --job-name=stl_dedup
#SBATCH --output=logs/slurm/dedup_%A_%a.out
#SBATCH --error=logs/slurm/dedup_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=32G --time=1-00:00:00
#SBATCH --partition=Main

# Duplicate-column removal (scripts/run_dedup_ablation.py): lasso on all M columns vs on
# the columns left by correlation_filter at each DEDUP_THRESHOLD, same bank, split and CV
# folds. Measures the change in validation accuracy / predictions / selected size and the
# fit time saved.
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_dedup_ablation.sh               # submit the array
#   DRY_RUN=1 bash scripts/run_dedup_ablation.sh     # print every task's command, submit nothing
#
# Each array task runs one (dataset, depth) for RUNS_PER_TASK consecutive runs, all M and
# all thresholds. seed = BASE_SEED + run, so the arms are paired with each other and with
# the until_weight=0 arm of run_ablation.sh (same split and formula stream).
# Fit times are compared within a task (same node), so tasks sharing a node do not bias them.

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/ablation/dedup"

# The 10 fast datasets of the Until, correlation and first-difference ablations.
DATASETS=(
    ERing RacketSports BasicMotions Libras NATOPS
    FingerMovements DuckDuckGeese Epilepsy PEMS-SF UWaveGestureLibrary
)
DEPTHS=(1 2 3)
N_RUNS=10
RUNS_PER_TASK=2

# Swept inside each task (one feature build at max M)
N_FORMULAS=(1000 5000 10000)
# none = all columns (the reference). 0.999999: exact duplicates only (float noise is
# ~1e-12 in float64); 0.9999 and 0.999 add near-duplicates.
DEDUP_THRESHOLDS=(none 0.999999 0.9999 0.999)
FIT_REPEATS=3   # fit time = min over repeats

# Fixed (as in run_ablation.sh)
FIT_INTERCEPT=false
VAL_RATIO=0.25
CV=3
CUT_POINT=1.0
BASE_SEED=0
# ===========================================================================

N_D=${#DATASETS[@]}
N_DEPTH=${#DEPTHS[@]}
N_GROUPS=$(( (N_RUNS + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
N_TASKS=$(( N_D * N_DEPTH * N_GROUPS ))

task_command() {  # $1 = task id -> prints the python command
    local id=$1
    local dataset=${DATASETS[$(( id % N_D ))]}
    local depth=${DEPTHS[$(( id / N_D % N_DEPTH ))]}
    local group=$(( id / (N_D * N_DEPTH) ))
    local first=$(( group * RUNS_PER_TASK ))
    local last=$(( first + RUNS_PER_TASK - 1 < N_RUNS - 1 ? first + RUNS_PER_TASK - 1 : N_RUNS - 1 ))
    echo python -u scripts/run_dedup_ablation.py \
        --datasets "$dataset" --runs $(seq "$first" "$last") --depths "$depth" \
        --n_formulas "${N_FORMULAS[@]}" --dedup_thresholds "${DEDUP_THRESHOLDS[@]}" \
        --fit_repeats "$FIT_REPEATS" --fit_intercept "$FIT_INTERCEPT" \
        --val_ratio "$VAL_RATIO" --cv "$CV" --cut_point "$CUT_POINT" \
        --base_seed "$BASE_SEED" --device cpu \
        --output "${OUT_DIR}/${dataset}_d${depth}_r${first}-${last}.csv"
}

# ---- not inside SLURM: submit (or print) the array ----
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Datasets: ${DATASETS[*]}"
    if [[ -n "$DRY_RUN" ]]; then
        for (( i = 0; i < N_TASKS; i++ )); do echo "$i: $(task_command $i)"; done
        exit 0
    fi
    cd "$PROJECT_DIR" || exit 1
    mkdir -p logs/slurm "$OUT_DIR"
    sbatch --array=0-$(( N_TASKS - 1 )) "$0"
    exit $?
fi

# ---- inside an array task ----
read -r -a CMD <<< "$(task_command "$SLURM_ARRAY_TASK_ID")"
OUT=${CMD[-1]}

# BLAS/OpenMP and torch otherwise size their pools from the physical node core
# count, not this job's allocation, oversubscribing when tasks share a node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source "${VENV}/bin/activate"
cd "$PROJECT_DIR" || exit 1
rm -f "$OUT"  # run_dedup_ablation.py appends: a resubmitted task must not duplicate rows

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CMD[*]}"
"${CMD[@]}"
