#!/bin/bash
#SBATCH --job-name=stl_corr
#SBATCH --output=logs/slurm/corr_%A_%a.out
#SBATCH --error=logs/slurm/corr_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=1-00:00:00
#SBATCH --partition=Main

# Correlation-filter ablation (scripts/run_ablation.py --corr_thresholds) at fixed depth
# and M, on the datasets of the Until ablation (scripts/run_until_ablation.sh).
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_corr_ablation.sh               # submit the array
#   DRY_RUN=1 bash scripts/run_corr_ablation.sh     # print every task's command, submit nothing
#
# Each array task runs one dataset for RUNS_PER_TASK consecutive runs and all thresholds.
# The unfiltered arm is not run here: it is the until_weight=0 arm of the Until ablation
# (results/ablation/until_d2), which has the same datasets, seeds, depth, M and classifier
# settings. seed = BASE_SEED + run, so every threshold is paired with it on the same
# fit/validation split and formula stream (the filtered bank is that stream, filtered).

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/ablation/corr_d2"

# Keep in sync with run_until_ablation.sh
DATASETS=(
    ERing RacketSports BasicMotions Libras NATOPS
    FingerMovements DuckDuckGeese Epilepsy PEMS-SF UWaveGestureLibrary
)
CORR_THRESHOLDS=(0.99 0.95)
# The filter resamples N_FORMULAS at a time until N_FORMULAS are kept, for at most this
# many rounds. With the default 20 some datasets already ran out at 0.95.
CORR_MAX_ROUNDS=100
DEPTH=2
N_FORMULAS=10000
N_RUNS=10
RUNS_PER_TASK=2

# Fixed (as in run_until_ablation.sh)
UNTIL_WEIGHT=0
FIT_INTERCEPT=false
VAL_RATIO=0.25
CV=3
CUT_POINT=1.0
BASE_SEED=0
# ===========================================================================

N_D=${#DATASETS[@]}
N_GROUPS=$(( (N_RUNS + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
N_TASKS=$(( N_D * N_GROUPS ))

task_command() {  # $1 = task id -> prints the python command
    local id=$1
    local dataset=${DATASETS[$(( id % N_D ))]}
    local group=$(( id / N_D ))
    local first=$(( group * RUNS_PER_TASK ))
    local last=$(( first + RUNS_PER_TASK - 1 < N_RUNS - 1 ? first + RUNS_PER_TASK - 1 : N_RUNS - 1 ))
    echo python -u scripts/run_ablation.py \
        --datasets "$dataset" --runs $(seq "$first" "$last") --depths "$DEPTH" \
        --corr_thresholds "${CORR_THRESHOLDS[@]}" --corr_max_rounds "$CORR_MAX_ROUNDS" \
        --until_weights "$UNTIL_WEIGHT" \
        --n_formulas "$N_FORMULAS" --fit_intercept "$FIT_INTERCEPT" \
        --val_ratio "$VAL_RATIO" --cv "$CV" --cut_point "$CUT_POINT" \
        --base_seed "$BASE_SEED" --device cpu \
        --output "${OUT_DIR}/${dataset}_d${DEPTH}_r${first}-${last}.csv"
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source "${VENV}/bin/activate"
cd "$PROJECT_DIR" || exit 1
rm -f "$OUT"  # run_ablation.py appends: a resubmitted task must not duplicate rows

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CMD[*]}"
"${CMD[@]}"
