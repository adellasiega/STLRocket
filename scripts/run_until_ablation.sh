#!/bin/bash
#SBATCH --job-name=stl_until
#SBATCH --output=logs/slurm/until_%A_%a.out
#SBATCH --error=logs/slurm/until_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=1-00:00:00
#SBATCH --partition=Main

# Until vs no Until (scripts/run_ablation.py --until_weights) at fixed depth and M, on
# the datasets where Until features are fast to compute.
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_until_ablation.sh               # submit the array
#   DRY_RUN=1 bash scripts/run_until_ablation.sh     # print every task's command, submit nothing
#
# Each array task runs one dataset for RUNS_PER_TASK consecutive runs and both until
# weights. seed = BASE_SEED + run, so the two arms are paired on the same fit/validation
# split (their formula banks differ: the sampler draws other operators).

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/ablation/until_d2"

# The 10 datasets with the lowest time per run (Until feature build + lasso fit) at M=10000 in
# results/old_scheme/ablation/full_until_d1 (10 to 170 s at depth 1), leaving out
# AtrialFibrillation, which is at chance in every ablation.
DATASETS=(
    ERing RacketSports BasicMotions Libras NATOPS
    FingerMovements DuckDuckGeese Epilepsy PEMS-SF UWaveGestureLibrary
)
UNTIL_WEIGHTS=(0 1)
DEPTH=2
N_FORMULAS=10000
N_RUNS=10
RUNS_PER_TASK=2

# Fixed (as in run_ablation.sh)
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
        --until_weights "${UNTIL_WEIGHTS[@]}" \
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
rm -f "$OUT" "${OUT%.csv}.paths.jsonl"  # run_ablation.py appends: a resubmitted task must not duplicate rows

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CMD[*]}"
"${CMD[@]}"
