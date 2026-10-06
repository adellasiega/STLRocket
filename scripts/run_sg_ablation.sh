#!/bin/bash
#SBATCH --job-name=stl_sg
#SBATCH --output=logs/slurm/sg_%A_%a.out
#SBATCH --error=logs/slurm/sg_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=1-00:00:00
#SBATCH --partition=turing-wide

# Savitzky-Golay derivative ablation (scripts/run_ablation.py --sg_windows): raw channels vs
# raw + SG first derivative, at fixed M on
# the N_DATASETS fastest datasets of the previous ablation (scripts/fastest_datasets.py).
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_sg_ablation.sh               # submit the array
#   DRY_RUN=1 bash scripts/run_sg_ablation.sh     # print every task's command, submit nothing
#
# Each array task runs one (dataset, depth) for RUNS_PER_TASK consecutive runs and all
# SG arms. seed = BASE_SEED + run, so arms are paired on the same split and
# formula stream.

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/ablation/sg"
PREV_RESULTS="${PROJECT_DIR}/results/ablation/full"

N_DATASETS=10                         # all but the 3 slowest (PenDigits, LSST, PhonemeSpectra)
DEPTHS=(3)
SG_WINDOWS=(none 0.05)               # window as a fraction of T; none = raw only
N_FORMULAS=10000
N_RUNS=10
RUNS_PER_TASK=2

# Fixed (as in run_ablation.sh)
UNTIL_WEIGHT=0
FIT_INTERCEPT=false
VAL_RATIO=0.25
CV=3
CUT_POINT=1.0
BASE_SEED=0
# ===========================================================================

cd "$PROJECT_DIR" || exit 1
source "${VENV}/bin/activate"

# Selected once at submission and passed to the tasks, so every task sees the same list.
if [[ -z "$DATASETS_STR" ]]; then
    DATASETS_STR=$(python scripts/fastest_datasets.py "$PREV_RESULTS"/*.csv --n "$N_DATASETS" --n_formulas "$N_FORMULAS") || exit 1
fi
read -r -a DATASETS <<< "$DATASETS_STR"
N_D=${#DATASETS[@]}
N_GROUPS=$(( (N_RUNS + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
N_TASKS=$(( N_D * N_GROUPS * ${#DEPTHS[@]} ))

task_command() {  # $1 = task id -> prints the python command
    local id=$1
    local dataset=${DATASETS[$(( id % N_D ))]}
    local group=$((               (id / N_D) % N_GROUPS ))
    local depth=${DEPTHS[$((        id / (N_D * N_GROUPS) ))]}
    local first=$(( group * RUNS_PER_TASK ))
    local last=$(( first + RUNS_PER_TASK - 1 < N_RUNS - 1 ? first + RUNS_PER_TASK - 1 : N_RUNS - 1 ))
    echo python -u scripts/run_ablation.py \
        --datasets "$dataset" --runs $(seq "$first" "$last") --depths "$depth" \
        --sg_windows "${SG_WINDOWS[@]}" --until_weights "$UNTIL_WEIGHT" \
        --n_formulas "$N_FORMULAS" --fit_intercept "$FIT_INTERCEPT" \
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
    mkdir -p logs/slurm "$OUT_DIR"
    sbatch --array=0-$(( N_TASKS - 1 )) --export=ALL,DATASETS_STR="$DATASETS_STR" "$0"
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

rm -f "$OUT"  # run_ablation.py appends: a resubmitted task must not duplicate rows

echo "Task ${SLURM_ARRAY_TASK_ID}: ${CMD[*]}"
"${CMD[@]}"
