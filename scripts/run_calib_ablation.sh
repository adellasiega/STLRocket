#!/bin/bash
#SBATCH --job-name=stl_calib
#SBATCH --output=logs/slurm/calib_%A_%a.out
#SBATCH --error=logs/slurm/calib_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --time=1-00:00:00
#SBATCH --partition=Main

# Threshold-calibration ablation (scripts/run_ablation.py --calibrate): atom thresholds
# uniform in [min, max] vs calibrated on the fit part (features.calibrate_thresholds),
# at fixed M.
#
# Usage, from PROJECT_DIR:
#   bash scripts/run_calib_ablation.sh              # submit the array
#   DRY_RUN=1 bash scripts/run_calib_ablation.sh    # print every task's command, submit nothing
#
# Each array task runs one (dataset, depth) for RUNS_PER_TASK consecutive runs and both
# arms. seed = BASE_SEED + run, so arms are paired on the same split and formula stream:
# they differ only in the thresholds of multi-atom formulae.

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/ablation/calib"

# Fast and mid-size datasets with signal in the previous ablations. Left out: at chance
# (AtrialFibrillation, StandWalkJump, SelfRegulationSCP2, MotorImagery), too slow
# (PenDigits, LSST, PhonemeSpectra, EigenWorms, FaceDetection, EthanolConcentration),
# Handwriting (too few samples per class for the fit/val split).
DATASETS=(
    ERing DuckDuckGeese BasicMotions RacketSports HandMovementDirection FingerMovements
    Heartbeat Epilepsy UWaveGestureLibrary PEMS-SF SelfRegulationSCP1 Cricket NATOPS
    ArticularyWordRecognition Libras
)
DEPTHS=(3)                   # calibration only touches multi-atom formulae: 54% of them at depth 3
CALIBRATE=(false true)
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
        --calibrate "${CALIBRATE[@]}" --until_weights "$UNTIL_WEIGHT" \
        --n_formulas "$N_FORMULAS" --fit_intercept "$FIT_INTERCEPT" \
        --val_ratio "$VAL_RATIO" --cv "$CV" --cut_point "$CUT_POINT" \
        --base_seed "$BASE_SEED" --device cpu \
        --output "${OUT_DIR}/${dataset}_d${depth}_r${first}-${last}.csv"
}

# ---- not inside SLURM: submit (or print) the array ----
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    if [[ -n "$DRY_RUN" ]]; then
        for (( i = 0; i < N_TASKS; i++ )); do echo "$i: $(task_command $i)"; done
        exit 0
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
