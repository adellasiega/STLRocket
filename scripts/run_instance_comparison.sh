#!/bin/bash
#SBATCH --job-name=stl_instcmp
#SBATCH --output=logs/slurm/instcmp_%A_%a.out
#SBATCH --error=logs/slurm/instcmp_%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=32G --time=2-00:00:00
#SBATCH --partition=Main

# STLRocket instance and global explanations against AXp, inflated AXp, LORE, Anchors, a
# surrogate tree and RIPPER (scripts/run_instance_comparison.py) as a SLURM array. Needs in the
# venv: pip install --no-deps anchor-exp lime && pip install lore-sa wittgenstein
#
# Usage, from PROJECT_DIR:
#   SET=fast bash scripts/run_instance_comparison.sh     # every method but LORE and AXp on the FAST datasets
#   SET=rest bash scripts/run_instance_comparison.sh     # the others, without LORE and Anchors
#   DRY_RUN=1 bash scripts/run_instance_comparison.sh    # print every task's command, submit nothing
#   ONLY=BasicMotions RUNS="0" bash scripts/run_instance_comparison.sh   # smoke test
#
# Each array task runs one (dataset, run); results go to OUT_DIR/<dataset>_run<r>.*.
# A run whose .done.json exists is skipped, so resubmitting the array only redoes
# unfinished runs. seed = BASE_SEED + run (same model as scripts/run_explanations.sh).
# Every method explains every train and test series. LORE and Anchors run in one process per CPU; per series LORE takes about 8 s with 22 selected formulae,
# 90 s with 500 and 1000 s with 2000, and Anchors grows about as the square of the number of
# selected formulae (4 s with 17, 114 s with 73). A series not explained within its timeout is
# recorded as such, and after MAX_TIMEOUTS consecutive timeouts the method stops for the run.
# Anchors' processes are bounded so that its samples fit in ANCHOR_MEM_GB. Ours, AXp and
# inflated AXp take milliseconds per series; RIPPER a few minutes per class on the largest.
#
# After the array drains:
#   sacct -j <JOBID> -X --format=JobID,State,Elapsed,MaxRSS | grep -v COMPLETED
#   ls OUT_DIR/*.done.json | wc -l      # complete runs

# ============================== configuration ==============================
PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
OUT_DIR="${PROJECT_DIR}/results/instance_comparison"

# The 26 datasets and 10 runs of scripts/run_explanations.sh, in two sets: FAST, where Anchors
# explains every series within a few hours per run (few selected formulae, few series), and
# REST, where only the other methods run (Anchors would need days per run). LORE is not run: it
# explains a black box without its training data, on a synthetic neighbourhood, so its rules
# are not meant to hold on the data (the run script still supports it). Of the abductive
# explanations only the inflated AXp is kept: it has the features of a subset-minimal AXp and
# the same guarantee on a larger region.
# Both write to OUT_DIR: their datasets differ, so their files do not collide.
FAST=(
    BasicMotions DuckDuckGeese EigenWorms Epilepsy ERing
    FingerMovements NATOPS RacketSports SelfRegulationSCP1 UWaveGestureLibrary
)
REST=(
    ArticularyWordRecognition AtrialFibrillation Cricket EthanolConcentration FaceDetection
    HandMovementDirection Handwriting Heartbeat Libras LSST MotorImagery PenDigits PEMS-SF
    PhonemeSpectra SelfRegulationSCP2 StandWalkJump
)
case "$SET" in
    fast) DATASETS=("${FAST[@]}"); SKIP="lore axp" ;;
    rest) DATASETS=("${REST[@]}"); SKIP="lore anchors axp" ;;
    *) echo "SET must be fast or rest" >&2; exit 1 ;;
esac
RUNS=(${RUNS:-0 1 2 3 4 5 6 7 8 9})
[[ -n "$ONLY" ]] && DATASETS=($ONLY)

# Model
DEPTH=2
N_FORMULAS=10000
CV=3
CUT_POINT=1.0
BASE_SEED=0

# Explanations: ours with the defaults of stlrocket.instance_rules.RuleConfig, Anchors with
# the defaults of AnchorTabularExplainer.explain_instance, both at most N_MAX
# literals/predicates; LORE with the defaults of lore_sa.TabularGeneticGeneratorLore
N_MAX=3
CONFIDENCE=0.95
MIN_NEW_SHARE=0.05
ANCHOR_THRESHOLD=0.95
ANCHOR_TIMEOUT=300         # seconds per series
LORE_TIMEOUT=600
MAX_TIMEOUTS=8
ANCHOR_MEM_GB=24
# if LORE and Anchors run on the models with thousands of series and hundreds of selected
# formulae (ONLY=... SET=fast), they explain only SAMPLED_MAX_EXPLAINED series per split there
BIG=" FaceDetection LSST PenDigits PhonemeSpectra "
SAMPLED_MAX_EXPLAINED=500

MAX_ARRAY_SIZE=1001   # cluster's MaxArraySize: task ids must be < this
# ===========================================================================

N_D=${#DATASETS[@]}
N_R=${#RUNS[@]}
N_TASKS=$(( N_D * N_R ))

task_command() {  # $1 = task id -> prints the python command
    local id=$1
    local dataset=${DATASETS[$(( id % N_D ))]}
    local run=${RUNS[$(( id / N_D ))]}
    echo python -u scripts/run_instance_comparison.py \
        --datasets "$dataset" --runs "$run" --depth "$DEPTH" --n_formulas "$N_FORMULAS" \
        --cv "$CV" --cut_point "$CUT_POINT" --base_seed "$BASE_SEED" --device cpu \
        --n_max "$N_MAX" --confidence "$CONFIDENCE" --min_new_share "$MIN_NEW_SHARE" \
        --anchor_threshold "$ANCHOR_THRESHOLD" --anchor_timeout "$ANCHOR_TIMEOUT" \
        --lore_timeout "$LORE_TIMEOUT" --max_timeouts "$MAX_TIMEOUTS" --anchor_mem_gb "$ANCHOR_MEM_GB" \
        $([[ "$BIG" == *" $dataset "* ]] && echo --sampled_max_explained "$SAMPLED_MAX_EXPLAINED") \
        $([[ -n "$SKIP" ]] && echo --skip $SKIP) \
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
    sbatch --export=ALL,SET="$SET",ONLY="$ONLY",RUNS="${RUNS[*]}" --array=0-$(( N_TASKS - 1 )) "$0"
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
