#!/bin/bash
# ROGE (stlrocket.roge_baseline, method "roge" of scripts/run_comparison.py) on 30 resamples of
# every dataset, as SLURM arrays of scripts/run_comparison.sh, into the same output directory as
# scripts/run_resample_comparison.sh (run r: resample r, seed r). Then:
#   python scripts/plot_comparison.py --res_dir results/comparison_resamples
#
# ROGE fits one model per class (one for binary data) in parallel over the task's CPUs, so the
# CPUs only help with many classes. Estimated cost per resample (ROGE's Python robustness loop,
# O(T) for F/G and O(T^2) for Until, plus ~2 s of GP-UCB per formula, ~216 formulae per class):
#   main      24 datasets, 4 CPUs, 2 days: under 3 h per task, except MotorImagery (~16 h, binary)
#             and EthanolConcentration (~5 h); ~2300 CPU-h for 30 resamples
#   phoneme   PhonemeSpectra, 39 classes: 16 CPUs, ~5 h per task; ~1900 CPU-h
#   eigen     EigenWorms (T = 17984): ~8 days per task on 5 CPUs, ~29000 CPU-h; not submitted by
#             default (PART=eigen, and a partition that allows the time limit)
#
# ROGE must be installed in the environment (VENV of run_comparison.sh):
#   pip install --no-deps --ignore-requires-python git+https://github.com/LogArtLab/roge@1391e33489c9ddc0a7b08f92be807349534b4cf6
#   (it declares Python >= 3.11 but runs on the 3.9 env; --no-deps leaves numpy/scipy/sklearn alone)
#
# Usage, from PROJECT_DIR (the environment variables of run_comparison.sh apply):
#   bash scripts/run_roge_comparison.sh                           # main and phoneme
#   PART=main bash scripts/run_roge_comparison.sh                 # one part: main, phoneme or eigen
#   ONLY=ERing RUNS="0" PART=main bash scripts/run_roge_comparison.sh   # smoke test
#   DRY_RUN=1 bash scripts/run_roge_comparison.sh                 # print the commands
#   MAIN_SBATCH="--time=3-00:00:00" ...                           # extra options per part:
#                                                                 # MAIN_SBATCH, PHONEME_SBATCH, EIGEN_SBATCH
# Results are per (dataset, resample) and complete runs are skipped, so resubmitting redoes only
# the unfinished ones.

PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
VENV="/share/ai-lab/adsiega/STLRocket/env"
export OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/comparison_resamples}"
export RESAMPLE=1
export RUNS="${RUNS:-$(seq -s ' ' 0 29)}"
export METHODS="roge"
PART="${PART:-default}"
HERE="$(dirname "$0")"

MAIN_DATASETS="ArticularyWordRecognition AtrialFibrillation BasicMotions Cricket
    DuckDuckGeese Epilepsy EthanolConcentration ERing FaceDetection
    FingerMovements HandMovementDirection Handwriting Heartbeat Libras LSST
    MotorImagery NATOPS PenDigits PEMS-SF RacketSports
    SelfRegulationSCP1 SelfRegulationSCP2 StandWalkJump UWaveGestureLibrary"

case "$PART" in default|main|phoneme|eigen) ;; *) echo "PART must be main, phoneme or eigen" >&2; exit 1 ;; esac

if [[ -z "$DRY_RUN" ]] && ! "${VENV}/bin/python" -c "import roge" 2>/dev/null; then
    echo "roge is not installed in ${VENV}; install it with:" >&2
    echo "  ${VENV}/bin/pip install --no-deps --ignore-requires-python git+https://github.com/LogArtLab/roge@1391e33489c9ddc0a7b08f92be807349534b4cf6" >&2
    exit 1
fi

# ONLY restricts every part to the given datasets; a part with none of them is skipped
ONLY_ALL="$ONLY"
submit() {  # $1 = datasets of the part, $2 = sbatch options
    local d sel=()
    for d in $1; do
        [[ -z "$ONLY_ALL" || " $(echo $ONLY_ALL) " == *" $d "* ]] && sel+=("$d")
    done
    (( ${#sel[@]} )) || return 0
    ONLY="${sel[*]}" SBATCH_EXTRA="$2" bash "$HERE/run_comparison.sh" || exit $?
}

if [[ "$PART" == "default" || "$PART" == "main" ]]; then
    submit "$MAIN_DATASETS" "--job-name=stl_roge --cpus-per-task=4 --time=2-00:00:00 ${MAIN_SBATCH:-}"
fi
if [[ "$PART" == "default" || "$PART" == "phoneme" ]]; then
    submit "PhonemeSpectra" "--job-name=stl_roge_ph --cpus-per-task=16 --time=2-00:00:00 ${PHONEME_SBATCH:-}"
fi
if [[ "$PART" == "eigen" ]]; then
    submit "EigenWorms" "--job-name=stl_roge_ew --cpus-per-task=5 --time=10-00:00:00 ${EIGEN_SBATCH:-}"
fi
