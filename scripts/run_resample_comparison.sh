#!/bin/bash
# Test-set comparison of STLRocket, MultiRocket-Hydra, catch22, Z-Time and the STL decision
# tree of Bombara and Belta on 30 resamples of every dataset, as SLURM arrays of
# scripts/run_comparison.sh: run r is trained and tested on resample r (0: the official
# split; r > 0: stratified resample with the same train size and class counts), seed r.
# Then:
#   python scripts/plot_comparison.py --res_dir results/comparison_resamples
#
# Bombara is far slower than the other methods (hours per run on the largest datasets), so it
# goes in its own array, which does not hold back the others. Results are per (dataset,
# resample, method) and complete methods are skipped, so resubmitting redoes only the
# unfinished ones.
#
# Z-Time runs the authors' code: clone https://github.com/zedshape/ztime and set ZTIME_DIR
# to the checkout before submitting (the tasks inherit the environment).
#
# Usage, from PROJECT_DIR (the environment variables of run_comparison.sh apply):
#   ZTIME_DIR=/path/to/ztime bash scripts/run_resample_comparison.sh          # both arrays
#   PART=fast bash scripts/run_resample_comparison.sh     # only stlrocket, multirockethydra, catch22, ztime
#   PART=bombara bash scripts/run_resample_comparison.sh  # only bombara
#   DRY_RUN=1 bash scripts/run_resample_comparison.sh     # print the commands
#   ONLY=BasicMotions RUNS="0 1" bash scripts/run_resample_comparison.sh       # smoke test

PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
export OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/comparison_resamples}"
export RESAMPLE=1
export RUNS="${RUNS:-$(seq -s ' ' 0 29)}"
PART="${PART:-all}"
HERE="$(dirname "$0")"

if [[ -z "$ZTIME_DIR" && "$PART" != "bombara" && -z "$DRY_RUN" ]]; then
    echo "set ZTIME_DIR to a checkout of https://github.com/zedshape/ztime" >&2
    exit 1
fi

if [[ "$PART" == "all" || "$PART" == "fast" ]]; then
    METHODS="stlrocket multirockethydra catch22 ztime" bash "$HERE/run_comparison.sh" || exit $?
fi
if [[ "$PART" == "all" || "$PART" == "bombara" ]]; then
    METHODS="bombara" SBATCH_EXTRA="${BOMBARA_SBATCH:-}" bash "$HERE/run_comparison.sh" || exit $?
fi
