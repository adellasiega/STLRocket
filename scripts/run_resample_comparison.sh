#!/bin/bash
# Test-set comparison of STLRocket, MultiRocket-Hydra, catch22, Z-Time and the STL decision
# tree of Bombara and Belta on 30 resamples of every dataset, as SLURM arrays of
# scripts/run_comparison.sh: run r is trained and tested on resample r (0: the official
# split; r > 0: stratified resample with the same train size and class counts), seed r.
# Then:
#   python scripts/plot_comparison.py --res_dir results/comparison_resamples
#
# Three arrays, so the slow methods do not hold back the others:
#   fast     stlrocket, multirockethydra, catch22 (the SBATCH settings of run_comparison.sh)
#   ztime    Z-Time: the authors' implementation is single-threaded, so 1 CPU per task;
#            on the largest datasets it can take days (their own timings: 42 h for a single
#            configuration on PEMS-SF), so pass a longer limit with ZTIME_SBATCH
#   bombara  hours per run on the largest datasets; extra options with BOMBARA_SBATCH
# Results are per (dataset, resample, method) and complete methods are skipped, so
# resubmitting redoes only the unfinished ones.
#
# Z-Time runs the authors' code: clone https://github.com/zedshape/ztime and set ZTIME_DIR
# to the checkout before submitting (the tasks inherit the environment).
#
# Usage, from PROJECT_DIR (the environment variables of run_comparison.sh apply):
#   ZTIME_DIR=/path/to/ztime bash scripts/run_resample_comparison.sh          # all three arrays
#   PART=fast bash scripts/run_resample_comparison.sh                         # one array: fast, ztime or bombara
#   PART=ztime ZTIME_SBATCH="--time=7-00:00:00" bash scripts/run_resample_comparison.sh
#   DRY_RUN=1 bash scripts/run_resample_comparison.sh                         # print the commands
#   ONLY=BasicMotions RUNS="0 1" bash scripts/run_resample_comparison.sh      # smoke test
# The longest time limit of each partition: sinfo -o "%P %l"

PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
export OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/comparison_resamples}"
export RESAMPLE=1
export RUNS="${RUNS:-$(seq -s ' ' 0 29)}"
PART="${PART:-all}"
HERE="$(dirname "$0")"

case "$PART" in all|fast|ztime|bombara) ;; *) echo "PART must be all, fast, ztime or bombara" >&2; exit 1 ;; esac

if [[ -z "$ZTIME_DIR" && ( "$PART" == "all" || "$PART" == "ztime" ) && -z "$DRY_RUN" ]]; then
    echo "set ZTIME_DIR to a checkout of https://github.com/zedshape/ztime" >&2
    exit 1
fi

if [[ "$PART" == "all" || "$PART" == "fast" ]]; then
    METHODS="stlrocket multirockethydra catch22" bash "$HERE/run_comparison.sh" || exit $?
fi
if [[ "$PART" == "all" || "$PART" == "ztime" ]]; then
    METHODS="ztime" SBATCH_EXTRA="--job-name=stl_ztime --cpus-per-task=1 ${ZTIME_SBATCH:-}" \
        bash "$HERE/run_comparison.sh" || exit $?
fi
if [[ "$PART" == "all" || "$PART" == "bombara" ]]; then
    METHODS="bombara" SBATCH_EXTRA="--job-name=stl_bombara ${BOMBARA_SBATCH:-}" \
        bash "$HERE/run_comparison.sh" || exit $?
fi
