#!/bin/bash
# STLRocket vs the decision tree of STL primitives of Bombara and Belta 2021
# (stlrocket/bombara.py) on the 10 ablation datasets, 10 runs each, as a SLURM array of
# scripts/run_comparison.sh. Both methods run in the same task (same node, same 4 CPUs),
# so their times are comparable. Then:
#   python scripts/compare_bombara.py --res_dir results/comparison_bombara
#
# Usage, from PROJECT_DIR (the environment variables of run_comparison.sh apply):
#   bash scripts/run_bombara_comparison.sh                        # submit the array
#   DRY_RUN=1 bash scripts/run_bombara_comparison.sh              # print the commands
#   RUNS="0" bash scripts/run_bombara_comparison.sh               # one run per dataset
#
# Expected cost of bombara per run, from the root split's time with 8 threads (one run
# grows 3 CV trees to depth 5 and the refit): seconds to minutes on most datasets, hours on
# PEMS-SF (963 variables) and DuckDuckGeese (1345); run_comparison.sh allows 2 days.

PROJECT_DIR="/share/ai-lab/adsiega/STLRocket"
export ONLY="ERing RacketSports BasicMotions Libras NATOPS FingerMovements DuckDuckGeese Epilepsy PEMS-SF UWaveGestureLibrary"
export METHODS="stlrocket bombara"
export OUT_DIR="${PROJECT_DIR}/results/comparison_bombara"
exec bash "$(dirname "$0")/run_comparison.sh"
