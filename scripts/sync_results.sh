#!/bin/bash
# Fetch results and logs from the cluster (they are not versioned in git).
#
# Usage, from the repository root:
#   CLUSTER=user@login-host bash scripts/sync_results.sh             # results/ and logs/
#   bash scripts/sync_results.sh user@login-host                      # same, host as argument
#   bash scripts/sync_results.sh user@login-host --only results/anchors   # one experiment
#   bash scripts/sync_results.sh user@login-host --dry-run           # show what would change
#
# Incremental (rsync --update: only newer files are copied) and never deletes local files.

set -euo pipefail

REMOTE_DIR="/share/ai-lab/adsiega/STLRocket"
PATHS=(results logs)
DRY_RUN=""

if [[ $# -gt 0 && "$1" != --* ]]; then
    CLUSTER="$1"
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) PATHS=("$2"); shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
if [[ -z "${CLUSTER:-}" ]]; then
    echo "set the cluster login, e.g. CLUSTER=user@login-host, or pass it as the first argument" >&2
    exit 1
fi

for p in "${PATHS[@]}"; do
    mkdir -p "$(dirname "$p")"
    echo "== $CLUSTER:$REMOTE_DIR/$p -> ./$p"
    rsync -az --update --human-readable --info=stats1 $DRY_RUN "$CLUSTER:$REMOTE_DIR/$p/" "./$p/"
done
