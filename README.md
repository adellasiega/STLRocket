# STLRocket

## Results

Git tracks only code. Experiments run on the cluster (the `scripts/run_*.sh` SLURM arrays) and
write to `results/` and `logs/`, which are ignored. Fetch them with

```
CLUSTER=user@login-host bash scripts/sync_results.sh             # results/ and logs/
bash scripts/sync_results.sh user@login-host --only results/anchors   # one experiment
```

The sync is incremental and never deletes local files. Figures and tables are then produced
locally by the `scripts/plot_*.py` scripts, and copied into the paper (a separate repository,
synced with Overleaf, cloned in `paper/`).
