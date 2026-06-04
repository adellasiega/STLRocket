from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    # Dataset
    dataset: str = "BasicMotions"

    # Feature extraction
    n_formulas: int = 100
    depth_max: int = 3
    batch_size: int = 200
    threshold_corr: float = 0.5
    max_iter: int = 100
    only_temporal: bool = True
    until_weight: float = 0.0

    # Classifier
    cv: int = 3

    # Explanation
    pool_size: int = 10
    precision_threshold: float = 0.75

    # Experiment loop
    n_run: int = 10
    base_seed: int = 0  # run i uses seed=base_seed+i

    # Explanations
    explain: bool = True

    # Output
    output_dir: str = "results"
