from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ExperimentConfig:
    # Dataset
    dataset: str = "BasicMotions"

    # Feature extraction
    n_formulas: int = 1000
    depth_max: int = 5
    batch_size: int = 200
    threshold_corr: float = 0.98
    max_iter: int = 20
    only_temporal: bool = True
    until_weight: float = 0.0

    # Classifier
    cv: int = 3
    max_iter_lr: int = 1000

    # Explanation
    pool_size: int = 10
    precision_threshold: float = 0.75

    # Experiment loop
    n_run: int = 5
    base_seed: int = 0  # run i uses seed=base_seed+i; extra run uses base_seed+n_run

    # Output
    output_dir: str = "results"
