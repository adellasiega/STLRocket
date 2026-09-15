from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Dataset
    dataset: str 

    # Feature extraction
    n_formulas: int
    depth_max: int 
    only_temporal: bool 
    until_weight: float

    # Classifier
    cv: int

    # Explanation
    pool_size: int 
    precision_threshold: float 

    # Global explanation simplification
    simplify_agreement: float 
    simplify_min_gain: float 
    simplify_decimals: int 

    # Experiment loop
    n_run: int 
    base_seed: int  # run i uses seed=base_seed+i

    # Explanations
    explain: bool

    # Output
    output_dir: str 

    # Hardware
    device: str
