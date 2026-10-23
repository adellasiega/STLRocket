from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Dataset
    dataset: str 

    # Feature extraction
    n_formulas: int
    depth_max: int 
    until_weight: float

    # Classifier
    cv: int
    cut_point: float      # glmnet: pick the sparsest lambda within cut_point SEs of the best CV score
    fit_intercept: bool

    # Explanations
    explain: bool

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

    # Output
    output_dir: str 

    # Hardware
    device: str

    # Set atom thresholds of multi-atom formulae from the train data
    # (features.calibrate_thresholds) instead of uniformly in [min, max]
    calibrate_thresholds: bool = False

    # Probability that a sampled atom is a moving-average atom Average[a,b](x ~ c)
    # (stlrocket/average.py); 0 = off
    average_prob: float = 0.0

    # Probability that a sampled atom is a two-variable linear atom a*x_i + b*x_j ~ c
    # (stlrocket/linear_atom.py); 0 = off
    cross_prob: float = 0.0
