from __future__ import annotations
import numpy as np
from .features import eval_robustness


def evaluate_local_explanation(
    phi_local,
    target_class,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[float, int, int]:
    """Precision, TP and FP of one local explanation on the train set."""
    rhos = eval_robustness(phi_local, X_tr)
    pos_mask = rhos > 0
    tot_positive = int(pos_mask.sum())
    if tot_positive == 0:
        return 0.0, 0, 0
    true_positive = int((y_tr[pos_mask] == target_class).sum())
    return float(true_positive / tot_positive), true_positive, tot_positive - true_positive


def evaluate_global(
    global_per_class: dict,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict:
    results = {}
    for cls, phi_global in global_per_class.items():
        rho = eval_robustness(phi_global, X_eval)
        target_mask = y_eval == cls
        pos_mask = rho > 0

        coverage = float((pos_mask & target_mask).sum() / target_mask.sum()) if target_mask.any() else float("nan")
        tp = int((pos_mask & target_mask).sum())
        total_pos = int(pos_mask.sum())
        precision = float(tp / total_pos) if total_pos > 0 else float("nan")
        f1 = float(2 * precision * coverage / (precision + coverage)) if (precision + coverage) > 0 else float("nan")

        # Share of the non-target instances the formula wrongly accepts. Reported
        # alongside precision because precision alone hides how much of the
        # negative pool was swept in when classes are imbalanced.
        n_neg = int((~target_mask).sum())
        fp_rate = float((pos_mask & ~target_mask).sum() / n_neg) if n_neg > 0 else float("nan")

        results[cls] = {
            "coverage": coverage,
            "precision": precision,
            "f1": f1,
            "fp_rate": fp_rate,
        }

    metrics = ["coverage", "precision", "f1", "fp_rate"]
    macro = {}
    for key in metrics:
        vals = [v[key] for v in results.values() if not np.isnan(v[key])]
        macro[key] = float(np.mean(vals)) if vals else float("nan")
    results["macro_avg"] = macro
    return results
