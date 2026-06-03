from __future__ import annotations

import numpy as np

from .features import eval_robustness


def evaluate_local_explanation(
    phi_local,
    target_class,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[float, int]:
    rhos = eval_robustness(phi_local, X_tr)
    pos_mask = rhos > 0
    tot_positive = int(pos_mask.sum())
    if tot_positive == 0:
        return 0.0, 0
    true_positive = int((y_tr[pos_mask] == target_class).sum())
    return float(true_positive / tot_positive), true_positive


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

        results[cls] = {"f1": f1}

    f1s = [v["f1"] for v in results.values() if not np.isnan(v["f1"])]
    results["macro_avg"] = {"f1": float(np.mean(f1s)) if f1s else float("nan")}
    return results
