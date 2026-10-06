"""What do the selected formulae look like, compared with the whole bank?

For each dataset and run: fit split as in scripts/run_ablation.py, depth 3, M=10000,
lasso as in train_classifier. Compares selected formulae with the bank by number of
atoms, root operator and window width of the root, and reports the share of |coef| mass.
For selected formulae with a Boolean root, also checks whether they behave like one of
their two children (|corr| > 0.95 on the fit part).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torcheck import stl

from run_ablation import load_train
from stlrocket.config import ExperimentConfig
from stlrocket.features import build_formula_bank, extract_features, _atoms
from stlrocket.classifier import train_classifier, evaluate_classifier

DATASETS = ["NATOPS"]
RUNS = [0]


def describe(phi, T):
    root = type(phi).__name__
    if isinstance(phi, (stl.Globally, stl.Eventually, stl.Until)):
        if phi.unbound:
            width = 1.0
        elif phi.right_unbound:
            width = (T - 1 - phi.left_time_bound) / (T - 1)
        else:
            width = (phi.right_time_bound - phi.left_time_bound) / (T - 1)
    else:
        width = np.nan
    return {"root": root, "n_atoms": len(_atoms(phi)), "width": width}


rows = []
for d in DATASETS:
    X, y = load_train(d)
    for run in RUNS:
        fit, val = next(StratifiedShuffleSplit(1, test_size=0.25, random_state=run).split(X, y))
        cfg = ExperimentConfig(d, 10000, 3, 0.0, 3, 1.0, False, False, 0, 0, 0, 0, 0, 1, 0, ".", "cpu")
        formulas, F_fit, F_val, _, _ = build_formula_bank(X[fit], X[val], cfg, run)
        model = train_classifier(F_fit, y[fit], cfg, seed=run)
        acc = evaluate_classifier(model, F_val, y[val])["balanced_accuracy"]
        mass = np.abs(np.atleast_2d(model.coef_)).sum(axis=0)
        sel = np.flatnonzero(mass > 0)
        for i, phi in enumerate(formulas):
            r = describe(phi, X.shape[2])
            r.update(dataset=d, run=run, selected=mass[i] > 0, mass=mass[i] / max(mass.sum(), 1e-12), val_acc=acc)
            if r["selected"] and isinstance(phi, (stl.And, stl.Or)):
                kids = extract_features(X[fit], [phi.left_child, phi.right_child])
                c = [abs(np.corrcoef(F_fit[:, i], kids[:, k])[0, 1]) for k in range(2)]
                r["like_one_child"] = np.nanmax(c) > 0.95
            rows.append(r)
        print(f"{d} run={run} val={acc:.3f} selected={len(sel)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / "results" / "selected_formulas.csv", index=False)

df["single"] = df.n_atoms == 1
df["boolean_root"] = df.root.isin(["And", "Or"])
summ = []
for d, g in df.groupby("dataset"):
    s, b = g[g.selected], g
    summ.append({
        "dataset": d, "val_acc": g.val_acc.mean(), "n_sel/run": len(s) / len(RUNS),
        "single-atom bank": b.single.mean(), "single-atom sel": s.single.mean(),
        "single-atom |coef| share": s[s.single].mass.sum() / s.mass.sum(),
        "bool-root bank": b.boolean_root.mean(), "bool-root sel": s.boolean_root.mean(),
        "bool sel like 1 child": s[s.boolean_root].like_one_child.mean(),
        "width bank": b.width.median(), "width sel": s.width.median(),
        "top-5 |coef| share": g[g.selected].groupby("run").mass.apply(lambda m: m.nlargest(5).sum()).mean(),
    })
pd.set_option("display.width", 250)
print(pd.DataFrame(summ).set_index("dataset").round(2).to_string())
