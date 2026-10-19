"""Decision tree on the STL feature matrix (a baseline of the comparison) and the STL formula
of each of its root-to-leaf paths."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from .classifier import cv_splitter
from .instance_rules import to_stl

DEPTHS = (1, 3, 5, 7, None)   # None: grown until the leaves are pure


def train_tree(X_tr_feats: np.ndarray, y_tr: np.ndarray, cv: int, seed: int, n_jobs: int = 1,
               depths=DEPTHS) -> GridSearchCV:
    """DecisionTreeClassifier(class_weight="balanced") with max_depth chosen by stratified CV
    on balanced accuracy, over the same folds as the lasso of train_classifier (cv_splitter);
    ties go to the shallower tree. Refitted on the whole train set at the chosen depth."""
    search = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        {"max_depth": list(depths)},
        scoring=make_scorer(balanced_accuracy_score),
        cv=cv_splitter(y_tr, cv, seed),
        n_jobs=n_jobs,
    )
    return search.fit(X_tr_feats, y_tr)


def path_literals(tree: DecisionTreeClassifier) -> list[dict]:
    """One dict per leaf, left to right: leaf (node id), class (the leaf's prediction),
    n_train (train series reaching it) and literals, the splits on its root-to-leaf path in
    the format of instance_rules: a split x_j <= t gives (j, -1, -t) on the true branch,
    x_j > t gives (j, 1, t) on the false one."""
    t = tree.tree_
    leaves = []

    def walk(node: int, literals: list) -> None:
        left, right = t.children_left[node], t.children_right[node]
        if left == right:   # leaf: both children are TREE_LEAF
            leaves.append({"leaf": int(node), "class": tree.classes_[int(np.argmax(t.value[node][0]))],
                           "n_train": int(t.n_node_samples[node]), "literals": literals})
            return
        j, thr = int(t.feature[node]), float(t.threshold[node])
        walk(left, literals + [(j, -1.0, -thr)])
        walk(right, literals + [(j, 1.0, thr)])

    walk(0, [])
    return leaves


def path_formulas(tree: DecisionTreeClassifier, formulas: list, sigma: np.ndarray) -> list[dict]:
    """path_literals with the STL formula of each path: the conjunction of NOT phi_j^{-sigma_j t}
    for each split x_j <= t taken on the true branch and phi_j^{-sigma_j t} for each one taken
    on the false branch (instance_rules.to_stl). formulas, sigma: the bank and feature scales
    of build_formula_bank, on which the tree was trained. A tree with a single leaf has the
    formula None (always true). A series reaches a leaf iff its formula has positive
    robustness, except for features exactly at a threshold."""
    return [{**leaf, "formula": to_stl(leaf["literals"], formulas, sigma)} for leaf in path_literals(tree)]
