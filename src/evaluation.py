"""Metrics, MCC threshold optimisation and the feature-selection stability check."""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score,
)

THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)


def compute_scores(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Threshold": threshold,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def mcc_curve(y_true, y_prob, grid=THRESHOLD_GRID):
    return [matthews_corrcoef(y_true, (y_prob >= t).astype(int)) for t in grid]


def best_mcc_threshold(y_true, y_prob, grid=THRESHOLD_GRID):
    mccs = mcc_curve(y_true, y_prob, grid)
    best_idx = int(np.argmax(mccs))
    return float(grid[best_idx]), mccs[best_idx]


def feature_selection_overlap(Xtr, ytr, cv, k):
    """Refit the ANOVA selector in each training fold and return the pairwise overlap matrix.

    A fixed gene list would give 100% overlap by construction; values well below 1
    show the selector really is refit per fold.
    """
    k = min(k, Xtr.shape[1])
    selected = []
    for tr_idx, _ in cv.split(Xtr, ytr):
        Xf = SimpleImputer(strategy="median").fit_transform(Xtr.iloc[tr_idx])
        sel = SelectKBest(f_classif, k=k).fit(Xf, ytr.iloc[tr_idx])
        selected.append(set(np.flatnonzero(sel.get_support())))
    n = len(selected)
    overlap = np.array([[len(selected[i] & selected[j]) / k for j in range(n)] for i in range(n)])
    return overlap, float(overlap[~np.eye(n, dtype=bool)].mean())
