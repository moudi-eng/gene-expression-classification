"""
Leakage-controlled gene-expression classification pipeline.

Works on any binary phenotype from a gene-expression dataset, either local files
or a series downloaded directly from NCBI GEO. Reference run: GSE2034
breast-cancer relapse (Wang et al., Lancet 2005), as described in the report.

Usage
-----
    # reproduce the report (GSE2034, local files in data/)
    python src/pipeline.py --config configs/gse2034.json

    # any GEO series: list its clinical columns first, then pick a target
    python src/pipeline.py --geo GSE7390 --list-columns
    python src/pipeline.py --geo GSE7390 --target er --positive 1

    # your own files
    python src/pipeline.py --expression my_expr.csv --clinical my_clin.csv --target status --positive relapse

Steps: load -> QC -> stratified train/test split -> 5-fold CV inside the training
set (imputation, ANOVA selection, scaling and PCA refit in every fold) ->
MCC threshold from out-of-fold probabilities -> one evaluation on the untouched
test set -> optional subgroup analysis -> top features.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plots  # noqa: E402
from config import Config  # noqa: E402
from data import describe_columns, load_annotation, load_dataset, make_target, preprocess  # noqa: E402
from evaluation import best_mcc_threshold, compute_scores, feature_selection_overlap  # noqa: E402
from models import build_models  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON config file (see configs/)")
    p.add_argument("--geo", dest="geo_id", help="GEO series accession, e.g. GSE7390")
    p.add_argument("--expression", dest="expression_file", help="expression matrix file")
    p.add_argument("--clinical", dest="clinical_file", help="clinical / phenotype table")
    p.add_argument("--target", dest="target_column", help="clinical column to predict")
    p.add_argument("--positive", dest="positive_label", help="value of the target counted as class 1")
    p.add_argument("--negative", dest="negative_label", help="value counted as class 0 (others dropped)")
    p.add_argument("--subgroup", dest="subgroup_column", help="clinical column for subgroup analysis")
    p.add_argument("--name", help="run name (output goes to results/<name>)")
    p.add_argument("--list-columns", action="store_true",
                   help="only load the data and print the clinical columns")
    return p.parse_args()


def make_config(args):
    """Config file first (if any), then command-line flags on top."""
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "list_columns") and v is not None}
    if args.geo_id:
        overrides["source"] = "geo"
        overrides.setdefault("name", args.geo_id)
    elif args.expression_file:
        overrides["source"] = "local"
    if args.config:
        return Config.load(args.config, **overrides)
    return Config(**overrides)


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    cfg = make_config(args)

    print(f"Loading data ({cfg.name})...")
    X, clin = load_dataset(cfg)
    print(f"Expression matrix: {X.shape[0]} samples x {X.shape[1]} features")

    if args.list_columns or not cfg.target_column:
        print("\nClinical columns (choose one with --target / target_column):")
        describe_columns(clin)
        if not cfg.target_column:
            print("\nNo target set: re-run with --target <column> [--positive <value>].")
        return

    y = make_target(clin, cfg)
    X, clin = X.loc[y.index], clin.loc[y.index]
    groups = clin[cfg.subgroup_column] if cfg.subgroup_column else pd.Series("all", index=y.index)
    print(f"Target {cfg.target_column!r}: {y.value_counts().sort_index().to_dict()} "
          f"(1 = {cfg.class_names[1]})")
    if cfg.subgroup_column:
        print(f"Subgroups {cfg.subgroup_column!r}: {groups.value_counts(dropna=False).to_dict()}")

    X, qc = preprocess(X, cfg)
    print(f"QC: {qc}")

    out, fig = cfg.out_dir, cfg.out_dir / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    plots.class_balance(y, cfg.class_names, fig / "class_balance.png")
    evr = plots.pca_overview(X, y, cfg.class_names, cfg.seed, fig / "pca_overview.png")

    # ---- stratified hold-out split: the test set is not touched until the end ----
    Xtr, Xte, ytr, yte, g_tr, g_te = train_test_split(
        X, y, groups, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    print(f"Train: {len(Xtr)} samples | Test: {len(Xte)} samples")

    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    overlap, stability = feature_selection_overlap(Xtr, ytr, cv, cfg.n_selected_features)
    plots.feature_stability(overlap, fig / "feature_stability.png")
    print(f"Mean pairwise feature-set overlap across folds: {stability:.3f}")

    min_fold_train = min(len(tr) for tr, _ in cv.split(Xtr, ytr))
    models = build_models(cfg, X.shape[1], min_fold_train)

    rows, curves, cms, oof, fitted = [], {}, {}, {}, {}
    for name, pipe in models.items():
        print(f"\n=== {name} ===")
        p_oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        best_t, best_mcc_oof = best_mcc_threshold(ytr, p_oof)
        print(f"  OOF best threshold={best_t:.2f} (OOF MCC={best_mcc_oof:.3f})")
        oof[name] = (ytr.values, p_oof, best_t)

        fitted[name] = pipe.fit(Xtr, ytr)
        p_test = fitted[name].predict_proba(Xte)[:, 1]

        s = compute_scores(yte, p_test, best_t)
        s["Model"] = name
        s["ROC_AUC"] = roc_auc_score(yte, p_test)
        s["PR_AUC"] = average_precision_score(yte, p_test)
        rows.append(s)

        fpr, tpr, _ = roc_curve(yte, p_test)
        prec, rec, _ = precision_recall_curve(yte, p_test)
        curves[name] = {"fpr": fpr, "tpr": tpr, "roc_auc": s["ROC_AUC"],
                        "precision": prec, "recall": rec, "pr_auc": s["PR_AUC"]}
        cms[name] = confusion_matrix(yte, (p_test >= best_t).astype(int), labels=[0, 1])
        print(f"  Test ROC-AUC={s['ROC_AUC']:.3f} PR-AUC={s['PR_AUC']:.3f} "
              f"MCC={s['MCC']:.3f} F1={s['F1']:.3f}")

    results = pd.DataFrame(rows).sort_values("MCC", ascending=False).reset_index(drop=True)
    results.to_csv(out / "model_comparison.csv", index=False)
    print("\n=== Final model comparison (sorted by MCC) ===")
    print(results[["Model", "ROC_AUC", "PR_AUC", "MCC", "Accuracy", "Precision",
                   "Sensitivity", "Specificity", "F1", "Threshold"]].round(3).to_string(index=False))

    best = results.iloc[0]["Model"]
    print(f"\nBest model by MCC: {best}")
    plots.roc_pr(curves, fig / "roc_curves.png", fig / "pr_curves.png")
    plots.confusion_matrices(cms, cfg.class_names, fig / "confusion_matrices.png")
    plots.metric_bars(results, fig / "metric_comparison.png")
    y_oof, p_oof, t_best = oof[best]
    plots.threshold_curve(y_oof, p_oof, t_best, best, fig / "threshold_optimization.png")
    joblib.dump({"model": fitted[best], "threshold": t_best, "features": list(X.columns)},
                out / "best_model.joblib")

    # ---- subgroup analysis on the test set, threshold already fixed ----
    if cfg.subgroup_column:
        print(f"\n=== {cfg.subgroup_column} subgroup analysis ({best}, held-out test set) ===")
        p_best = fitted[best].predict_proba(Xte)[:, 1]
        sub_rows = []
        for level in sorted(g_te.dropna().unique(), key=str):
            mask = (g_te == level).values
            if mask.sum() < 5:
                print(f"  {level}: n={mask.sum()} (too small, skipped)")
                continue
            s = compute_scores(yte[mask], p_best[mask], t_best)
            s["Subgroup"], s["N"] = level, int(mask.sum())
            try:
                s["ROC_AUC"] = roc_auc_score(yte[mask], p_best[mask])
            except ValueError:  # only one class present in this subgroup
                s["ROC_AUC"] = np.nan
            sub_rows.append(s)
            print(f"  {level} (n={s['N']}): MCC={s['MCC']:.3f} Sensitivity={s['Sensitivity']:.3f} "
                  f"Specificity={s['Specificity']:.3f} ROC-AUC={s['ROC_AUC']:.3f}")
        if sub_rows:
            sub = pd.DataFrame(sub_rows)
            sub.to_csv(out / "subgroup_analysis.csv", index=False)
            plots.subgroups(sub, best, cfg.subgroup_column, fig / "subgroup_performance.png")

    # ---- top features of the reference linear model ----
    ref = next(m for m in ["LASSO", "Elastic Net", "Logistic Regression"] if m in fitted)
    selector = fitted[ref].named_steps["select"]
    mask = selector.get_support()
    sel_names, sel_scores = X.columns[mask], selector.scores_[mask]
    order = np.argsort(sel_scores)[::-1][:25]
    top = pd.DataFrame({"Feature_ID": np.array(sel_names)[order], "ANOVA_F_score": sel_scores[order]})
    ann = load_annotation(cfg)
    top["Gene_Symbol"] = top["Feature_ID"].map(ann) if ann is not None else np.nan
    top["Label"] = [f"{f} ({g})" if isinstance(g, str) and g else f
                    for f, g in zip(top["Feature_ID"], top["Gene_Symbol"])]
    top.drop(columns="Label").to_csv(out / "top_selected_features.csv", index=False)
    plots.top_features(top, ref, fig / "top_features.png")

    summary = {
        "name": cfg.name, "target": cfg.target_column,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "qc": qc,
        "pca_explained_variance_first10": [float(v) for v in evr],
        "feature_selection_fold_overlap": stability,
        "best_model": best,
        "best_model_test": {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                            for k, v in results.iloc[0].items()},
        "config": {k: v for k, v in vars(cfg).items()},
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nAll results, figures, and tables saved to:", out)


if __name__ == "__main__":
    main()
