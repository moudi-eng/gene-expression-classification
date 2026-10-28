"""
Loading expression + clinical data from local files or directly from NCBI GEO,
and turning a clinical column into a binary target.

Only pandas/numpy are needed here (GEOparse is imported lazily), so this module
can be used to inspect a dataset before any modelling.
"""
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# GEO download (GEOparse)
# ----------------------------------------------------------------------
def download_geo(geo_id, dest):
    """Download a GEO series and cache expression / clinical / annotation tables in `dest`.

    Clinical columns come from each sample's "characteristics" lines, which GEO
    stores as "key: value" (e.g. "er: 1", "relapse status: yes").
    """
    import GEOparse

    dest = Path(dest)
    expr_path = dest / "expression.csv.gz"
    clin_path = dest / "clinical.tsv"
    annot_path = dest / "annotation.tsv"
    if expr_path.exists() and clin_path.exists():
        return expr_path, clin_path, (annot_path if annot_path.exists() else None)

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {geo_id} from NCBI GEO (cached in {dest})...")
    gse = GEOparse.get_GEO(geo=geo_id, destdir=str(dest / "raw"), silent=True)

    expr = gse.pivot_samples("VALUE")            # probes x samples
    expr.to_csv(expr_path, sep="\t", compression="gzip")

    records = {}
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata
        rec = {
            "title": meta.get("title", [""])[0],
            "source_name": meta.get("source_name_ch1", [""])[0],
        }
        for key in ("characteristics_ch1", "characteristics_ch2"):
            for item in meta.get(key, []):
                if ":" in item:
                    k, v = item.split(":", 1)
                    rec[k.strip()] = v.strip()
        records[gsm_name] = rec
    clin = pd.DataFrame.from_dict(records, orient="index")
    clin.index.name = "GSM"
    clin.to_csv(clin_path, sep="\t")

    annot_written = None
    for gpl in gse.gpls.values():
        table = gpl.table
        symbol_col = next((c for c in table.columns
                           if c.lower().replace("_", " ") in ("gene symbol", "symbol", "gene_symbol")), None)
        if symbol_col is not None and "ID" in table.columns:
            table[["ID", symbol_col]].rename(columns={symbol_col: "Gene Symbol"}) \
                .to_csv(annot_path, sep="\t", index=False)
            annot_written = annot_path
            break

    print(f"  expression: {expr.shape[0]} probes x {expr.shape[1]} samples")
    print(f"  clinical columns: {list(clin.columns)}")
    return expr_path, clin_path, annot_written


# ----------------------------------------------------------------------
# Generic loading
# ----------------------------------------------------------------------
def load_dataset(cfg):
    """Return (X samples x features, clinical table) aligned on sample ID."""
    if cfg.source == "geo":
        if not cfg.geo_id:
            raise ValueError("source='geo' requires geo_id")
        expr_path, clin_path, annot = download_geo(cfg.geo_id, cfg.path(f"data/geo/{cfg.geo_id}"))
        expr_sep, clin_sep, samples_in_columns, id_col = "\t", "\t", True, "GSM"
        if cfg.annotation_file is None and annot is not None:
            cfg.annotation_file = str(annot)
    else:
        expr_path, clin_path = cfg.path(cfg.expression_file), cfg.path(cfg.clinical_file)
        if expr_path is None or clin_path is None:
            raise ValueError("source='local' requires expression_file and clinical_file")
        expr_sep, clin_sep = cfg.expression_sep, cfg.clinical_sep
        samples_in_columns, id_col = cfg.samples_in_columns, cfg.sample_id_column

    expr = pd.read_csv(expr_path, sep=expr_sep, index_col=0)
    if samples_in_columns:
        expr = expr.T
    expr.index = expr.index.astype(str)
    for s in cfg.sample_id_strip:
        expr.index = expr.index.str.replace(s, "", regex=False)
    expr = expr.apply(pd.to_numeric, errors="coerce")  # non-numeric annotation columns become NaN
    expr = expr.loc[:, expr.notna().any(axis=0)]

    clin = pd.read_csv(clin_path, sep=clin_sep)
    clin = clin.set_index(id_col if id_col else clin.columns[0])
    clin.index = clin.index.astype(str)

    ids = expr.index.intersection(clin.index)
    if len(ids) == 0:
        raise ValueError("No sample IDs shared between expression and clinical files. "
                         "Check samples_in_columns, sample_id_column and sample_id_strip.")
    dropped = (len(expr) - len(ids), len(clin) - len(ids))
    if any(dropped):
        print(f"Unmatched samples dropped: {dropped[0]} expression-only, {dropped[1]} clinical-only")
    return expr.loc[ids], clin.loc[ids]


def describe_columns(clin, max_levels=8):
    """Print each clinical column with its most frequent values (to help choose a target)."""
    for col in clin.columns:
        vc = clin[col].value_counts(dropna=False)
        levels = ", ".join(f"{k}={v}" for k, v in vc.head(max_levels).items())
        more = f" (+{len(vc) - max_levels} more)" if len(vc) > max_levels else ""
        print(f"  {col!r}: {len(vc)} distinct -> {levels}{more}")


# ----------------------------------------------------------------------
# Target and preprocessing
# ----------------------------------------------------------------------
def _norm(v):
    """Compare labels as text so 1, 1.0 and "1" all match."""
    if pd.isna(v):
        return None
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v).strip().lower()


def make_target(clin, cfg):
    """Binary target (1 = positive class). Samples with a missing/unwanted label are dropped."""
    if cfg.target_column not in clin.columns:
        raise ValueError(f"target_column {cfg.target_column!r} not in clinical columns: {list(clin.columns)}")
    raw = clin[cfg.target_column].map(_norm)
    keep = raw.notna()

    if cfg.positive_label is None:
        levels = set(raw[keep])
        if not levels <= {"0", "1"}:
            raise ValueError(f"Target {cfg.target_column!r} has values {sorted(levels)}; "
                             "set positive_label (and optionally negative_label) in the config.")
        y = raw[keep].astype(int)
    else:
        pos = _norm(cfg.positive_label)
        if cfg.negative_label is not None:
            keep &= raw.isin([pos, _norm(cfg.negative_label)])
        y = (raw[keep] == pos).astype(int)

    if y.nunique() != 2:
        raise ValueError(f"Target needs two classes after filtering, got counts {y.value_counts().to_dict()}")
    return y


def preprocess(X, cfg):
    """Label-free cleaning applied once to the whole matrix (no outcome information is used)."""
    report = {}
    missing = X.isna().mean(axis=0)
    report["features_dropped_missing"] = int((missing > cfg.max_missing_fraction).sum())
    X = X.loc[:, missing <= cfg.max_missing_fraction]

    do_log = cfg.log2_transform == "yes" or (cfg.log2_transform == "auto" and np.nanmax(X.values) > 100)
    if do_log:
        X = np.log2(X.clip(lower=0) + 1)
    report["log2_applied"] = bool(do_log)

    zero_var = X.var(axis=0) == 0
    report["zero_variance_removed"] = int(zero_var.sum())
    X = X.loc[:, ~zero_var]
    report["remaining_missing_values"] = int(X.isna().sum().sum())  # imputed inside each CV fold
    return X, report


def load_annotation(cfg):
    path = cfg.path(cfg.annotation_file)
    if path is None or not path.exists():
        return None
    ann = pd.read_csv(path, sep="\t", dtype=str)
    return ann.set_index(ann.columns[0])[ann.columns[1]]
