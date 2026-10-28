"""
Dataset / run configuration.

A run is described by a small JSON file (see configs/). Every key has a
default, so a config only needs to say where the data is and which clinical
column is the target. Paths are resolved relative to the repository root.
"""
from dataclasses import dataclass, field, fields
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # ---- dataset ------------------------------------------------------
    name: str = "dataset"
    source: str = "local"                 # "local" (files on disk) or "geo" (download with GEOparse)
    geo_id: str | None = None             # e.g. "GSE7390" when source == "geo"

    expression_file: str | None = None    # CSV/TSV (optionally .gz)
    expression_sep: str = "\t"
    samples_in_columns: bool = True       # GEO convention: probes x samples
    sample_id_strip: list = field(default_factory=list)  # substrings removed from sample IDs, e.g. [".CEL.gz"]

    clinical_file: str | None = None
    clinical_sep: str = "\t"
    sample_id_column: str | None = None   # clinical column holding sample IDs (None = first column)

    annotation_file: str | None = None    # optional probe -> gene symbol table (columns: ID, Gene Symbol)

    # ---- target / subgroups -------------------------------------------
    target_column: str | None = None
    positive_label: str | int | None = None  # value counted as class 1 (None = target already 0/1)
    negative_label: str | int | None = None  # if set, samples with any other value are dropped
    class_names: list = field(default_factory=lambda: ["Negative", "Positive"])
    subgroup_column: str | None = None    # e.g. "ER_status"; evaluated on the test set only

    # ---- preprocessing --------------------------------------------------
    log2_transform: str = "auto"          # "auto" (if max > 100), "yes" or "no"
    max_missing_fraction: float = 0.2     # drop features missing in more than this fraction of samples

    # ---- modelling --------------------------------------------------------
    seed: int = 42
    test_size: float = 0.20
    n_splits: int = 5
    n_selected_features: int = 1000
    pca_prefilter_features: int = 5000
    pca_components: int = 30
    knn_neighbors: int = 11
    rf_trees: int = 500

    output_dir: str | None = None         # default: results/<name>

    # ------------------------------------------------------------------
    def path(self, value):
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    @property
    def out_dir(self):
        return self.path(self.output_dir or f"results/{self.name}")

    @classmethod
    def load(cls, path, **overrides):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)
