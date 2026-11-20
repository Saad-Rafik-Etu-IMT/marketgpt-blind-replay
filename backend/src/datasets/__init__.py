"""Reproducible dataset construction from the feature catalogue (tranche T3).

See ``docs/datasets.md``. Pure computation plus an optional manifest file
write; no database or network access.
"""

from src.datasets.builder import (
    PART_NAMES,
    Dataset,
    DatasetError,
    DatasetSpec,
    Part,
    Representation,
    build_dataset,
)
from src.datasets.examples import AssetExamples, ContextRule, select_examples
from src.datasets.folds import (
    Fold,
    FoldError,
    inner_cutoff,
    inner_split,
    walk_forward,
    walk_forward_cutoffs,
)
from src.datasets.labels import DEFAULT_LABEL, LABELS, LabelError, LabelSpec, get_label
from src.datasets.manifest import ManifestError, read_manifest, write_manifest
from src.datasets.preprocessing import Scaler, ScalerError, ScalerKind, fit_scaler

__all__ = [
    "DEFAULT_LABEL",
    "LABELS",
    "PART_NAMES",
    "AssetExamples",
    "ContextRule",
    "Dataset",
    "DatasetError",
    "DatasetSpec",
    "Fold",
    "FoldError",
    "LabelError",
    "LabelSpec",
    "ManifestError",
    "Part",
    "Representation",
    "Scaler",
    "ScalerError",
    "ScalerKind",
    "build_dataset",
    "fit_scaler",
    "get_label",
    "inner_cutoff",
    "inner_split",
    "read_manifest",
    "select_examples",
    "walk_forward",
    "walk_forward_cutoffs",
    "write_manifest",
]
