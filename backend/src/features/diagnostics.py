"""Redundancy diagnostics on TRAINING rows only.

``training_rows`` selects the rows of a feature matrix whose bar lies in the
F2 training period, so a diagnostic can never look at validation or test data.
``redundancy_report`` then measures:

- pairs of columns whose absolute correlation reaches a threshold;
- clusters of columns linked by such pairs (connected components);
- how many principal components explain 90%, 95% and 99% of the variance of
  the standardized columns (the effective number of independent directions).

These are descriptive statistics. A high correlation shows shared information
on this period, not that a feature is useless; the ablation (T7) decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.split import Split, SplitConfig, assign_splits


class DiagnosticsError(ValueError):
    """Not enough usable rows or columns for a diagnostic."""


@dataclass(frozen=True)
class CorrelatedPair:
    first: str
    second: str
    correlation: float


@dataclass(frozen=True)
class RedundancyReport:
    method: str
    threshold: float
    rows: int
    columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    pairs: tuple[CorrelatedPair, ...]
    clusters: tuple[tuple[str, ...], ...]
    components_for_variance: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "rows": self.rows,
            "columns": list(self.columns),
            "constant_columns": list(self.constant_columns),
            "pairs": [
                {"first": p.first, "second": p.second, "correlation": p.correlation}
                for p in self.pairs
            ],
            "clusters": [list(c) for c in self.clusters],
            "components_for_variance": dict(self.components_for_variance),
        }


def training_rows(values: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Rows whose bar date is in the training period, with every column defined."""
    labels = assign_splits(list(values.index.date), config)
    mask = np.array([label is Split.TRAIN for label in labels], dtype=bool)
    return values.loc[mask].dropna()


def redundancy_report(
    rows: pd.DataFrame, threshold: float = 0.95, method: str = "spearman"
) -> RedundancyReport:
    """Correlation pairs, clusters and PCA summary of already-selected rows."""
    if method not in ("spearman", "pearson"):
        raise DiagnosticsError("method must be 'spearman' or 'pearson'")
    if not 0 < threshold <= 1:
        raise DiagnosticsError("threshold must be in (0, 1]")
    rows = rows.dropna()
    if len(rows) < 3:
        raise DiagnosticsError("at least 3 complete rows are required")
    spread = rows.std(ddof=0)
    constant = tuple(c for c in rows.columns if spread[c] == 0)
    usable = rows.drop(columns=list(constant))
    if usable.shape[1] < 2:
        raise DiagnosticsError("at least 2 non-constant columns are required")

    corr = usable.corr(method=method).to_numpy()
    names = tuple(usable.columns)
    pairs: list[CorrelatedPair] = []
    parent = list(range(len(names)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) >= threshold:
                pairs.append(
                    CorrelatedPair(names[i], names[j], round(float(corr[i, j]), 6))
                )
                parent[root(i)] = root(j)
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        groups.setdefault(root(i), []).append(name)
    clusters = tuple(tuple(g) for g in groups.values() if len(g) > 1)

    standardized = (usable - usable.mean()) / usable.std(ddof=0)
    singular = np.linalg.svd(standardized.to_numpy(), compute_uv=False) ** 2
    share = np.cumsum(singular / singular.sum())
    components = {
        f"{int(q * 100)}%": int(np.searchsorted(share, q - 1e-12) + 1)
        for q in (0.90, 0.95, 0.99)
    }
    return RedundancyReport(
        method=method,
        threshold=threshold,
        rows=len(usable),
        columns=names,
        constant_columns=constant,
        pairs=tuple(sorted(pairs, key=lambda p: -abs(p.correlation))),
        clusters=clusters,
        components_for_variance=components,
    )
