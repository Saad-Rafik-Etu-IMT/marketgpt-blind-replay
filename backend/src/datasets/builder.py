"""Configurable, reproducible dataset construction (tranche T3).

``build_dataset(frames, spec)`` turns one or several assets' bars into model
inputs and labels:

1. features are computed per asset by the point-in-time engine (blind mode
   by default); the frames are cut at ``validation_end`` unless the sealed
   test set is explicitly requested;
2. decision bars are selected per partition (``examples.select_examples``)
   with one effective lookback for every asset;
3. an inner-validation fold is carved from training with a calendar cut-off
   shared by every asset, purged and embargoed (``folds.inner_split``);
4. rows with an undefined feature are dropped and counted, never filled;
5. the scaler is fitted on the ``fit_on`` part only and applied to all parts;
6. a manifest records data, features, label, split, counts and scaler, with a
   dataset fingerprint (``manifest.py``).

The ticker never enters X: assets are pooled by row, and ``Part.index`` keeps
asset and date as metadata for reports only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from src.data.split import SplitConfig, load_split_config
from src.datasets.examples import AssetExamples, ContextRule, select_examples
from src.datasets.folds import INNER_FRACTION, inner_cutoff, inner_split
from src.datasets.labels import DEFAULT_LABEL, LabelSpec, get_label
from src.datasets.preprocessing import Scaler, ScalerKind, fit_scaler
from src.features import DEFAULT_REGISTRY, FeatureRegistry, compute, get_profile
from src.features.canonical import to_canonical


class DatasetError(ValueError):
    """The dataset cannot be built from this specification and data."""


class Representation(str, Enum):
    TABULAR = "tabular"
    SEQUENCE = "sequence"


PART_NAMES: tuple[str, ...] = (
    "train",
    "inner_train",
    "inner_validation",
    "validation",
    "test",
)


class DatasetSpec(BaseModel):
    """Everything that defines a dataset; validated, JSON-serializable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    features: tuple[str, ...] = ()
    profile: str | None = None
    label: str = DEFAULT_LABEL
    representation: Representation = Representation.TABULAR
    sequence_length: int = Field(default=1, ge=1, le=256)
    context: ContextRule = ContextRule.OWN_PARTITION
    min_lookback: int = Field(default=0, ge=0, le=2000)
    scaler: ScalerKind = ScalerKind.STANDARD
    fit_on: Literal["inner_train", "train"] = "inner_train"
    inner_fraction: float = Field(default=INNER_FRACTION, gt=0, lt=1)
    primary_asset: str | None = None
    blind: bool = True
    include_test: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if bool(self.features) == bool(self.profile):
            raise ValueError("give exactly one of features or profile")
        if self.representation is Representation.TABULAR and self.sequence_length != 1:
            raise ValueError("sequence_length > 1 needs the sequence representation")
        return self

    def feature_ids(self) -> tuple[str, ...]:
        return get_profile(self.profile).features if self.profile else self.features


@dataclass(frozen=True)
class Part:
    """One partition: scaled inputs, raw inputs, labels and row metadata."""

    x: np.ndarray
    x_raw: np.ndarray
    label_values: np.ndarray
    y: np.ndarray
    index: pd.DataFrame

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class Dataset:
    spec: DatasetSpec
    feature_ids: tuple[str, ...]
    columns: tuple[str, ...]
    lookback: int
    label: LabelSpec
    scaler: Scaler
    parts: dict[str, Part]
    manifest: dict[str, object] = field(repr=False)

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["dataset_fingerprint"])


@dataclass
class _AssetData:
    canonical: pd.DataFrame
    features: np.ndarray
    label_values: np.ndarray
    examples: AssetExamples
    parts: dict[str, tuple[int, ...]] = field(default_factory=dict)


def _rows(asset: _AssetData, indices: tuple[int, ...], length: int) -> np.ndarray:
    if not indices:
        return np.zeros((0, length, asset.features.shape[1]))
    idx = np.asarray(indices)[:, None] + np.arange(1 - length, 1)[None, :]
    return asset.features[idx]


def build_dataset(
    frames: Mapping[str, pd.DataFrame],
    spec: DatasetSpec,
    config: SplitConfig | None = None,
    registry: FeatureRegistry = DEFAULT_REGISTRY,
) -> Dataset:
    from src.datasets.manifest import build_manifest

    if not frames:
        raise DatasetError("no asset frames given")
    config = config if config is not None else load_split_config()
    label = get_label(spec.label)
    feature_ids = spec.feature_ids()
    columns = tuple(registry.columns(feature_ids))
    length = spec.sequence_length
    lookback = max(registry.lookback(feature_ids) + length - 1, spec.min_lookback)
    primary = spec.primary_asset or next(iter(frames))
    if primary not in frames:
        raise DatasetError(f"primary asset {primary!r} not among the frames")

    assets: dict[str, _AssetData] = {}
    for name in sorted(frames):
        frame = frames[name]
        if not spec.include_test:
            dates = pd.to_datetime(
                frame["date"] if "date" in frame.columns else frame.index
            )
            frame = frame.loc[np.asarray(dates <= pd.Timestamp(config.validation_end))]
        canonical = to_canonical(frame)
        matrix = compute(frame, feature_ids, registry, blind=spec.blind)
        examples = select_examples(
            list(canonical.index.date),
            config,
            lookback=lookback,
            horizon=label.horizon,
            context=spec.context,
        )
        assets[name] = _AssetData(
            canonical=canonical,
            features=matrix.values.to_numpy(dtype=np.float64),
            label_values=label.values(canonical),
            examples=examples,
        )

    primary_train = assets[primary].examples
    if not primary_train.train:
        raise DatasetError(f"{primary}: no training example (history too short?)")
    cutoff = inner_cutoff(
        [primary_train.dates[t] for t in primary_train.train], spec.inner_fraction
    )
    for data in assets.values():
        fold = inner_split(
            data.examples, cutoff, embargo=config.embargo, context=spec.context
        )
        data.parts = {
            "train": data.examples.train,
            "inner_train": fold.train,
            "inner_validation": fold.evaluate,
            "validation": data.examples.validation,
        }
        if spec.include_test:
            data.parts["test"] = data.examples.test

    raw: dict[str, dict[str, object]] = {}
    dropped: dict[str, int] = {}
    for part in [p for p in PART_NAMES if p != "test" or spec.include_test]:
        xs, values, index = [], [], []
        dropped[part] = 0
        for name, data in assets.items():
            indices = data.parts[part]
            x = _rows(data, indices, length)
            label_values = data.label_values[list(indices)] if indices else np.zeros(0)
            keep = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(label_values)
            dropped[part] += int((~keep).sum())
            kept = (
                np.asarray(indices, dtype=int)[keep]
                if indices
                else np.zeros(0, dtype=int)
            )
            xs.append(x[keep])
            values.append(label_values[keep])
            index.append(
                pd.DataFrame(
                    {
                        "asset": name,
                        "date": [data.examples.dates[t] for t in kept],
                        "bar": kept,
                    }
                )
            )
        x_all = np.concatenate(xs) if xs else np.zeros((0, length, len(columns)))
        raw[part] = {
            "x": (
                x_all
                if spec.representation is Representation.SEQUENCE
                else x_all[:, -1, :]
            ),
            "values": np.concatenate(values),
            "index": pd.concat(index, ignore_index=True),
        }

    fit_rows = raw[spec.fit_on]["x"]
    if len(fit_rows) == 0:  # type: ignore[arg-type]
        raise DatasetError(f"no {spec.fit_on} rows to fit the scaler")
    scaler = fit_scaler(spec.scaler, fit_rows, columns)  # type: ignore[arg-type]
    parts = {
        name: Part(
            x=scaler.transform(r["x"]),  # type: ignore[arg-type]
            x_raw=r["x"],  # type: ignore[arg-type]
            label_values=r["values"],  # type: ignore[arg-type]
            y=(r["values"] > 0).astype(np.int64),  # type: ignore[operator]
            index=r["index"],  # type: ignore[arg-type]
        )
        for name, r in raw.items()
    }
    manifest = build_manifest(
        spec=spec,
        config=config,
        registry=registry,
        feature_ids=feature_ids,
        columns=columns,
        lookback=lookback,
        label=label,
        cutoff=cutoff,
        canonical={name: data.canonical for name, data in assets.items()},
        parts=parts,
        dropped=dropped,
        scaler=scaler,
    )
    return Dataset(
        spec=spec,
        feature_ids=feature_ids,
        columns=columns,
        lookback=lookback,
        label=label,
        scaler=scaler,
        parts=parts,
        manifest=manifest,
    )
