"""Experiment orchestration on top of features, datasets, targets and the MLP (T6).

See ``docs/experiments.md``.
"""

from src.experiments.estimate import estimate
from src.experiments.runner import CancelledError, Runner, run_folder, write_spec
from src.experiments.sources import (
    SYNTHETIC,
    SourceError,
    health,
    load_frames,
    synthetic_frames,
)
from src.experiments.spec import (
    EXPERIMENT_PROFILES,
    MAX_RUNS,
    ExperimentProfile,
    ExperimentSpec,
    Run,
)

__all__ = [
    "EXPERIMENT_PROFILES",
    "MAX_RUNS",
    "SYNTHETIC",
    "CancelledError",
    "ExperimentProfile",
    "ExperimentSpec",
    "Run",
    "Runner",
    "SourceError",
    "estimate",
    "health",
    "load_frames",
    "run_folder",
    "synthetic_frames",
    "write_spec",
]
