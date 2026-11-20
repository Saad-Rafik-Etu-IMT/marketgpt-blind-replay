"""Scientific evaluation of experiments: baselines, paired intervals, ablation (T7).

See ``docs/evaluation.md``.
"""

from src.evaluation.baselines import UNAVAILABLE, baseline_outputs
from src.evaluation.evaluate import (
    PARTS,
    EvaluationError,
    evaluate_experiment,
    read_evaluation,
)
from src.evaluation.stats import (
    block_bootstrap_mean,
    drift,
    paired_difference,
    psi,
    row_losses,
    verdict,
)

__all__ = [
    "PARTS",
    "UNAVAILABLE",
    "EvaluationError",
    "baseline_outputs",
    "block_bootstrap_mean",
    "drift",
    "evaluate_experiment",
    "paired_difference",
    "psi",
    "read_evaluation",
    "row_losses",
    "verdict",
]
