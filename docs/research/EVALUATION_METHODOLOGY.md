# Evaluation methodology

The existing T7 implementation is documented in [evaluation.md](../evaluation.md).
The laboratory displays model-vs-baseline loss, per-seed loss, paired
date-block intervals and verdicts, reliability, ROC/precision-recall curves,
binary confusion heatmap, feature ablations, market-regime slices and
feature-shift sensitivity. ROC/PR points account for score ties and are
downsampled for plotting, not used to recompute the stored AUC.

- Predictive metrics are task-specific, with no fabricated unavailable values.
- Loss differences are model minus baseline; negative is favorable. An interval
  crossing zero is **no measurable difference**, not equivalence.
- Seed spread assesses training variation. Calendar-block intervals assess
  uncertainty over market dates, keeping correlated assets together.
- Cost-sensitivity plots use the existing vectorized long/flat return proxy
  with 0, 0.5, 2 and 5 bps per change. They are not broker execution results.
- Registered binary replay uses the existing broker/risk/portfolio engine,
  identical timeline and execution profile for the selected model and two
  trading baselines. It records fills, equity and risk pause state; a next-close
  snapshot is not yet the realized next-open-to-following-open label.
- Feature shifts measure association, not causality. LLM output can reorder
  three recorded decision facts, with an explicit measured-template fallback.

Results can be exported from saved artifacts as Markdown, CSV and JSON.
Existing scientific plot exports are SVG. PNG and slide generation are not
implemented as report exports. Uncertainty from repeated model selection
requires a separate frozen confirmation; no automatic promotion based on
profit or final holdout outcome.
