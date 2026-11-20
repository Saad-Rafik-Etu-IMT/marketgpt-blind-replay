# Interactive research workstation

Extends the unified cockpit without changing the registered training objective,
checkpoint-selection rule, broker assumptions, or protected final test.

## Research workflows and controls

| Area | Implementation |
|---|---|
| Charts | Shared SVG chart surface: anchored wheel zoom, drag horizontal navigation, Shift-wheel/drag price scaling, manual/auto Y, crosshair with native values, fit/reset, earlier/later controls, expanded charts and existing PNG/SVG export. Large line plots retain extrema using bounded decimation; dense candles fall back to decimated closing prices. |
| Linked exploration | Shared time range and crosshair on aligned data plots and the replay market, equity, P&L, drawdown, RSI, volume and volatility plots. Shared domains contain only revealed observations. |
| Replay | Toggle SMA20, EMA20, Bollinger bands, simple RSI14, recorded volume and annualized volatility; descriptive causal regime shading; model decision markers; clickable historical decisions, persistent local bookmarks and two-point trend lines. Compare compatible checkpoints (including MLP/Transformer when registered), baseline leaderboard, pause at model disagreement, revealed execution ledger and JSON report. |
| Human exercise | Optional UI-blind human direction prediction, locked per session in memory; hides current model decision until the user guesses. Scores resolve only after both relevant open prices become visible. Not a tamper-proof competition or holdout: clients can read recorded development artifacts. |
| Training | Layer schematic with inspectable stage, duplication of selected campaign spec, epoch diagnostics and possible overfitting warning, retained checkpoint/early stop, measured elapsed time and process CPU relative to one core, optional Unix peak RSS and rough epoch-rate ETA. No fabricated GPU usage. |
| Evaluation | Config loss heatmap and ablation selection, regime loss plots, real single-checkpoint interactive confusion matrix, threshold and asset/regime filters, probability bins, rolling accuracy, error/confidence ordering, pairwise matching by asset/date/label. |
| Explainability | Selected prediction's exact registered inputs, sequential scaled-input replacement waterfall, bounded per-column counterfactual, measured MLP hidden activations, local constrained factual verbalization. |
| Workspace | Ctrl/Cmd+K palette, persistent sidebar/layout/panel split preferences, shareable URL, mobile stacking, existing theme, history and presentation functionality. |

## Scientific interpretation

The checkpoint inspector is distinct from the existing ensemble report. It only
loads completed checkpoints, verifies source/split and dataset fingerprints,
uses the actual trained column order, and allows binary classification on a
previously evaluated development partition. Historical validation reads are logged.
The final test is never built or queried. The threshold slider is a local
exploration aid, not an optimization or promotion process.

Sequential input replacement starts at a zero model-input vector, then restores
the actual columns in registered order. Signed probability changes telescope to
the observed prediction, but depend on the order and reference. These are not
SHAP values or causal claims. Counterfactuals replace one complete feature
column over the input window on the registered scaled-input representation,
bounded to [-3, 3]; they need not represent physically attainable market states.
Transformer predictions are supported through the existing registered adapter;
individual hidden activations are only available for MLPs.

Descriptive replay regime shading is separate from evaluation regimes fitted on
training volatility terciles. The human mode is educational and does not prevent
inspection of locally stored development data. GPU utilization is not measured;
the process CPU and RSS are actual worker observations (RSS optional by OS).
ETA excludes early stopping and is not a promise. No new runtime dependency.

## Validation

- Backend: `python -m pytest` from backend (995 passed, 8 optional Torch tests skipped,
  94.08% coverage). New checkpoint inspection test asserts exact prediction parity,
  telescoping contributions, bounds, and rejection of the protected test.
- Frontend: typecheck, lint, 42 unit tests, production build.
- Chromium laboratory workflow passed with training, report, interactive confusion
  matrix, exact case attribution, counterfactual, deterministic replay, SVG/PNG
  download, URL restoration, presentation, and mobile width check.
- All ten existing Chromium integration workflows passed. An additional
  chart-specific Chromium test passed actual wheel zoom, drag navigation,
  Shift-wheel Y scaling, synchronized chart ranges and fit/reset.
- Replay defaults to a full-width market chart, with a side-by-side option;
  the fixed-height crosshair readout avoids plot movement during gestures.
- Existing saved reports remain usable; the inspection endpoint reconstructs
  development cases on demand from the matching saved checkpoint.

The hosted static Sites preview is a separate deployment; it must be rebuilt with
recorded results to expose backend-dependent new features.
