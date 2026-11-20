# Visualization guidelines

Plot recorded data, identify partition, units and scope, and never use protected final observations. Missing/non-finite values are unavailable, not zero. Confidence intervals use model-minus-baseline loss; the zero reference is visible. An interval spanning zero is inconclusive, not evidence of equivalence.

Use aligned date axes for market/feature history, epoch axes for convergence, paired intervals for baseline comparisons, candles for revealed OHLC, and equity/drawdown for execution outcomes. Secondary diagnostics must not distract from the main finding. Line plots support series toggles, recent-window selection, pointer readout, expanded view and paper-friendly SVG/PNG. Exports reflect the displayed window and selected series. Do not describe sensitivity as causal attribution.

Replay uses the recorded decision clock. Do not display later candles or portfolio snapshots at an earlier cursor. Cross-model overlay requires matching market observations, execution profile, asset, partition, policy, accounting and compatible decision clocks.
