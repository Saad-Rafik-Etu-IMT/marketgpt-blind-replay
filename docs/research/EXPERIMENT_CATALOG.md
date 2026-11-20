# Experiment catalogue

Choose a recipe in Training or run a headless campaign from backend:

```bash
python -m src.experiments.campaigns positive --epochs 8
python -m src.experiments.campaigns negative --epochs 8
python -m src.experiments.campaigns architectures --source SPY --device auto
```

The command prints a new UUID and uses the same artifact registry as the UI.
Transformer recipes appear only with optional CPU/CUDA PyTorch installed.

| Recipe | Question | Control |
|---|---|---|
| smoke | Does the pipeline run? | Short deterministic development run |
| positive | Can the learner detect causal signal? | Generated autoregressive returns |
| negative | Does noise falsely look predictive? | Seeded random walk |
| capacity | Does depth help? | Same feature rows, tasks, seeds and selection |
| features | Which feature families add value? | Returns-only profile, paired comparison |
| optimization | Do hyperparameters improve development loss? | Deterministic bounded random/grid search |
| targets | Does joint prediction help? | Direction + volatility multi-head |
| horizon | How does the horizon change prediction? | One- and five-day expected returns |
| cost | What changes with a fixed hurdle? | Binary direction vs above-1-bp target |
| confirmation | Does performance survive chronological folds? | Nested purged expanding folds |
| architectures | MLP vs reference attention encoder | Shared 64-return contract |
| attention | Which encoder component matters? | Four paired ablations vs reference |

Cost sensitivity and market-regime slices are recorded automatically in
development evaluation. Single vs pooled source and feature/architecture
comparisons can be executed as separately configured campaigns. Absolute
lookback sweeps, optimizer-only factorial searches, rolling/asset holdout
campaigns, successive halving, and automated human-rater collection are not
implemented as turnkey recipes. Do not claim the 18 original experiment ideas
are all executed. See the backlog.
