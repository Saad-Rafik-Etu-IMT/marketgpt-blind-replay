# Model architectures

| Model family | Implemented contract |
|---|---|
| NumPy MLP | Existing configurable shallow/deep networks, multi-head targets, SGD/AdamW, batch/layer normalization, dropout, learning rate schedules, early stopping |
| Reference Transformer | Existing 64-return token encoder, 16-dimensional embeddings, two attention heads; reference plus four ablations; optional CPU/CUDA |
| Statistical baselines | Class frequencies, logistic regression, depth-one decision stump, mean/zero, ridge, volatility and quantile baselines |
| Trading baselines | Buy-and-hold and moving-average crossover in the deterministic replay |

Reference Transformer campaigns are restricted to `returns_only_v1` and
`direction_binary_v1` so architecture comparisons share the same 64 numeric
inputs. The tokenizer fits training values only; Transformer checkpoints load
weights-only after hash and column checks. MLP checkpoints use the existing
validated NumPy manifest. Both produce the same raw-output interface.

Targets are listed in [targets.md](../targets.md). Added
`direction_above_1bp_v1`: positive when next-open to following-open log return
exceeds the fixed hypothetical 1 bp hurdle, not guaranteed realized profit.
The generalized replay supports only the binary executed-return direction
head, with an explicit fixed long/flat probability threshold. Other tasks
are predictive experiments, not automatically mapped to orders.
