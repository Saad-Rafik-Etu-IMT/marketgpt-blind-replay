# Configurable MLP

`backend/src/ai/mlp/` is a NumPy multi-layer perceptron with a shared trunk and one linear head per [target](targets.md). It reads the datasets of [datasets.md](datasets.md), trains deterministically and saves safe, self-describing checkpoints. PyTorch is not required.

The team's `learning/mlp/` code is unchanged (decision of 2026-11-20, option 1). Two presets reproduce it and are tested for parity.

## Train one model

```python
from src.ai.mlp import MLP, TrainingData, preset, save, train
from src.datasets import DatasetSpec, build_dataset

ds = build_dataset(frames, DatasetSpec(profile="core_features", target="direction_binary_v1"))
data = TrainingData.from_datasets([ds])          # inner_train / inner_validation
architecture, config = preset("medium")
model = MLP(data.x_train.shape[1], architecture, seed=0)
result = train(model, data, config)               # best epoch restored in model.params
save(Path("data/models/medium"), model, config, dataset=ds,
     target_states=data.target_states, history=result.history, best_epoch=result.best_epoch)
```

`train` never sees the validation part. The selection set (inner validation) picks the epoch.

## Architecture

Block `i`: `z = h @ W + b`, then optional norm, then activation, then optional residual, then inverted dropout. After the last block, an optional input shortcut `scale * x @ W_skip` is added. Each head is linear.

| Field | Values |
| --- | --- |
| `hidden` | widths, for example `(64, 32)`; `()` gives a linear model |
| `activation` | `relu`, `gelu` (tanh approximation), `tanh` |
| `norm` | `none`, `batch` (momentum 0.1, eps 1e-5, running statistics at inference), `layer` |
| `dropout` | one rate, or one per hidden layer |
| `residual` | adds the block input when the width is unchanged |
| `input_skip` | scale of the linear input shortcut (`deep_mlp` uses 0.35) |
| `output_init` | `he` (N(0, 2/fan_in), W1) or `lecun` (N(0, 1/fan_in), `deep_mlp`) |
| `heads` | one `Head(target, weight)` per T4 target |

Hidden weights use He-normal initialization from the model seed. Sequence inputs `[N, T, F]` are flattened to `T x F` columns.

## Training

| Field | Values |
| --- | --- |
| `optimizer` | `sgd` (optional momentum) or `adamw` (beta 0.9 / 0.999, eps 1e-8) |
| `weight_decay` | decoupled; weight matrices only, never biases or norm parameters |
| `schedule` | `constant` or `cosine` down to `schedule_floor`, optional `warmup_epochs`, set per epoch |
| `clip` | `none`, `global_norm` or `value` (element-wise, as in `deep_mlp`) |
| `patience`, `min_delta` | early stopping on the selection loss; the best epoch is always restored |
| `seed` | drives epoch shuffles and dropout masks; the model seed drives initialization |

Batches of one row are skipped with BatchNorm. A non-finite output, loss, gradient or parameter raises `TrainingDivergedError` with the history so far.

Each epoch records: learning rate, total and per-head losses on training and selection rows, the cosine similarity between heads' trunk gradients, and `selected` for the restored epoch.

## Presets

| Preset | Trunk | Training | Role |
| --- | --- | --- | --- |
| `w1_baseline` | 64 inputs, 32 ReLU | SGD 0.05, batch 64, 40 epochs | permanent baseline; reproduces `step8_train.py` |
| `small` | 32 | AdamW 1e-3, decay 1e-2, cosine to 10%, dropout 0.1, global-norm clip 1, patience 7 | candidate |
| `medium` | 64, 32 | same | candidate |
| `large` | 128, 64, 32 | same | candidate |
| `deep_v1` | 128, 64, 32, BatchNorm, dropout 0.25 / 0.25 / 0, input shortcut 0.35 | AdamW 1.5e-3, decay 2e-3, cosine to 15%, value clip 2, 40 epochs | the architecture of `deep_mlp.py` |

The candidates are starting points, not results. T7 compares them.

`deep_v1` reproduces the `deep_mlp` network, not its preprocessing. `deep_mlp` builds its own features with `extract_blind_features` and a robust scaler; here the inputs come from a feature profile.

## Multi-task

Give several heads. The total loss is `sum_k weight_k * loss_k`. Continuous targets are standardized in T4, so losses are comparable without extra normalization.

The datasets of all heads must share their rows. Build them with the same spec, `min_horizon` set to the longest target horizon and `min_lookback` covering every target:

```python
targets = ("direction_binary_v1", "expected_return_h5", "volatility_h5")
common = dict(profile="core_features", min_horizon=6, min_lookback=251)
data = TrainingData.from_datasets([build_dataset(frames, DatasetSpec(target=t, **common)) for t in targets])
```

`from_datasets` refuses datasets whose rows or inputs differ. `gradient_cosine` in the history shows whether heads pull the trunk in opposite directions (negative values suggest negative transfer). Multi-task is not a default: T7 must show it beats the single-task models.

## Checkpoints

`save(path, ...)` writes `path.npz` (float arrays) and `path.json` with these fields:

- schema and compatibility version, model id, creation time, code revision;
- architecture, training configuration, seed, input width, parameter count;
- feature ids, versions, exact column order and catalogue fingerprint;
- representation, scaler, fitted target states;
- dataset fingerprint, split and selection rules;
- history, best epoch, SHA-256 and shapes of the weights.

`load(path, expected_columns=...)` uses `np.load(allow_pickle=False)` and refuses:

- another schema or compatibility version;
- weights whose hash differs from the manifest;
- shapes that do not match the architecture;
- a column order different from training.

Inputs are never reordered, dropped or substituted. `LoadedModel.predict_raw_features` applies the saved scaler, then the network.

## Tests

`backend/tests/test_mlp.py`:

- **W1 parity**: `w1_baseline` reproduces `step8_train.train`. Same weights to 1e-12 after 6 epochs, same training and selection losses per epoch, same best epoch.
- **deep_mlp parity**: `deep_v1` with the same seed has identical initial weights. With the same dropout seed it gives the same logits, the same gradients for every parameter and the same BatchNorm running statistics.
- **Gradients**: finite differences for ReLU, GELU with BatchNorm, tanh with LayerNorm and residual, input shortcut, a linear model, four heads at once, and BatchNorm in inference mode.
- **Training**: a known synthetic signal is learned (AUC > 0.8); pure noise stays at AUC 0.5 ± 0.1; same seeds give identical bits; early stopping restores the best epoch; divergence is detected.
- **Multi-task**: per-head losses, gradient cosines in [-1, 1], misaligned datasets refused.
- **Checkpoints**: round trip with identical predictions; refusal of reordered columns, modified weights, pickled objects, another schema and mismatched shapes.

## First run on real data (inner validation only)

Six ETFs pooled, target `direction_binary_v1`, `min_lookback=251`, seed 0. Inner train has 16,695 rows and inner validation 5,424 rows. The validation part was not read. One seed is not evidence; T7 runs the proper comparison.

| Preset | Profile | Parameters | Best epoch | Inner-validation log loss | Constant | AUC | Share predicted up |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `w1_baseline` | `returns_only_v1` | 2,146 | 2 of 40 | 0.6949 | 0.6919 | 0.492 | 91% |
| `small` | `core_features` | 2,530 | 3 of 10 | 0.6979 | 0.6919 | 0.504 | 78% |
| `medium` | `core_features` | 7,074 | 2 of 9 | 0.6968 | 0.6919 | 0.497 | 90% |
| `large` | `core_features` | 20,258 | 1 of 8 | 0.6925 | 0.6919 | 0.502 | 98% |
| `deep_v1` | `core_features` | 23,138 | 3 of 40 | 0.6944 | 0.6919 | 0.509 | 80% |

No preset beats the constant training up-share on log loss. AUC stays within 0.49 to 0.51. This agrees with `docs/training_audit.md`: no detectable next-day signal. The best epochs are early, then the models fit noise. Bigger models change nothing measurable.
