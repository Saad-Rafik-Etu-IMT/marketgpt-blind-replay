# Experiments

`backend/src/experiments/` runs multi-seed campaigns on top of the [features](features.md), [datasets](datasets.md), [targets](targets.md) and [MLP](mlp.md). The `/api/experiments` routes expose it.

It runs alongside `/api/studies` and does not replace it. The study worker (`learning/mlp/run_study.py`), its routes and every saved study are unchanged.

## An experiment

One experiment trains one multi-head model (one head per target) for every combination of feature profile, preset and seed.

```json
{
  "name": "volatility features vs baseline",
  "profile": "research",
  "sources": ["SPY", "IEF", "QQQ", "XLE", "GLD", "EEM"],
  "primary_asset": "SPY",
  "feature_profiles": ["returns_only_v1", "returns_plus_volatility"],
  "presets": ["small"],
  "targets": ["direction_binary_v1"],
  "context": "own_partition",
  "scaler": "standard"
}
```

Optional fields: `epochs` (at most the profile maximum), `seeds` (exactly the profile's count), `report_validation`, `max_minutes`.

`sources` lists local CSVs from the F1 download, or `["synthetic"]` for two generated assets covering 2005 to 2019.

## Profiles

| Profile | Seeds | Max epochs | Patience | Walk-forward folds | Validation read |
| --- | ---: | ---: | --- | ---: | --- |
| `development` | 1 | 10 | the preset's | 0 | only if `report_validation` |
| `research` | 5 | 40 | 7 when the preset has none | 0 | only if `report_validation` |
| `confirmation` | 10 | 60 | 7 when the preset has none | 3 | always |

These are configurable starting points, not validated optimums.

The spec is stored with its SHA-256 fingerprint. A worker refuses a `spec.json` whose content no longer matches its fingerprint, which locks the protocol once launched.

## Limits

- At most 8 feature profiles, 5 presets, 4 targets and **60 runs** per experiment.
- **Compute budget.** `estimate` computes about `6 x training rows x parameters x epochs x (1 + folds)` operations, calibrated at 3e9 operations per second. An experiment above 4 hours is refused. It is an order of magnitude: early stopping usually ends runs sooner.
- **One experiment at a time** (bounded concurrency).
- Optional `max_minutes`: the worker stops between epochs when it is exceeded (`stopped`).

## What a run does

1. **Datasets.** One dataset per target, built once per feature profile. Every feature profile and target of the experiment uses the same rows: a common `min_lookback` (the longest profile lookback, at least 20 when a target needs sigma) and a common `min_horizon` (the longest target horizon).
2. **Training.** Train on `inner_train` and pick the epoch on `inner_validation`, with the preset's configuration capped by the profile and seeded per run.
3. **Evaluation.** Every head's metrics on `inner_validation` (T4 `evaluate`).
4. **Validation**, only if the spec reads it. Each read is appended to `<data_dir>/experiments/validation_reads.json` (experiment, run, time, dataset and spec fingerprints), so repeated looks at validation stay visible.
5. **Walk-forward** (confirmation). 3 expanding folds inside the training rows, from 40% of the training dates. Each fold has a purged and embargoed training side, refits its own scaler, trains a fresh model for the run's best epoch count without early stopping, and reports every head.
6. **Checkpoint and report.** The checkpoint is written (T5, with manifest), then the run report.

A run that raises (divergence, invalid data) is recorded with its error and the campaign continues (`complete_with_failures`). Cancellation is checked after every epoch.

## Files

```text
<data_dir>/experiments/
  validation_reads.json
  <uuid>/
    spec.json            spec + fingerprint
    status.json          state, done/total, failed runs, current run and epoch
    cancel               present = cancellation requested
    worker.log, worker_error.txt
    datasets/<profile>__<target>.json   dataset manifests
    runs/<profile>__<preset>__s<seed>.json   run reports
    checkpoints/<profile>__<preset>__s<seed>.{npz,json}
```

States: `queued`, `running`, `complete`, `complete_with_failures`, `cancelled`, `stopped`, `failed`, and `interrupted` (reported by the API when no worker is alive). Resuming skips runs already complete under the same spec fingerprint.

Headless run, from `backend/` with a folder holding `spec.json`:

```bash
python -m src.experiments.worker data/experiments/<uuid>
```

## API

| Method and path | Purpose |
| --- | --- |
| `GET /api/experiments/catalog` | features, feature profiles, targets, presets, experiment profiles, context rules, scalers |
| `GET /api/experiments/features/{id}` | one feature's definition and redundancies |
| `GET /api/experiments/sources` | data-source health: rows, dates, file SHA-256, match with `provenance.json` |
| `GET /api/experiments/sources/{source}/preview?feature=&last=` | last feature values, never after `validation_end` |
| `POST /api/experiments/estimate` | cost estimate for a spec |
| `POST /api/experiments` | validate, estimate, start (202); 409 if one is running, 422 over budget |
| `GET /api/experiments` | list with status |
| `GET /api/experiments/{id}` | spec, fingerprint, status |
| `POST /api/experiments/{id}/cancel` | request cancellation |
| `POST /api/experiments/{id}/resume` | restart an unfinished experiment (409 if complete or busy) |
| `GET /api/experiments/{id}/runs` | run summaries |
| `GET /api/experiments/{id}/runs/{key}` | full run report, history included |
| `GET /api/experiments/{id}/compare?part=inner_validation` | per feature profile and preset: mean, std, min, max and count of every numeric metric over seeds; `part=validation` only for runs that read it |

Identifiers are UUIDs and run keys match `^[a-z0-9_]+__[a-z0-9_]+__s[0-9]+$`; no request value becomes a path. Source names are checked against the local folder.

## First research experiment (inner validation only)

Six ETFs, primary SPY, `direction_binary_v1`, preset `small`, profile `research` (5 seeds), common lookback 271. Inner train has 16,599 rows and inner validation 5,280. 15 runs in 17 s through the subprocess worker. Validation was not read.

| Feature profile | Log loss (mean ± std, 5 seeds) | Constant | AUC (mean ± std) | Best epochs |
| --- | --- | ---: | --- | --- |
| `returns_only_v1` | 0.6958 ± 0.0019 | 0.6918 | 0.505 ± 0.006 | 4, 4, 4, 4, 4 |
| `returns_plus_volatility` | 0.6961 ± 0.0026 | 0.6918 | 0.509 ± 0.010 | 4, 5, 6, 4, 4 |
| `core_features` | 0.6973 ± 0.0012 | 0.6918 | 0.504 ± 0.004 | 4, 4, 4, 4, 4 |

No profile beats the constant training up-share. The differences between profiles are smaller than the spread across seeds. Formal comparisons (paired intervals, baselines, calibration) are tranche T7.

## Tests

`backend/tests/test_experiments.py`:

- **Spec**: invalid specs, including grids above 60 runs; profile caps on seeds, epochs and patience; fingerprints.
- **Sources**: synthetic coverage of the split; health with provenance match; unreadable files and unknown sources.
- **Estimate**: cost and budget fields.
- **Runner**: end-to-end run with a checkpoint that reloads; profiles sharing the same rows; resume skipping completed runs; failure isolation; cancellation; time budget; crash report; refusal of a tampered spec.
- **Confirmation**: walk-forward folds are chronological and growing, and validation reads are logged.
- **Worker and API**: the worker entry point; every route, including 404, 409 and 422 cases and path-like run keys.

## Changes outside `src/experiments`

- `src/main.py`: two lines to register the router (not in jury-only mode).
- `src/ai/mlp/train.py`: optional `on_epoch` callback, used for progress, cancellation and time budget.
