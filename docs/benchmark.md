# Benchmark scenarios, blind observations and execution (W2, Stages 1–2)

Stage 1 implements F3 (blind observations) and F4 (scenario definitions and
frozen manifests). Stage 2 implements the F5 player interface, the execution
adapter, the fill-time affordability check and the benchmark profile.
Baselines, training, the evaluation runner and any results are **not
implemented**; their agreed decisions are recorded below.

## F4: scenarios

Rules live in [scenarios.json](../backend/config/scenarios.json); the code is
[scenarios.py](../backend/src/data/scenarios.py).

- **Two-asset case study:** SPY (broad US equity) and IEF (intermediate US
  Treasuries), predeclared for asset-class diversity before any returns were
  inspected. Model training later uses the configured 15-ETF universe; results
  on two assets are a case study, not a claim about ETFs in general.
- **Windows:** complete, fixed calendar half-years (Jan–Jun, Jul–Dec), shared by
  every instrument, keeping each window's actual trading bars.
- **Sets:** `development` windows come from the validation split; `test` windows
  from the test split. Development scenarios are for debugging, UI work and
  demos; the test set is sealed (below).
- **Selection uses dates only.** The generator hashes each CSV against
  `provenance.json` and parses only its `date` column. Prices, returns,
  volatility or model results are never read, so they cannot influence the
  selection; generation prints counts, dates and exclusion reasons only.
- **Coverage rules**, applied per window and instrument, every rejection recorded
  in the manifest's `exclusions` with its reason:
  1. the calendar window lies fully inside the set's split;
  2. the dataset has a bar after the window end (otherwise coverage is incomplete);
  3. at least `min_scoring_bars` (100) scoring bars;
  4. `lookback` (64, from [split.json](../backend/config/split.json)) warm-up bars
     exist before the first scoring bar, under the set's warm-up policy;
  5. no calendar gap over `max_calendar_gap_days` (7) across warm-up and scoring.

  Rules 2, 3 and 5 are **provisional completeness heuristics**, not proof of a
  complete exchange trading calendar: they detect truncated or visibly gapped
  data but cannot detect, for example, a single missing session. No exchange
  calendar is checked.
- **Expected, not guaranteed, counts:** 24 test and 8 development scenarios;
  regeneration on a different download may differ, and exclusions say why.

### Warm-up policy (follows F2, no new rule)

The [F2 split rules](anti_lookahead.md#trainvalidationtest-split-f2) already say:
a validation example keeps its whole input window inside validation; a test
decision may read pre-test bars as look-back context. Scenarios apply exactly this:

| Set | `warmup_policy` | Meaning |
| --- | --- | --- |
| development | `inside_split` | all 64 warm-up bars lie inside validation (2017H2 is excluded for this reason) |
| test | `pre_split_context_allowed` | warm-up may precede `test_start`; recorded per scenario as `warmup_crosses_split_boundary` |

With the current data, no test warm-up actually crosses `test_start`.
Warm-up bars are read-only context: they never create orders, decisions, scored
outcomes, fitting examples or preprocessing-fit data (later stages must exclude
them; training uses F2 train examples only).

### Indexing (session index 0 = `warmup_start`)

| Index | Meaning |
| --- | --- |
| `0 .. 63` | warm-up: visible context, no decision, no order |
| `64` = `first_decision_index` | `scoring_start`; first decision, at its close |
| `64 .. total_bars - 2` | decision bars; an order decided at N executes at N+1 or later |
| `total_bars - 1` = `last_bar_index` | `scoring_end`; fills and final mark only, no decision |

The first decision's observation holds 64 log returns computed from the
**65 closes** at indices `0 .. 64`. No execution, label or outcome extends
past `scoring_end`. Regression tests: `test_off_by_one_first_decision_needs_lookback_plus_one_closes`
and `test_first_scored_decision_index_through_replay_engine` in
[test_anonymizer.py](../backend/tests/test_anonymizer.py), and
`test_off_by_one_warmup_and_decision_indices` in
[test_scenarios.py](../backend/tests/test_scenarios.py).

### Manifests and sealing

`python -m src.data.scenarios` (from `backend`, after the F1 download) writes
[development.json](../backend/config/scenarios/development.json) and
[test.json](../backend/config/scenarios/test.json). Manifests hold dates, counts
and hashes, not market data. Each scenario records `scenario_id`, `instrument`,
`calendar_group`, calendar window, `warmup_start/end`, `scoring_start/end`,
`warmup_bars`, `scoring_bars`, `total_bars`, the decision indices,
`instrument_file_sha256`, `dataset_fingerprint` (the whole frozen download),
`split_config_sha256` and `scenario_config_sha256`.

- **Frozen:** regeneration that would change an existing manifest fails unless
  `--replace` is passed. `--replace` is only for correcting a manifest before any
  model has been evaluated on that snapshot. A test checks the committed
  manifests still match the current split and scenario configs.
- **Sealed:** `load_manifest` refuses the test manifest unless
  `allow_sealed_test=True`. Generating it produces no observations or performance
  summaries. Test scenarios must not be used for UI iteration, threshold
  selection or debugging; single final use remains a procedural rule.
- **Statistics (later):** SPY and IEF in the same `calendar_group` share dates
  and are not independent samples; uncertainty must resample calendar groups.
  With 4 development and 12 test groups, uncertainty must be reported cautiously.
- **Not player-facing:** scenario IDs (which contain the ticker), calendar groups
  and all other manifest metadata must never enter blind player observations or
  any blind human-rating interface. Opaque display aliases can be added when
  that interface is implemented.

### Dataset versioning

- The manifests describe one frozen download, identified by `dataset_fingerprint`
  (source, library version, retrieval time and every file's hash and row range).
- Generation already fails clearly if a CSV no longer matches its
  `provenance.json` hash or the provenance split config differs from `split.json`.
  Any later consumer (the planned runner) must likewise fail with a clear error
  when the local dataset's fingerprint differs from the manifest's; it must never
  run on mismatched data.
- A revised download (adjusted prices change with every new distribution) is a
  **new dataset version** and requires new manifests, written alongside the
  existing ones (e.g. a new `--out-dir`), never over them. Never silently replace
  the frozen benchmark snapshot; results always name the fingerprint they used.
- Market data is never committed or redistributed; manifests contain only dates,
  counts and hashes. Data-sharing permissions remain subject to the provider's
  terms, so other team members obtain their own download, which will normally
  carry a different fingerprint and therefore need its own dataset version.

## F3: blind observation

[anonymizer.py](../backend/src/data/anonymizer.py) turns the replay engine's
bounded visible bars into a `BlindObservation` with exactly two fields:

- `log_returns`: the 64 most recent `ln(adjusted_close[t] / adjusted_close[t-1])`,
  oldest first, unit "natural log return, close to close";
- `bar_offsets`: `-63 .. 0`, relative to the decision bar.

Ticker, dates, scenario identity, raw price levels and accounting metadata
(multiplier, contract kind, cash, position) stay outside the observation.
`adjusted_close` is validated first: missing, non-finite or non-positive values
in the window are rejected, as are unordered bars and short histories.

Synthetic example (`lookback = 4`, adjusted closes 100, 101, 99.5, 100.2, 102):

```json
{
  "log_returns": [0.00995033, -0.01496287, 0.00701054, 0.01780462],
  "bar_offsets": [-3, -2, -1, 0]
}
```

**Limits.** Anonymization reduces identity cues; it does not guarantee that a
historical period cannot be recognised from its return pattern. Adjusted returns
do not solve corporate-action consistency between learning data and raw-price
execution: fills and accounting keep using raw OHLC, and the two series can
differ around dividends and splits (see the blocker below).

## Blocker for real-data benchmark results: corporate actions

Players observe split- and dividend-adjusted returns, but the replay executes
and marks positions at Yahoo's raw OHLC (`auto_adjust=False`), which is
split-adjusted retroactively and **not** dividend-adjusted. The replay has no
distribution or corporate-action cash flows. Consequences on real data:

- on each ex-distribution date the raw close drops by the distribution and a held
  position loses that value with no matching cash credit, so holding strategies
  (including buy-and-hold) are understated versus total return; IEF pays monthly
  distributions, SPY quarterly, so the bias differs by asset;
- split-adjusted raw prices are not the historically traded prices or share
  counts, and a later re-download can restate them;
- the player's input series and the accounting series can disagree on the same day.

**No real-data benchmark result may be reported until an explicit, consistent
policy for splits and distributions is approved and tested**, for example
crediting distributions as cash on ex-dates, or executing and marking on a single
consistently adjusted series, documented with its own limitations. Synthetic
tests are unaffected.

## Players and identifiers

Scenario IDs, instruments, dates, calendar groups and any other manifest metadata
stay outside blind player observations and any future blind human-rating
interface; opaque display aliases are added when that interface is built.

## Stage 2: players and execution (F5)

### Player interface

[player.py](../backend/src/ai/player.py): `Player.decide(BlindObservation) ->
PlayerDecision`. A decision is a binary target, `long` or `flat`, plus optional
`BinaryProbabilities(long, flat)`: each finite, in `[0, 1]`, summing to 1 within
`1e-6` (the cockpit's tolerance). They are model class probabilities, not
probabilities of profit. Rule-based players return `probabilities=None`; none
are invented. No short target or short probability exists. A label definition
(what a model is trained to predict) stays separate from this target policy, and
classification metrics stay separate from trading metrics.

### Benchmark profiles (provisional)

Profiles live in [benchmark_profiles/](../backend/config/benchmark_profiles).
Each is identified by `profile_id` and a SHA-256 of its content; a profile is
never edited in place, a new policy gets a new profile.

| Profile | Status | Sizing | Use |
| --- | --- | --- | --- |
| [benchmark-v1](../backend/config/benchmark_profiles/benchmark-v1.json) | superseded, kept unchanged | no cash reserve | Stage 2 records; reproduces the earlier sizing |
| [benchmark-v2](../backend/config/benchmark_profiles/benchmark-v2.json) | **default** | `cash_reserve_fraction = 0.005` | all later benchmark work |

Both share: initial cash 100,000; zero commission; `spread_bps = 1`, the
**full** synthetic bid–ask spread, so each fill pays 0.5 bp; one open position;
per-trade risk and exposure limits 100%; drawdown limit 100% (auto-pause only at
a total loss). v1 has no `cash_reserve_fraction` field, and its hash is computed
exactly as before (`a96bad18…`), so v1 records keep their meaning; a test pins it.

`start_benchmark_session` creates the session with the profile's costs and
applies the limits through the existing `risk_service.update_limits`;
`verify_profile` runs before every decision and refuses on any mismatch. These
are experimental settings shared by every player, not realistic cost or risk
estimates and not a safety or execution guarantee. The default manual Replay
limits are unchanged.

### Target policy and sizing

[execution.py](../backend/src/eval/execution.py) is the trusted adapter; the
player never sees it. At most one market order per decision, via the existing
order service (pre-trade checks unchanged) and executed at N+1 or later:

| Target | Position | Pending top-level order | Action |
| --- | --- | --- | --- |
| any | any | yes | `none_pending_order` |
| long | flat | no | `submit_buy`, or `none_zero_quantity` |
| long | long | no | `none_already_long` (no rebalancing) |
| flat | long | no | `submit_sell` of the whole position |
| flat | flat | no | `none_already_flat` |

A pre-trade refusal is recorded as `pre_trade_rejected` with its message.
Decisions are refused during warm-up, after `last_decision_index` and on a
completed session.

Sizing uses bar-N information only. The profile's reserve is taken off first,
then the cost-aware formula applies:

```text
budget   = cash × (1 − cash_reserve_fraction)
unit     = close_N × multiplier × (1 + spread_bps/10000) + 2 × fee_per_unit
quantity = floor(budget / unit)        (whole units; 0 submits nothing)
```

The unit cost reserves the full spread and round-trip fees, which keeps the
existing pre-trade risk, exposure and cash estimates within cash. With
`cash_reserve_fraction = 0` this is exactly the benchmark-v1 policy (tested).
`cash_reserve_fraction` must be finite and in `[0, 1)`; other values are refused
when the profile loads and by the sizing function.

The reserve is **headroom, not an execution guarantee**. With benchmark-v2 a next
open up to roughly 0.5% (plus the 0.5 bp unused spread reserve and whole-unit
rounding slack) above `close_N` still fits; a larger gap is rejected at fill time
(see [fill logic](fill_logic.md#fill-time-affordability-fill-time-cash-check-v1))
with no fill, resizing, partial fill or limit-order substitution. The player may
decide again on a later bar. Repeated Long never rebalances, so the reserve and
any rounding remainder stay as cash while long. No borrowing, leverage or
negative cash on the supported paths (cash instruments, zero commission).

Background: under benchmark-v1 the only headroom was the unused spread and the
rounding remainder, which varies with the price level, so all-in entries were
rejected after almost any gap up (observed on synthetic rising bars).

### Records (schema version 2)

Two separate JSON-serialisable records; neither is player-facing.

- `DecisionRecord`, written once at decision time and never updated:
  `session_id`, `player_name`, `decision_bar_index`, `observation`
  (`log_returns`, `bar_offsets`), `target`, `probabilities` (`{long, flat}` or
  null), `execution_context` (`cash`, `position_side`, `position_quantity`,
  `reference_close`, `multiplier`, `pending_order_id`), `action`, `order_id`,
  `order_side`, `order_quantity`, `pre_trade_rejection`,
  `execution_policy_version`, `risk_profile_id`, `risk_profile_sha256`,
  `cash_reserve_fraction` (the reserve the sizing actually applied; 0.0 for v1).
  Version 2 added `cash_reserve_fraction`; no version 1 records were persisted.
- `ExecutionOutcome` (schema version 1), read later from the order and its fill:
  `order_id`, `decision_bar_index`, `observed_at_bar`, `status`,
  `rejection_reason`, `fill_bar`, `fill_price`, `fill_quantity`, `fee`.

Explanation references are not part of these records yet; the explanation
pipeline is out of scope.

## Recorded decisions for later stages (not implemented)

- **Prediction task is binary for the MVP:** long/flat. Labels, output head,
  probabilities, calibration and explanations are two-class. No short probability
  is fabricated for the existing three-class cockpit contract; frontend
  integration is a separate follow-up. Long/flat/short is a later extension.
- **Run metadata** (Stage 4) must record `execution_policy_version`
  (`fill-time-cash-check-v1`), the risk profile ID, hash and
  `cash_reserve_fraction`, and the dataset fingerprint.
- **Evaluation reporting** (Stage 4 runner; not implemented): per player and
  scenario, and aggregated by calendar group, the runner must be able to report
  - entry attempts (submitted buy orders), separately from decisions with a Long
    target;
  - the count of fill-time `INSUFFICIENT_CASH` rejections, and of pre-trade
    rejections;
  - the first successful entry bar (index and offset from `first_decision_index`),
    or none;
  - exposure: fraction of scored bars with an open position, and average
    invested fraction of equity (the reserve keeps this below 100%).
  All of it is derivable from `DecisionRecord` and `ExecutionOutcome`.
- **Baselines:** buy-and-hold; MA 20/50 crossover on a relative price path;
  small NumPy MLP (20 returns → 16 ReLU → 2 softmax). The MLP label definition is
  distinct from its target-position policy, and classification metrics stay
  separate from trading metrics. Settings and seeds are configured and recorded,
  chosen with train/validation data only.

## Open follow-ups (not completed)

- **Corporate actions:** still a [blocker](#blocker-for-real-data-benchmark-results-corporate-actions)
  for any real-data benchmark result.
- **Frontend `rejection_reason`:** exposed by the backend `OrderRead`; the
  frontend types and research cockpit do not use it yet.
- **LLM explanations (F16):** no explanation references in records and no
  generation pipeline.

## Stage 1 verification record — 2026-11-16

Local run, Python virtual environment at `.venv`, after the Stage 1 changes:

| Check | Result |
| --- | --- |
| `python -m pytest` (backend) | 348 passed; total coverage 95.87% in the final reported run (gate 85%) |
| `python -m ruff check src tests` | Passed |
| `python -m ruff check .` (whole backend) | Fails: 24 pre-existing findings in `backend/learning/mlp/` (W1 scripts, unchanged) |
| `python -m mypy` on the two new modules | No findings in them; strict mypy still reports pre-existing errors in other modules (`data_validator.py`, missing pandas stubs) |
| `python -m src.data.scenarios` re-run | Both manifests reported unchanged |

The repository as a whole is **not** lint- or type-check clean. These dated local
results are not a CI report or a guarantee for later edits.

## Stage 2 verification record — 2026-11-17

| Check | Result |
| --- | --- |
| `python -m pytest` (backend) | 375 passed (27 new); total coverage 96.15% (gate 85%) |
| `python -m ruff check src tests` | Passed |
| `python -m ruff check .` (whole backend) | Fails: the same 24 pre-existing findings in `backend/learning/mlp/` |
| `alembic upgrade head` + `alembic check` on a fresh target-only SQLite file | 0001 → 0002 applied; no drift detected |
| strict mypy on new modules (`execution.py`, `player.py`) | No findings |
| strict mypy on `broker.py` | Pre-existing `Optional` session errors; the new check lines inherit the same unchecked `db.get` result |

The repository as a whole is still **not** lint- or type-check clean.

## Stage 2 follow-up verification record — 2026-11-17 (benchmark-v2 sizing)

| Check | Result |
| --- | --- |
| `python -m pytest` (backend) | 386 passed (11 new); total coverage 96.16% (gate 85%) |
| `python -m ruff check src tests` | Passed |
| `python -m ruff check .` (whole backend) | Fails: the same 24 pre-existing findings in `backend/learning/mlp/` |
| strict mypy on `src/eval/execution.py` | No findings |

No migration was added in this follow-up.
