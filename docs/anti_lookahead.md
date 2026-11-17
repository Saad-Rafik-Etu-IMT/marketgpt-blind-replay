# Temporal integrity (simulation version 2)

The current replay bar is fully visible. A decision must not execute at a price
available only before that information existed.

## Retained rules

- Visible-bars responses stop at `current_bar_index`.
- Manual orders submitted at N wait until N+1 or later. Markets use the next open,
  not the visible bar's open; limit/stop orders require a later trigger.
- Market-entry protection was precommitted and may execute on the entry bar,
  conservatively stop-first. Limit/stop-entry protection starts on a later bar
  because intrabar sequencing is unknown.
- The pure underlying reference lies inside OHLC. Synthetic spread can move the
  executed price outside it. Commissions are separate cash flows.
- Indicators use visible bars. Risk metrics use snapshots produced up to the
  current replay time.
- Only the replay engine advances time. Backward jumps are rejected; a same-index
  jump is allowed without reopening a completed session. This behavior is unchanged.
- The unrestricted full-history bars endpoint is absent, including for a loaded
  instrument, and has a 404 regression.

See [fill logic](fill_logic.md) for protection timing and accounting assumptions.

## Train/validation/test split (F2)

Boundaries live in [split.json](../backend/config/split.json), never in code, and
apply to every asset on the same calendar dates:

| Set | Dates |
| --- | --- |
| Train | asset start → 2016-12-31 |
| Gap 1 | 2017-01-01 → 2017-03-31 |
| Validation | 2017-04-01 → 2019-12-31 |
| Gap 2 | 2020-01-01 → 2020-01-31 |
| Test | 2020-02-01 → latest available; used once, at the end |

Defaults: `lookback = 64`, `label_horizon = 1`, `embargo = 5` bars.
[split.py](../backend/src/data/split.py) implements the rules below.

- **Example window.** A decision at bar `t` reads inputs `t - lookback + 1 .. t`
  and is labelled by bars `t + 1 .. t + label_horizon`. A train or validation
  example is kept only if the whole window lies inside its own set; examples
  crossing a boundary are dropped, never reassigned.
- **Gap rule: `label_horizon + embargo` bars, not `lookback`.** Because the
  window rule already keeps every input inside its set, the gap only has to stop
  labels from one set sitting next to the start of the next, with `embargo`
  extra bars of margin. Each gap must contain at least that many of the asset's
  bars. Config loading rejects gaps with too few weekdays (an upper bound), and
  the exact count is checked on the asset's bars. A gap is checked only when the
  asset has bars on both sides of it.
- **Test warm-up.** A test decision may read bars before `test_start` as
  look-back context: past data at decision time is legitimate, not leakage.
  Decisions and scoring (decision and label bars) start only at `test_start`.
  Warm-up bars are never test examples, and gap bars are never examples in any set.
- Config validation rejects overlapping or unordered sets, too-short gaps,
  `lookback < 1`, `label_horizon < 1` and `embargo < 0`.

Evidence: [test_split.py](../backend/tests/test_split.py), with synthetic weekday
calendars only (no network, no market data). Not yet covered: F1 data loading and
fitting the tokenizer or normalization on train only. Benchmark scenarios are
covered by [test_scenarios.py](../backend/tests/test_scenarios.py).

## Evidence and limits

The six core tests in [test_no_lookahead.py](../backend/tests/test_no_lookahead.py)
are retained unchanged. Additional chronology, costs and protection regressions
remain in [test_financial_integrity.py](../backend/tests/test_financial_integrity.py).

These tests do not certify every possible transformation or imported series.
Instrument names and dates remain visible in the API; it is not a blind-session
API. The F3 [blind observation](benchmark.md#f3-blind-observation) builder and
F4 [scenario warm-up policy](benchmark.md#warm-up-policy-follows-f2-no-new-rule)
exist, but no AI player uses them yet. AI players must be forward-only and receive
that explicitly bounded observation interface when implemented.

This is local educational temporal integrity, not anti-cheat protection. A local
user can inspect their own database. Adjusted data, contract rolls and known
historical outcomes may contaminate an exercise without an API lookahead leak.
A profitable replay or passing suite is not evidence of trading skill.
