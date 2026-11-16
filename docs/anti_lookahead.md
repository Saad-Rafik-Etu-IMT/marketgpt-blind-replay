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

## Evidence and limits

The six core tests in [test_no_lookahead.py](../backend/tests/test_no_lookahead.py)
are retained unchanged. Additional chronology, costs and protection regressions
remain in [test_financial_integrity.py](../backend/tests/test_financial_integrity.py).

These tests do not certify every possible transformation or imported series.
Instrument names and dates remain visible; blind-session anonymization is future
work. AI players must be forward-only and receive an explicitly bounded
observation interface when implemented.

This is local educational temporal integrity, not anti-cheat protection. A local
user can inspect their own database. Adjusted data, contract rolls and known
historical outcomes may contaminate an exercise without an API lookahead leak.
A profitable replay or passing suite is not evidence of trading skill.
