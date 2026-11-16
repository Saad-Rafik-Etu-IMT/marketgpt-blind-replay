# Fill Logic

This document explains how the simulated broker decides whether and at
what price each order type fills against a single OHLCV bar.

All logic lives in `src/core/fill_logic.py` — a pure module with no
I/O. The broker (`src/services/broker.py`) calls it for every pending
order on every bar transition.

## Rules at a glance

| Order type   | Fills on   | Trigger condition        | Fill price                |
| ------------ | ---------- | ------------------------ | ------------------------- |
| Market       | N+1        | next bar available       | `bar.open`                |
| Limit buy    | N+1..      | `bar.low ≤ limit_price`  | `min(bar.open, limit)`    |
| Limit sell   | N+1..      | `bar.high ≥ limit_price` | `max(bar.open, limit)`    |
| Stop buy     | N+1..      | `bar.high ≥ stop_price`  | `max(bar.open, stop)`     |
| Stop sell    | N+1..      | `bar.low ≤ stop_price`   | `min(bar.open, stop)`     |

Where "N" is the fully visible bar at which the order was submitted. **Manual
orders never fill on their submission bar** — this is §5
invariant 2.

The pure reference is within `[bar.low, bar.high]`. The broker then applies
the synthetic spread (half on each side), so cash execution may be outside
that range. Limit orders use synthetic quote touches and respect their limit.
Commissions are separate cash flows per unit per side. **Reference price is inside
OHLC; executed price may be outside OHLC after synthetic spread.** Extraction
preserves this distinction and does not clamp executed prices back into the bar.

## Examples

### Market buy at the open

Bar: `open=100, high=105, low=98, close=102`
Order: market buy submitted after bar `N=4`, executed on bar `5`.
Result: **reference fill at 100**, plus synthetic half-spread and commission.
An order submitted at bar 5 instead waits for bar 6.

### Limit buy with intrabar trigger

Bar `N+1`: `open=100, high=105, low=98, close=102`. Limit-buy at 99.
- `bar.low (98) ≤ 99` → trigger.
- Fill price = `min(100, 99) = 99`.

### Limit buy with gap-down (open already below limit)

Bar `N+1`: `open=95, high=100, low=94, close=98`. Limit-buy at 99.
- `bar.low (94) ≤ 99` → trigger.
- Fill price = `min(95, 99) = 95`. The order fills *better* than the
  limit because the market gapped through it on the open.

### Limit buy that does not fill

Bar `N+1`: `open=102, high=105, low=100, close=104`. Limit-buy at 99.
- `bar.low (100) > 99` → no trigger. Order remains pending.

### Stop buy: triggered, no gap

Bar `N+1`: `open=100, high=105, low=98, close=102`. Stop-buy at 104.
- `bar.high (105) ≥ 104` → trigger.
- Fill price = `max(100, 104) = 104`.

### Stop buy: gap-up beyond the stop

Bar `N+1`: `open=108, high=110, low=106, close=109`. Stop-buy at 104.
- `bar.high (110) ≥ 104` → trigger.
- Fill price = `max(108, 104) = 108`. The price gapped past the stop
  on the open — this is realistic slippage.

### Stop sell: triggered, no gap

Bar `N+1`: `open=100, high=105, low=98, close=102`. Stop-sell at 99.
- `bar.low (98) ≤ 99` → trigger.
- Fill price = `min(100, 99) = 99`.

### Stop sell: gap-down beyond the stop

Bar `N+1`: `open=95, high=100, low=90, close=92`. Stop-sell at 99.
- `bar.low (90) ≤ 99` → trigger.
- Fill price = `min(95, 99) = 95`. The price gapped past the stop —
  realistic adverse slippage.

## Anti-lookahead

Two rules in this module enforce §5 invariants:

1. **`current_bar > submitted_at_bar`** for manual orders. Without
   this, a strategy could use the bar's intra-day data as the basis
   for placing the order — pure lookahead.
2. **`bar.low ≤ reference_price ≤ bar.high`** in the pure engine. If
   a computation ever leaves the range (a code bug, not a data issue),
   the function raises rather than write a corrupt fill.

Tests for both live in `tests/test_no_lookahead.py`.

## OCO (one-cancels-other)

When a parent order fills with both `stop_loss` and `take_profit` set:

1. A child **stop order** on the opposite side is created at
   `stop_loss`.
2. A child **limit order** on the opposite side is created at
   `take_profit`.
3. Both children share an `oco_pair_id`.
4. On the first child fill, the broker sets the sibling's status to
   `cancelled`.

If only `stop_loss` (or only `take_profit`) is provided, a single
lone child is created with no `oco_pair_id`.

Market entries execute at the open: precommitted protections can execute on
that bar, stop-first if both levels are touched. Limit/stop entries have unknown
intrabar timing: children are eligible on the following bar only. This can miss
entry-bar risk; daily bars are not a tick model. Protective/manual reducing
exits cannot exceed the remaining position or open a reverse position.

Pre-trade budgets use visible prices; gaps can exceed them. Completion cancels
pending orders and marks open positions without forced liquidation. See
[temporal integrity](anti_lookahead.md).
