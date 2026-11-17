Data pipeline (F1), split (F2), scenarios (F4) and blind observations (F3); no data is stored here.

- `sources.py`: F1 `PriceSource` interface and the yfinance adapter. Downstream code
  depends only on the interface and its standard columns
  (`date,open,high,low,close,volume,adjusted_close`). yfinance is imported lazily
  (optional `[data]` extra). Stooq is a documented fallback, not implemented.
- `pipeline.py` / `download.py`: F1 licence-gated, all-or-nothing download with
  `provenance.json`; run `python -m src.data.download` from `backend`. Configured by
  [backend/config/data_sources.json](../../config/data_sources.json).
  Raw OHLCV is kept exactly as downloaded for fills; `adjusted_close` is a separate
  return series.
- `split.py`: F2 chronological train/validation/test split, configured by
  [backend/config/split.json](../../config/split.json). Rules and reasoning:
  [temporal integrity, split section](../../../docs/anti_lookahead.md#trainvalidationtest-split-f2).
- `scenarios.py`: F4 fixed calendar half-year scenarios for the SPY + IEF case
  study; `python -m src.data.scenarios` writes frozen manifests to
  [backend/config/scenarios/](../../config/scenarios) using dates and hashes only.
  The test manifest is sealed. See [benchmark scenarios](../../../docs/benchmark.md).
- `anonymizer.py`: F3 blind observation (64 adjusted-close log returns and bar
  offsets) from the replay engine's visible bars; no ticker, date, price level
  or accounting metadata. Not yet wired to any player.
