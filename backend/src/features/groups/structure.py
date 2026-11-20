"""Group D: price structure of the current bar and its place in recent ranges.

All values are ratios of adjusted OHLC fields of the same bar(s), so no price
level is exposed. A zero-range session (high == low) has no wick or body: the
ratios are defined as 0 and the close location value as 0 (neutral).

Covered elsewhere, not duplicated here: open-to-close return is
``intraday_return`` and the overnight gap is ``overnight_return`` (Group A).
"""

from __future__ import annotations

import numpy as np

from src.features.groups._window import frame, rolling, safe_ratio
from src.features.spec import FeatureGroup, FeatureSpec, Visibility

OHLC = ("adj_open", "adj_high", "adj_low", "adj_close")
RANGE_SPANS: tuple[int, ...] = (20, 60)


def _parts(bars):
    return tuple(bars[c].to_numpy() for c in OHLC)


def _hl_range(bars, _):
    _, high, low, close = _parts(bars)
    return frame(bars.index, hl_range_pct=(high - low) / close)


def _clv(bars, _):
    _, high, low, close = _parts(bars)
    value = safe_ratio((close - low) - (high - close), high - low, when_zero=0.0)
    return frame(bars.index, clv=value)


def _upper_wick(bars, _):
    open_, high, low, close = _parts(bars)
    value = safe_ratio(high - np.maximum(open_, close), high - low, when_zero=0.0)
    return frame(bars.index, upper_wick_ratio=value)


def _lower_wick(bars, _):
    open_, high, low, close = _parts(bars)
    value = safe_ratio(np.minimum(open_, close) - low, high - low, when_zero=0.0)
    return frame(bars.index, lower_wick_ratio=value)


def _body(bars, _):
    open_, high, low, close = _parts(bars)
    return frame(
        bars.index,
        body_ratio=safe_ratio(np.abs(close - open_), high - low, when_zero=0.0),
    )


def _dist_high(span: int):
    def compute(bars, _):
        _, high, _, close = _parts(bars)
        top = rolling(high, span, lambda w: w.max(axis=1))
        return frame(bars.index, **{f"dist_high_{span}": close / top - 1.0})

    return compute


def _dist_low(span: int):
    def compute(bars, _):
        _, _, low, close = _parts(bars)
        bottom = rolling(low, span, lambda w: w.min(axis=1))
        return frame(bars.index, **{f"dist_low_{span}": close / bottom - 1.0})

    return compute


def _spec(**fields) -> FeatureSpec:
    base = dict(
        version="1.0.0",
        group=FeatureGroup.PRICE_STRUCTURE,
        required_fields=OHLC,
        visibility=Visibility.PRICE_RELATIVE,
        lookback=0,
        units="fraction",
    )
    base.update(fields)
    return FeatureSpec(**base)


def specs() -> list[FeatureSpec]:
    out = [
        _spec(
            id="hl_range_pct",
            description="High-low range of the session as a fraction of the close.",
            definition="(H - L) / C",
            compute=_hl_range,
            hard_bounds=(0.0, None),
        ),
        _spec(
            id="clv",
            description="Close location value: -1 at the low, +1 at the high.",
            definition="((C - L) - (H - C)) / (H - L); 0 if H == L",
            compute=_clv,
            hard_bounds=(-1.0, 1.0),
        ),
        _spec(
            id="upper_wick_ratio",
            description="Upper shadow as a share of the session range.",
            definition="(H - max(O, C)) / (H - L); 0 if H == L",
            compute=_upper_wick,
            hard_bounds=(0.0, 1.0),
        ),
        _spec(
            id="lower_wick_ratio",
            description="Lower shadow as a share of the session range.",
            definition="(min(O, C) - L) / (H - L); 0 if H == L",
            compute=_lower_wick,
            hard_bounds=(0.0, 1.0),
        ),
        _spec(
            id="body_ratio",
            description="Candle body as a share of the session range.",
            definition="|C - O| / (H - L); 0 if H == L",
            compute=_body,
            hard_bounds=(0.0, 1.0),
            determined_by=("upper_wick_ratio", "lower_wick_ratio"),
            redundancy_reason="body = 1 - upper wick - lower wick whenever H > L",
        ),
    ]
    for span in RANGE_SPANS:
        out += [
            _spec(
                id=f"dist_high_{span}",
                description=f"Distance of the close below the {span}-session high.",
                definition=f"C[t] / max(H[t-{span - 1} .. t]) - 1",
                lookback=span - 1,
                compute=_dist_high(span),
                hard_bounds=(-1.0, 0.0),
            ),
            _spec(
                id=f"dist_low_{span}",
                description=f"Distance of the close above the {span}-session low.",
                definition=f"C[t] / min(L[t-{span - 1} .. t]) - 1",
                lookback=span - 1,
                compute=_dist_low(span),
                hard_bounds=(0.0, None),
            ),
        ]
    return out
