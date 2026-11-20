from datetime import date

import numpy as np
import pandas as pd
import pytest

from learning.mlp.order_replay import (
    CHOICES,
    Choice,
    Costs,
    _choose,
    run,
    simulate,
)
from src.data.split import SplitConfig


@pytest.mark.parametrize(("target", "sign"), [("long", 1), ("short", -1)])
def test_market_entry_uses_next_open_and_third_close_with_costs(target, sign):
    bars = np.array([[101, 105, 99, 104], [104, 106, 103, 105], [105, 108, 104, 107]])
    result = simulate(Choice(target, "market"), 100, bars, Costs(0.01, 20, 0.25))
    assert result.fill_bar == 1
    expected = sign * (107 - 101) - (107 + 101) * 0.001 - 0.50
    assert result.net_pnl == pytest.approx(expected)


@pytest.mark.parametrize(
    ("target", "column", "touch", "trade_through", "close", "expected"),
    [
        ("long", 2, 99.97, 99.96, 100.25, 0.26),
        ("short", 1, 100.03, 100.04, 99.75, 0.26),
    ],
)
def test_limit_touch_expires_and_third_bar_penetration_fills_at_limit(
    target, column, touch, trade_through, close, expected
):
    bars = np.full((3, 4), 100.0)
    bars[:, column] = touch
    choice = Choice(target, "limit", 3)
    costs = Costs(0.01, 0, 0.01)
    expired = simulate(choice, 100, bars, costs)
    assert expired.limit_price == touch
    assert expired.fill_bar is None
    assert expired.net_pnl == 0

    bars[-1, column] = trade_through
    bars[-1, 3] = close
    if target == "long":
        bars[-1, 1] = close
    else:
        bars[-1, 2] = close
    filled = simulate(choice, 100, bars, costs)
    assert filled.fill_bar == 3
    assert filled.net_pnl == pytest.approx(expected)
    assert simulate(choice, 100, bars, Costs(0.01, 20)).fill_bar is None


def test_policy_can_choose_short_limit_and_fixed_policy_cannot_change_offset():
    scores = np.full((2, len(CHOICES)), -10.0)
    scores[:, 0] = 100
    scores[0, 8] = 5
    scores[0, 7] = 2
    flexible = _choose(scores, fixed_offset=False)
    fixed = _choose(scores, fixed_offset=True)
    assert [CHOICES[i].name for i in flexible] == ["short_limit_5", "flat"]
    assert [CHOICES[i].name for i in fixed] == ["short_limit_3", "flat"]


def test_train_and_compare_on_separate_nonoverlapping_holdout():
    split = SplitConfig(
        train_end=date(2021, 3, 31),
        validation_start=date(2021, 5, 1),
        validation_end=date(2021, 7, 31),
        test_start=date(2021, 9, 1),
        test_end=date(2021, 11, 30),
        lookback=4,
        label_horizon=3,
    )
    dates = pd.bdate_range("2021-01-01", "2021-12-01")
    close = 100 - 0.1 * np.arange(len(dates))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.05,
            "high": close + 0.06,
            "low": close - 0.01,
            "close": close,
            "adjusted_close": close,
        }
    )
    frame.loc[dates.date > split.test_end, "open"] = np.nan
    report = run(frame, split, epochs=20, seed=2)
    holdout = report["windows"]["test"]
    assert holdout == (len(pd.bdate_range(split.test_start, split.test_end)) - 1) // 3
    assert report["mlp"]["choices"]["short_market"] == holdout
    assert report["mlp"]["total_net_pnl"] > 0
    assert report["fixed_offset_baseline"]["choices"]["flat"] == holdout
    assert report["fixed_offset_baseline"]["total_net_pnl"] == 0
