"""MLP learning sprint - Step 2: turn daily returns into training examples.

Context
-------
Step 1 gave us one long list of daily returns. A neural network cannot learn
from "one long list": it needs many separate *examples*, each made of
    - an input X: what the model is allowed to see, and
    - a label y: the right answer we want it to predict.

Our example design (fixed in backend/config/split.json):
    - lookback = 64      -> X is the last 64 daily returns, up to and
                            including the "decision day" t;
    - label_horizon = 1  -> y says whether the NEXT day (t + 1) goes up.

Picture for one example (r[t] means the return on day t):

    PAST + PRESENT (visible to the model)  |  FUTURE (hidden from the model)
    r[t-63], r[t-62], ..., r[t-1], r[t]    |  r[t+1]
    <--------- X: 64 returns --------->    |  next day's return
                                  today    |
                                  day t    |  no-lookahead boundary

At the end of day t, the model sees X and predicts whether tomorrow goes up.
Tomorrow's return is NOT part of X. We use it only to set the answer:
    y = 1 if r[t+1] > 0; otherwise y = 0.

The golden rule (no lookahead): X must contain ONLY days <= t. If even one
day after t slipped into X, the model would be "reading tomorrow's newspaper"
and its results would be fake.

Train / validation / test
-------------------------
Examples are split by date, never randomly:
    - train      (up to 2016-12-31): the model learns from these;
    - validation (2017-04 -> 2019-12): we check progress and tune settings;
    - test       (from 2020-02): used ONCE, at the very end of the project.
Gaps between the sets stop one set's labels from leaking into another.
The project's function `build_examples` (backend/src/data/split.py) already
chooses which decision days are valid for each set: we reuse it, never
re-implement it.

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step2_windows.py
"""

import sys
from pathlib import Path

import numpy as np

# Make the project's code importable from this learning script:
# - this folder (backend/learning/mlp) -> to import step 1;
# - backend/                           -> to import src.data.split.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from step1_returns import CSV_PATH, load_returns  # noqa: E402
from src.data.split import build_examples, load_split_config  # noqa: E402


def make_xy(values: np.ndarray, decision_days: tuple[int, ...], lookback: int):
    """Build the inputs X and labels y for a list of decision days.

    values        : all daily returns as a plain NumPy array (position 0, 1, 2...).
    decision_days : positions t chosen by build_examples for one set.
    lookback      : how many past returns go into X (64).

    Returns X with shape (number_of_examples, lookback) and y with shape
    (number_of_examples,), where y is 1 for "next day up" and 0 otherwise.
    """
    windows = []
    labels = []
    for t in decision_days:
        # 1. The input window: the `lookback` returns ending AT day t (included).
        #    Python slicing values[a:b] takes positions a, a+1, ..., b-1
        #    (b is excluded!). Choose a and b so you get exactly 64 values
        #    whose last one is values[t].
        window = values[t - lookback + 1 : t + 1]
        
        # 2. The label: 1 if the return of the day AFTER t is positive, else 0.
        #    A return of exactly 0 counts as "not up" (0).
        label = 1 if values[t + 1] > 0 else 0
        
        windows.append(window)
        labels.append(label)

    # np.stack turns a list of 64-value arrays into one 2-D table:
    # one row per example, one column per past day.
    return np.stack(windows), np.array(labels)


if __name__ == "__main__":
    returns = load_returns(CSV_PATH)
    config = load_split_config()

    # build_examples needs the date of every return (to know which set each
    # day belongs to) and gives back, for each set, the valid decision days t.
    dates = list(returns.index.date)
    examples = build_examples(dates, config)

    # From here on we only need the numbers, so drop the dates.
    values = returns.to_numpy()

    X_train, y_train = make_xy(values, examples.train, config.lookback)
    X_val, y_val = make_xy(values, examples.validation, config.lookback)
    X_test, y_test = make_xy(values, examples.test, config.lookback)

    for name, X, y, days in (
        ("train", X_train, y_train, examples.train),
        ("validation", X_val, y_val, examples.validation),
        ("test", X_test, y_test, examples.test),
    ):
        first, last = returns.index[days[0]].date(), returns.index[days[-1]].date()
        print(f"{name:<10} X {X.shape}  share of 'up' labels {y.mean():.3f}  "
              f"decision days {first} -> {last}")

    # --- Self-check ---
    assert X_train.shape == (5961, 64), f"train X shape is {X_train.shape}"
    assert X_val.shape == (628, 64), f"validation X shape is {X_val.shape}"
    assert X_test.shape == (1705, 64), f"test X shape is {X_test.shape}"
    assert set(np.unique(y_train)) <= {0, 1}, "labels must be 0 or 1"

    # The window of every example must END on its decision day t...
    for X, days in ((X_train, examples.train), (X_val, examples.validation)):
        for row, t in zip(X, days):
            assert row[-1] == values[t], (
                f"window for day {t} does not end on day t: wrong slice bounds"
            )
            assert row[0] == values[t - 63], (
                f"window for day {t} does not start 63 days before t"
            )
    # ...and its label must come from day t + 1 (the future), never from X.
    t = examples.train[0]
    assert y_train[0] == int(values[t + 1] > 0), "label must use the return of day t + 1"
    print("\nStep 2 OK")
