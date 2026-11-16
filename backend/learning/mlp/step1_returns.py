"""MLP learning sprint - Step 1: load SPY prices and compute daily returns.

Context
-------
We are building a small neural network (an MLP) that looks at recent market
history and predicts whether the price will go up or down the next day.
This first step prepares the raw material: a clean list of daily returns.

Why returns and not prices?
    A price of $44 (1993) and $757 (2026) are not comparable: a model fed raw
    prices would mostly learn "which era is this". A *return* is the percentage
    change from one day to the next, so it means the same thing in any period:

        return_t = price_t / price_{t-1} - 1

    Example: price goes 100 -> 102, return = 102 / 100 - 1 = 0.02 (= +2%).

Input:  backend/data/raw/yfinance/SPY.csv (downloaded by the F1 data pipeline,
        git-ignored, never committed).
Output: a pandas Series of daily returns, one per trading day, indexed by date.

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step1_returns.py
It prints a few statistics and ends with "Step 1 OK" if everything is correct.
"""

from pathlib import Path

import pandas as pd

# Location of the SPY data file, built relative to this script so the script
# works no matter which folder it is launched from.
# parents[2] climbs three folders up: mlp/ -> learning/ -> backend/
CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "yfinance" / "SPY.csv"


def load_returns(csv_path: Path) -> pd.Series:
    """Read a price CSV and return its daily returns, indexed by date.

    The first day is dropped because it has no previous day to compare with.
    """
    # 1. Read the CSV into a table (DataFrame).
    #    The file's columns are: date, open, high, low, close, volume, adjusted_close.
    #    - parse_dates=["date"]: convert the text "1993-01-29" into a real date,
    #      so pandas can sort, compare and filter by date.
    #    - index_col="date": use the date as the row label instead of 0, 1, 2...
    #      Every row is then identified by its trading day.
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    # 2. Choose which price to compute returns from.
    #    The file holds two kinds of price, which must never be mixed:
    #    - "close": the raw traded price. Reserved for the replay engine
    #      (order fills and accounting).
    #    - "adjusted_close": corrected for dividends and splits. It reflects
    #      what an investor actually earned, so it is the one used for
    #      training and research (rule stated in the project README).
    prices = df["adjusted_close"]

    # 3. Compute the daily return: price_t / price_{t-1} - 1.
    #    prices.shift(1) moves every value one row down, so on each row it
    #    holds *yesterday's* price:
    #
    #        date        prices   prices.shift(1)   return
    #        1993-01-29  24.05    NaN               NaN      <- no yesterday
    #        1993-02-01  24.22    24.05             +0.71%
    #        1993-02-02  24.28    24.22             +0.21%
    #
    #    Dividing the two columns row by row gives every return at once.
    returns = prices / prices.shift(1) - 1

    # 4. The first day has no "yesterday", so its return is NaN (missing).
    #    dropna() removes missing values; the model must never see a NaN.
    returns = returns.dropna()

    return returns


# This block only runs when the file is executed directly
# (not when another script imports load_returns from it).
if __name__ == "__main__":
    returns = load_returns(CSV_PATH)

    # Quick look at the data. With SPY you should see roughly:
    # - 8506 returns (one per trading day since 1993, minus the first day);
    # - a mean daily return of ~0.0005 (+0.05% on an average day);
    # - a daily standard deviation of ~0.012 (a typical day moves about +/-1.2%,
    #   much larger than the mean: daily moves are mostly noise);
    # - ~54% of up days. A "model" that always answers "up" is therefore right
    #   54% of the time: this is the bar our MLP will have to beat (step 9).
    print("First 3 returns:\n", returns.head(3), "\n")
    print("Number of returns:", len(returns))
    print(f"Mean daily return: {returns.mean():.5f}")
    print(f"Daily std:         {returns.std():.5f}")
    print(f"Share of up days:  {(returns > 0).mean():.3f}")

    # --- Self-check ---
    # Each assert stops the script with the given message if its condition
    # is false. Reaching the final print means step 1 is correct.
    assert isinstance(returns, pd.Series), "returns should be a pandas Series"
    assert isinstance(returns.index, pd.DatetimeIndex), "index should be the dates"
    assert returns.isna().sum() == 0, "there are still NaN values"
    assert len(returns) == 8506, f"expected 8506 returns, got {len(returns)}"
    # The first return must equal the first two adjusted_close values of the
    # CSV (1993-01-29 and 1993-02-01) plugged into the formula by hand.
    # 1e-9 tolerates tiny floating-point rounding differences.
    assert abs(returns.iloc[0] - (24.22462272644043 / 24.05353546142578 - 1)) < 1e-9, (
        "first return is wrong - check the column and the formula"
    )
    print("\nStep 1 OK")
