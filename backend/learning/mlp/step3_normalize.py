"""MLP learning sprint - Step 3: normalize the inputs (using training data only).

Context
-------
Our inputs X are daily returns: tiny numbers around 0.0004 +/- 0.012.
Neural networks learn much better when inputs are roughly centered on 0 with
a spread of about 1. Too-small inputs give tiny gradients, so learning crawls.

The fix is called *standardization* (or z-score):

    X_normalized = (X - mean) / std

    - subtracting the mean   -> centers the values around 0;
    - dividing by std        -> scales them so a "typical" move is about +/-1.

Example with mean = 0.0004 and std = 0.012:
    a +1.24% day (0.0124)  -> (0.0124 - 0.0004) / 0.012 = +1.0
    a -2.36% day (-0.0236) -> (-0.0236 - 0.0004) / 0.012 = -2.0

The trap: where do mean and std come from?
------------------------------------------
They must be computed on the TRAINING set ONLY, then the very same two numbers
are applied to validation and test.

Why? Computing them on all the data would let information from 2017-2026
(validation/test) leak into how training data is prepared: a subtle form of
lookahead. In real life, on the day you train the model, you do not know
future market statistics yet. Rule from the project README: "train-only
fitting".

We use ONE mean and ONE std for all 64 columns, because every column holds
the same kind of number (a daily return of the same asset).

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step3_normalize.py
"""

import sys
from pathlib import Path

import numpy as np

# Make step 1, step 2 and the project's code importable (see step 2).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from step1_returns import CSV_PATH, load_returns  # noqa: E402
from step2_windows import make_xy  # noqa: E402
from src.data.split import build_examples, load_split_config  # noqa: E402


def fit_normalizer(X_train: np.ndarray) -> tuple[float, float]:
    """Learn the normalization statistics from the TRAINING inputs only.

    Returns (mean, std): one number each, computed over every value of X_train
    (all examples, all 64 columns together).
    """
    # 1. The average of all values in X_train.
    #    Hint: NumPy arrays have a .mean() method; with no argument it averages
    #    every value of the array.
    mean = X_train.mean()

    # 2. The standard deviation (spread) of all values in X_train.
    #    Hint: same idea, with the .std() method.
    std = X_train.std()

    return float(mean), float(std)


def normalize(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply the standardization formula with ALREADY-LEARNED statistics.

    Used identically on train, validation and test: never re-compute mean/std
    here, always use the ones returned by fit_normalizer(X_train).
    """
    # 3. Apply (X - mean) / std. NumPy applies it to every value at once.
    return (X - mean) / std


if __name__ == "__main__":
    # Rebuild the step 2 datasets (same code as step 2).
    returns = load_returns(CSV_PATH)
    config = load_split_config()
    examples = build_examples(list(returns.index.date), config)
    values = returns.to_numpy()

    X_train, y_train = make_xy(values, examples.train, config.lookback)
    X_val, y_val = make_xy(values, examples.validation, config.lookback)

    # Learn the statistics on train ONLY...
    mean, std = fit_normalizer(X_train)
    print(f"Learned on train: mean = {mean:.6f}, std = {std:.6f}\n")

    # ...then apply the SAME statistics to every set.
    # The test set is deliberately not touched here: it stays locked until
    # the final evaluation of the project.
    X_train_n = normalize(X_train, mean, std)
    X_val_n = normalize(X_val, mean, std)

    print(f"{'':<12}{'before: mean':>14}{'std':>10}   {'after: mean':>12}{'std':>8}")
    for name, before, after in (
        ("train", X_train, X_train_n),
        ("validation", X_val, X_val_n),
    ):
        print(f"{name:<12}{before.mean():>14.5f}{before.std():>10.5f}   "
              f"{after.mean():>12.3f}{after.std():>8.3f}")

    # What to notice:
    # - train becomes exactly mean 0 / std 1 (the statistics come from it);
    # - validation is close but NOT exactly 0 / 1. That is expected and
    #   correct: it was scaled with train's statistics, and 2017-2019 was a
    #   calmer market than 1993-2016 (smaller daily moves -> std below 1).
    #   If validation came out exactly 0 / 1 too, it would mean its own
    #   statistics were used: that would be the leakage we want to avoid.

    # --- Self-check ---
    assert abs(mean - 0.000410571551532188) < 1e-12, (
        "mean is wrong - compute it on X_train only, over all values"
    )
    assert abs(std - 0.011816303187532757) < 1e-12, (
        "std is wrong - compute it on X_train only, over all values"
    )
    assert X_train_n.shape == X_train.shape, "normalizing must not change the shape"
    assert abs(X_train_n.mean()) < 1e-9 and abs(X_train_n.std() - 1) < 1e-9, (
        "normalized train should have mean 0 and std 1 - check the formula"
    )
    assert abs(X_val_n.std() - 0.729687433514451) < 1e-9, (
        "validation must be scaled with TRAIN statistics, not its own"
    )
    print("\nStep 3 OK")
