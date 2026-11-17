"""MLP learning sprint - Step 6: the loss (how wrong are the predictions?).

Context
-------
Step 5 gives, for each example, probabilities [P(down), P(up)]. To *learn*,
the network needs ONE number that says how bad its predictions are. That
number is the *loss*. Training (step 8) = repeatedly nudging the weights so
the loss goes down.

Cross-entropy loss
------------------
For one example, look only at the probability given to the CORRECT class:

    loss = -log(probability given to the correct answer)

Example where the correct answer is "up" (label y = 1):

    model says P(up) = 0.9   -> -log(0.9)  = 0.11   good, small loss
    model says P(up) = 0.5   -> -log(0.5)  = 0.69   unsure, medium loss
    model says P(up) = 0.1   -> -log(0.1)  = 2.30   confidently WRONG, big loss
    model says P(up) = 0.001 -> -log(0.001)= 6.91   very confidently wrong

Why -log? It is 0 for a perfect answer (P = 1) and grows without limit as
P -> 0, so confident mistakes are punished much harder than hesitant ones.
That pushes the model to be honest about its uncertainty, which matters
for calibration later in the project (F13).

For the whole dataset, the loss is the AVERAGE over all examples.

Picking "the probability of the correct class" for every example at once
------------------------------------------------------------------------
probs has one row per example and one column per class; y holds the correct
column of each row. NumPy "fancy indexing" picks one value per row:

    probs = [[0.2, 0.8],      y = [1, 0]
             [0.9, 0.1]]
    probs[[0, 1], [1, 0]] -> [probs[0, 1], probs[1, 0]] = [0.8, 0.9]

    np.arange(n) gives [0, 1, ..., n-1], i.e. every row number.

Reference values to remember (2 classes)
----------------------------------------
- Coin flip (always 50/50):                 loss = ln(2) = 0.693
- Always predicting the train up-rate (53.6% up), for every example:
                                            loss = 0.6905
  A useful model must go BELOW this on validation, not just below 0.693.

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step6_loss.py
"""

import sys
from pathlib import Path

import numpy as np

# Make steps 1-5 and the project's code importable (see step 2).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from step1_returns import CSV_PATH, load_returns  # noqa: E402
from step2_windows import make_xy  # noqa: E402
from step3_normalize import fit_normalizer, normalize  # noqa: E402
from step5_forward import init_mlp, mlp_forward  # noqa: E402

from src.data.split import build_examples, load_split_config  # noqa: E402

# log(0) is minus infinity, which would break all the arithmetic. Clipping
# probabilities to at least this tiny value keeps the loss finite.
EPS = 1e-12


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Average cross-entropy loss over all examples.

    probs : shape (n_examples, n_classes), each row sums to 1 (from softmax).
    y     : shape (n_examples,), the correct class of each example (0 or 1).
    """
    n = len(y)

    # 1. For every row, the probability given to the correct class.
    #    Hint: probs[row_numbers, correct_columns] (see "fancy indexing" above).
    p_correct = probs[np.arange(n), y]

    # Never take log(0): replace anything below EPS by EPS.
    p_correct = np.clip(p_correct, EPS, 1.0)

    # 2. The loss: the average of -log(p_correct).
    #    Hint: np.log(...) and np.mean(...). Don't forget the minus sign.
    return float(-np.mean(np.log(p_correct)))


if __name__ == "__main__":
    # --- Part A: the small example from the top of this file ---
    tiny_probs = np.array([[0.2, 0.8], [0.9, 0.1]])
    tiny_y = np.array([1, 0])
    print(f"tiny example loss: {cross_entropy(tiny_probs, tiny_y):.4f}  "
          "(= (-log 0.8 - log 0.9) / 2 = 0.1643)")
    perfect = cross_entropy(np.array([[0.0, 1.0]]), np.array([1]))
    awful = cross_entropy(np.array([[1.0, 0.0]]), np.array([1]))
    # abs() only hides the harmless "-0.0" that -log(1.0) produces.
    print(f"perfect prediction: {abs(perfect):.4f}    "
          f"100% sure and wrong: {awful:.1f} (finite thanks to EPS)\n")

    # --- Part B: the untrained network on real data ---
    returns = load_returns(CSV_PATH)
    config = load_split_config()
    examples = build_examples(list(returns.index.date), config)
    values = returns.to_numpy()
    X_train, y_train = make_xy(values, examples.train, config.lookback)
    mean, std = fit_normalizer(X_train)
    X_train_n = normalize(X_train, mean, std)

    params = init_mlp(n_in=64, n_hidden=32, n_out=2, seed=0)
    probs, _ = mlp_forward(X_train_n, params)
    loss = cross_entropy(probs, y_train)

    # The two reference "models" that don't even look at the inputs:
    coin_flip = np.full((len(y_train), 2), 0.5)
    up_rate = y_train.mean()
    constant = np.tile([1 - up_rate, up_rate], (len(y_train), 1))

    print(f"coin flip (50/50):                    {cross_entropy(coin_flip, y_train):.4f}")
    print(f"constant train up-rate ({up_rate:.1%} up):    "
          f"{cross_entropy(constant, y_train):.4f}")
    print(f"untrained network (random weights):   {loss:.4f}")

    # What to notice: the untrained network is WORSE than a coin flip.
    # Its random weights make it confident (e.g. 70% up) for no reason, and
    # cross-entropy punishes unjustified confidence. Step 8's training should
    # push the loss down to at least the constant model's 0.6905.

    # --- Self-check ---
    assert abs(cross_entropy(tiny_probs, tiny_y) - 0.164252033486018) < 1e-12, (
        "tiny example is wrong - pick the correct-class probability of each row, "
        "then average -log of it"
    )
    assert abs(perfect) < 1e-12, "a perfect prediction must have loss 0"
    assert np.isfinite(awful), "loss must stay finite - keep the np.clip line"
    assert abs(loss - 0.901978281522877) < 1e-9, (
        "loss on real data is wrong - check you average over ALL examples"
    )
    print("\nStep 6 OK")
