"""MLP learning sprint - Step 5: ReLU, softmax and the full forward pass.

Context
-------
Step 4 built one linear layer. This step assembles the complete network:

    X (64 returns)
      -> Linear 64 -> 32      (step 4)
      -> ReLU                 (new: lets the network learn curved patterns)
      -> Linear 32 -> 2       (step 4 again: one score per class)
      -> softmax              (new: turns the 2 scores into 2 probabilities)
      -> [P(down), P(up)]

The 32 neurons in the middle are the *hidden layer*: we never look at their
values directly, only at the final probabilities.

Why ReLU?  (ReLU = "Rectified Linear Unit")
--------------------------------------------
    relu(z) = max(0, z)      -> negatives become 0, positives pass unchanged

    relu([-2.0, 0.5, 3.0]) = [0.0, 0.5, 3.0]

Without it, stacking linear layers is pointless: Linear(Linear(X)) is still
just ONE linear formula (a weighted sum), however many layers you stack. A
straight-line formula can only learn "the more X, the more Y". ReLU bends the
line, so the network can learn rules like "only when the last days were very
negative...". This bend is what makes it a real neural network.

Why softmax?
------------
The last layer outputs 2 raw *scores* (called "logits"), any real numbers,
e.g. [0.3, 1.5]. We want probabilities: between 0 and 1, summing to 1.

    softmax(z)_i = exp(z_i) / (exp(z_1) + exp(z_2) + ...)

    scores [0.3, 1.5] -> exp -> [1.35, 4.48] -> divide by the sum 5.83
                      -> [0.23, 0.77]  = 23% down, 77% up

exp() makes every value positive, and dividing by the total makes them sum
to 1. A higher score always means a higher probability.

Numerical safety trick: exp(1000) overflows to infinity in a computer. But
softmax gives the same result if we subtract the same number from every score
first, so we subtract each row's maximum: the largest score becomes 0 and
exp() never overflows.
    [1000, 1001] -> [-1, 0] -> same probabilities [0.27, 0.73], no overflow.

Why 2 outputs and not 1?
------------------------
Class 0 = "next day not up", class 1 = "next day up" (the labels of step 2).
One output per class generalizes directly to the 3 classes long/flat/short
that the project's MarketGPT will use later.

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step5_forward.py
"""

import sys
from pathlib import Path

import numpy as np

# Make steps 1-4 and the project's code importable (see step 2).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from step1_returns import CSV_PATH, load_returns  # noqa: E402
from step2_windows import make_xy  # noqa: E402
from step3_normalize import fit_normalizer, normalize  # noqa: E402
from step4_linear import init_linear, linear_forward  # noqa: E402
from src.data.split import build_examples, load_split_config  # noqa: E402


def relu(z: np.ndarray) -> np.ndarray:
    """Replace every negative value by 0; keep positive values unchanged."""
    # 1. Hint: np.maximum(a, b) compares element by element and keeps the
    #    larger one. Compare with 0.0.
    return np.maximum(z, 0.0)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Turn each row of scores into probabilities (positive, summing to 1).

    scores: shape (n_examples, n_classes). Each ROW is one example.
    """
    # Subtract each row's maximum first, so exp() can never overflow.
    # axis=1 means "along the row"; keepdims=True keeps the result as a
    # column of shape (n_examples, 1) so it can be subtracted from every
    # value of its own row.
    shifted = scores - scores.max(axis=1, keepdims=True)
    exps = np.exp(shifted)

    # 2. Divide each value by the total of its row.
    #    Hint: like the line above, but with .sum instead of .max.
    return exps / exps.sum(axis=1, keepdims=True)


def init_mlp(n_in: int, n_hidden: int, n_out: int, seed: int) -> dict:
    """Create all the learnable parameters of the network, reproducibly."""
    rng = np.random.default_rng(seed)
    W1, b1 = init_linear(n_in, n_hidden, rng)   # 64 -> 32
    W2, b2 = init_linear(n_hidden, n_out, rng)  # 32 -> 2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def mlp_forward(X: np.ndarray, params: dict):
    """Full forward pass: inputs -> probabilities.

    Returns (probs, cache):
    - probs : shape (n_examples, 2), [P(down), P(up)] for every example;
    - cache : the intermediate values. They are not needed to predict, but
              step 7 (learning) needs them to work out how to fix the weights,
              so we keep them instead of recomputing them.
    """
    # 3. First layer: linear 64 -> 32, using W1 and b1.
    z1 = linear_forward(X, params["W1"], params["b1"])

    # 4. Apply ReLU to z1.
    h1 = relu(z1)

    # 5. Second layer: linear 32 -> 2, using h1 as input, W2 and b2.
    scores = linear_forward(h1, params["W2"], params["b2"])

    # 6. Turn the scores into probabilities.
    probs = softmax(scores)

    cache = {"X": X, "z1": z1, "h1": h1, "probs": probs}
    return probs, cache


if __name__ == "__main__":
    # --- Part A: small hand-checkable examples ---
    print("relu([-2, 0.5, 3])        =", relu(np.array([-2.0, 0.5, 3.0])))
    print("softmax([[0.3, 1.5]])     =", softmax(np.array([[0.3, 1.5]])).round(2))
    print("softmax([[1000, 1001]])   =", softmax(np.array([[1000.0, 1001.0]])).round(2),
          "(no overflow)\n")

    # --- Part B: the full network on real (normalized) data ---
    returns = load_returns(CSV_PATH)
    config = load_split_config()
    examples = build_examples(list(returns.index.date), config)
    values = returns.to_numpy()
    X_train, y_train = make_xy(values, examples.train, config.lookback)
    mean, std = fit_normalizer(X_train)
    X_train_n = normalize(X_train, mean, std)

    params = init_mlp(n_in=64, n_hidden=32, n_out=2, seed=0)
    probs, cache = mlp_forward(X_train_n, params)

    n_params = sum(p.size for p in params.values())
    print(f"learnable parameters: {n_params}  (64*32 + 32 + 32*2 + 2)")
    print(f"probs shape: {probs.shape}  (examples, [P(down), P(up)])")
    print("first 3 examples:\n", probs[:3].round(3))
    print(f"share of hidden values set to 0 by ReLU: {(cache['h1'] == 0).mean():.2f}")

    # The network is untrained: its weights are random, so these
    # predictions are meaningless. Accuracy should be close to a coin flip
    # (or to always guessing one class). Training starts in step 7.
    predicted_class = probs.argmax(axis=1)   # index of the larger probability
    accuracy = (predicted_class == y_train).mean()
    print(f"untrained accuracy on train: {accuracy:.3f}  (meaningless for now)")

    # --- Self-check ---
    assert np.array_equal(relu(np.array([-2.0, 0.5, 3.0])), [0.0, 0.5, 3.0]), (
        "relu is wrong - negatives must become 0, positives unchanged"
    )
    p = softmax(np.array([[0.3, 1.5], [1000.0, 1001.0]]))
    assert np.all(np.isfinite(p)), "softmax overflowed - keep the max-shift line"
    assert np.allclose(p.sum(axis=1), 1.0), (
        "each row of softmax must sum to 1 - divide by the sum of ITS row "
        "(axis=1, keepdims=True)"
    )
    assert np.allclose(p[0], [0.2315, 0.7685], atol=1e-4), "softmax values are wrong"
    assert cache["z1"].shape == (5961, 32), "z1 should be (5961, 32): first layer"
    assert np.all(cache["h1"] >= 0), "h1 must be relu(z1): no negative values"
    assert probs.shape == (5961, 2), f"probs should be (5961, 2), got {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0), "every row of probs must sum to 1"
    print("\nStep 5 OK")
