"""MLP learning sprint - Step 7: backpropagation (which way to nudge each weight).

Context
-------
The network has 2146 weights and biases. The loss (step 6) tells us how wrong
it is. To improve it, we need to know, for EVERY single parameter:

    "If I increase this parameter a tiny bit, does the loss go up or down,
     and how strongly?"

That number is the *gradient* of the loss with respect to the parameter
(the slope of the loss along that parameter):
    - gradient > 0 -> increasing the parameter increases the loss -> decrease it;
    - gradient < 0 -> increasing the parameter decreases the loss -> increase it;
    - large |gradient| -> the loss is very sensitive to it.
Step 8 then does:  parameter = parameter - learning_rate * gradient
("take a small step downhill"). This is *gradient descent*.

Backpropagation = computing all 2146 gradients efficiently
----------------------------------------------------------
The forward pass is a chain of simple operations:

    X -> [z1 = X@W1 + b1] -> [h1 = relu(z1)] -> [s = h1@W2 + b2]
      -> [p = softmax(s)] -> [loss = cross_entropy(p, y)]

The *chain rule* of calculus says: to know how the loss depends on something
early in the chain, multiply the local sensitivities of every link after it.
So we walk the chain BACKWARDS, from the loss to the inputs, reusing each
result for the next link. Notation: dA means "gradient of the loss w.r.t. A";
it always has exactly the SAME SHAPE as A.

    shapes: X (n,64)  W1 (64,32)  b1 (32,)  z1,h1 (n,32)
            W2 (32,2) b2 (2,)     s,p (n,2)

1) Scores (softmax + cross-entropy together give a famously simple result):
       ds = (p - y_onehot) / n                                  (n,2)
   y_onehot is the label written as a row: class 1 -> [0, 1], class 0 -> [1, 0].
   Intuition: say the model gave [P(down), P(up)] = [0.3, 0.7] and the answer
   was up ([0, 1]). Then p - y_onehot = [+0.3, -0.3]:
     - "up" score:   -0.3 -> negative gradient -> gradient descent RAISES it;
     - "down" score: +0.3 -> positive gradient -> gradient descent LOWERS it.
   The bigger the mistake, the bigger the push.
   The /n is there because the loss is an AVERAGE over n examples.

2) Second linear layer  s = h1 @ W2 + b2:
       dW2 = h1.T @ ds       (32,n) @ (n,2) -> (32,2)   same shape as W2
       db2 = ds summed over the examples (axis=0)  -> (2,)  same shape as b2
       dh1 = ds @ W2.T       (n,2) @ (2,32) -> (n,32)   same shape as h1
   Why? A weight W2[j,k] multiplies h1[:, j] to build s[:, k], so its effect
   on the loss is h1[:, j] times ds[:, k], added up over all examples: that is
   exactly what the matrix product h1.T @ ds computes. A bias is added to
   every example, so its gradient is the sum of ds over the examples. And h1
   reached the scores through W2, so its gradient goes back through W2.T.

3) ReLU  h1 = relu(z1):
       dz1 = dh1 * (z1 > 0)                                     (n,32)
   Where z1 was positive, ReLU let the value through unchanged (slope 1);
   where it was negative, ReLU output a constant 0 (slope 0), so no gradient
   flows back. (z1 > 0) is a True/False array that NumPy treats as 1/0.

4) First linear layer  z1 = X @ W1 + b1: same rules as in 2):
       dW1 = X.T @ dz1        (64,n) @ (n,32) -> (64,32)
       db1 = dz1 summed over the examples (axis=0)  -> (32,)
   (We don't need dX: the inputs are data, not something we can change.)

Gradient check: proving the formulas are right
----------------------------------------------
A bug in backprop is silent: training would still "run", just badly. So we
verify with the definition of a slope, numerically, on a few parameters:

    slope ~ (loss(w + h) - loss(w - h)) / (2h)      with a tiny h, e.g. 1e-5

This is far too slow to train with (2 forward passes PER parameter), but
perfect for checking. If the two methods agree to ~6 digits, backprop is right.

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step7_backprop.py
"""

import sys
from pathlib import Path

import numpy as np

# Make steps 1-6 and the project's code importable (see step 2).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from step1_returns import CSV_PATH, load_returns  # noqa: E402
from step2_windows import make_xy  # noqa: E402
from step3_normalize import fit_normalizer, normalize  # noqa: E402
from step5_forward import init_mlp, mlp_forward  # noqa: E402
from step6_loss import cross_entropy  # noqa: E402

from src.data.split import build_examples, load_split_config  # noqa: E402


def mlp_backward(cache: dict, y: np.ndarray, params: dict) -> dict:
    """Gradients of the cross-entropy loss w.r.t. every parameter.

    cache  : intermediate values saved by mlp_forward (X, z1, h1, probs).
    y      : correct classes, shape (n,).
    params : current weights (needed to send the gradient back through W2).
    Returns a dict with the same keys and shapes as params.
    """
    X, z1, h1, probs = cache["X"], cache["z1"], cache["h1"], cache["probs"]
    n = len(y)

    # 1) Gradient of the scores (given: see explanation above).
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), y] = 1.0
    ds = (probs - y_onehot) / n

    # 2) Second linear layer. Check each result has the shape of its parameter.
    dW2 = h1.T @ ds
    db2 = ds.sum(axis=0)
    dh1 = ds @ params["W2"].T

    # 3) Through ReLU: keep the gradient only where z1 was positive.
    dz1 = dh1 * (z1 > 0)

    # 4) First linear layer (same pattern as 2), with X and dz1).
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def numerical_gradient(params: dict, X: np.ndarray, y: np.ndarray,
                       name: str, index: tuple, h: float = 1e-5) -> float:
    """Slope of the loss along ONE parameter entry, by finite differences.

    Temporarily moves params[name][index] by +h and -h, measures the loss
    each time, then puts the original value back.
    """
    original = params[name][index]

    params[name][index] = original + h
    loss_plus = cross_entropy(mlp_forward(X, params)[0], y)

    params[name][index] = original - h
    loss_minus = cross_entropy(mlp_forward(X, params)[0], y)

    params[name][index] = original  # restore: the check must not change the model
    return (loss_plus - loss_minus) / (2 * h)


if __name__ == "__main__":
    returns = load_returns(CSV_PATH)
    config = load_split_config()
    examples = build_examples(list(returns.index.date), config)
    values = returns.to_numpy()
    X_train, y_train = make_xy(values, examples.train, config.lookback)
    mean, std = fit_normalizer(X_train)
    X_train_n = normalize(X_train, mean, std)

    params = init_mlp(n_in=64, n_hidden=32, n_out=2, seed=0)
    probs, cache = mlp_forward(X_train_n, params)
    grads = mlp_backward(cache, y_train, params)

    print("Gradient shapes (must equal the parameter shapes):")
    for name in params:
        print(f"  d{name:<3} {str(grads[name].shape):<9} vs {name:<3} {params[name].shape}")

    # --- Gradient check on 5 random entries of each parameter ---
    # Relative error = |a - b| / (|a| + |b|): ~1e-7 or less means the two
    # methods agree; ~1e-2 or more means a bug in mlp_backward.
    print("\nGradient check (backprop vs numerical slope):")
    rng = np.random.default_rng(seed=1)
    worst = 0.0
    for name, p in params.items():
        for _ in range(5):
            index = tuple(int(rng.integers(dim)) for dim in p.shape)
            analytic = grads[name][index]
            numeric = numerical_gradient(params, X_train_n, y_train, name, index)
            rel_error = abs(analytic - numeric) / max(abs(analytic) + abs(numeric), 1e-12)
            worst = max(worst, rel_error)
            print(f"  {name}{list(index)!s:<9} backprop {analytic:+.8f}  "
                  f"numerical {numeric:+.8f}  rel. error {rel_error:.1e}")
    print(f"worst relative error: {worst:.1e}")

    # --- One step downhill: the whole point of gradients ---
    # Move every parameter a little in the direction that lowers the loss.
    # (Step 8 will repeat this thousands of times; 0.1 is the "learning rate".)
    loss_before = cross_entropy(probs, y_train)
    stepped = {name: params[name] - 0.1 * grads[name] for name in params}
    loss_after = cross_entropy(mlp_forward(X_train_n, stepped)[0], y_train)
    print(f"\nloss before one step: {loss_before:.4f}   after: {loss_after:.4f}")

    # --- Self-check ---
    for name in params:
        assert grads[name].shape == params[name].shape, (
            f"d{name} has shape {grads[name].shape}, expected {params[name].shape}"
        )
    assert worst < 1e-5, (
        "backprop and numerical gradients disagree - one of the formulas is wrong; "
        "look at which parameter has the large errors"
    )
    assert loss_after < loss_before, "one step downhill should reduce the loss"
    print("\nStep 7 OK")
