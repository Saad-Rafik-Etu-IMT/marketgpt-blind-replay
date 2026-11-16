"""MLP learning sprint - Step 4: one linear layer (forward pass only).

Context
-------
An MLP is a stack of *layers*. The basic building block is the LINEAR layer
(also called "fully connected" or "dense"). This step builds one, and only
its *forward pass*: computing an output from an input. Learning (changing the
weights) comes in step 7.

One neuron
----------
A neuron takes all the inputs, multiplies each one by its own *weight*,
adds them up, and adds a *bias*:

    output = x1*w1 + x2*w2 + ... + x64*w64 + b

- The weights say how much each input matters (positive, negative or ~0).
- The bias shifts the result up or down, whatever the inputs are.
At first they are random; training will adjust them to make good predictions.

A layer = many neurons side by side
-----------------------------------
A layer with 32 neurons computes 32 such sums from the same 64 inputs.
Instead of writing 32 loops, we store all weights in one matrix and use a
single matrix multiplication (the `@` operator in NumPy):

    X  : (n_examples, n_in)    - one row per example (e.g. 5961 x 64)
    W  : (n_in, n_out)         - one COLUMN per neuron (e.g. 64 x 32)
    b  : (n_out,)              - one bias per neuron (e.g. 32)
    out = X @ W + b : (n_examples, n_out)   (e.g. 5961 x 32)

Shape rule for `@`: (a, k) @ (k, c) -> (a, c). The inner sizes (k) must match:
each example's 64 inputs meet each neuron's 64 weights.

Tiny worked example (2 inputs, 3 neurons, 1 example):
    X = [[1, 2]]
    W = [[1, 0, -1],        neuron 1 weights = column 1 = (1, 2)
         [2, 1,  0]]        neuron 2 weights = column 2 = (0, 1)
    b = [0.5, 0, 0]         neuron 3 weights = column 3 = (-1, 0)

    neuron 1: 1*1 + 2*2  + 0.5 = 5.5
    neuron 2: 1*0 + 2*1  + 0   = 2.0
    neuron 3: 1*-1 + 2*0 + 0   = -1.0
    X @ W + b = [[5.5, 2.0, -1.0]]

Starting values (initialization)
--------------------------------
- Weights: small random numbers. If they were all equal, every neuron would
  compute the same thing and learn the same thing - useless. We use "He
  initialization": a normal distribution with std = sqrt(2 / n_in). This keeps
  outputs at a sensible size however many inputs there are, and suits the
  ReLU activation we add in step 5.
- Biases: start at 0. The random weights already make the neurons different.
- A fixed random *seed* makes the "random" numbers identical on every run,
  so results are reproducible (a project requirement).

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step4_linear.py
"""

import sys
from pathlib import Path

import numpy as np

# Make steps 1-3 and the project's code importable (see step 2).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from step1_returns import CSV_PATH, load_returns  # noqa: E402
from step2_windows import make_xy  # noqa: E402
from step3_normalize import fit_normalizer, normalize  # noqa: E402
from src.data.split import build_examples, load_split_config  # noqa: E402


def init_linear(n_in: int, n_out: int, rng: np.random.Generator):
    """Create the starting weights W and biases b of a linear layer.

    n_in  : number of inputs each neuron receives (e.g. 64).
    n_out : number of neurons in the layer (e.g. 32).
    rng   : NumPy random generator, created once with a fixed seed.
    """
    # 1. Weight matrix, filled with random numbers (He initialization).
    #    It must have one ROW per input and one COLUMN per neuron.
    #    Fill the shape: size=(rows, columns).
    W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))

    # 2. Bias vector: one zero per neuron.
    #    Hint: np.zeros(k) creates an array of k zeros.
    b = np.zeros(n_out)

    return W, b


def linear_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Forward pass of a linear layer: every neuron, every example, at once."""
    # 3. Apply the formula out = X @ W + b.
    return X @ W + b


if __name__ == "__main__":
    # --- Part A: the tiny worked example from the top of this file ---
    X_tiny = np.array([[1.0, 2.0]])
    W_tiny = np.array([[1.0, 0.0, -1.0],
                       [2.0, 1.0, 0.0]])
    b_tiny = np.array([0.5, 0.0, 0.0])
    out_tiny = linear_forward(X_tiny, W_tiny, b_tiny)
    print("Tiny example output:", out_tiny, "(expected [[ 5.5  2.  -1. ]])\n")

    # --- Part B: a real layer on real (normalized) data ---
    returns = load_returns(CSV_PATH)
    config = load_split_config()
    examples = build_examples(list(returns.index.date), config)
    values = returns.to_numpy()
    X_train, y_train = make_xy(values, examples.train, config.lookback)
    mean, std = fit_normalizer(X_train)
    X_train_n = normalize(X_train, mean, std)

    rng = np.random.default_rng(seed=0)
    W1, b1 = init_linear(n_in=64, n_out=32, rng=rng)
    hidden = linear_forward(X_train_n, W1, b1)

    print(f"X_train_n shape: {X_train_n.shape}  (examples, inputs)")
    print(f"W1 shape:        {W1.shape}  (inputs, neurons)")
    print(f"b1 shape:        {b1.shape}  (neurons,)")
    print(f"output shape:    {hidden.shape}  (examples, neurons)")
    print(f"W1 std: {W1.std():.3f}  (He target sqrt(2/64) = {np.sqrt(2 / 64):.3f})")
    # He init deliberately gives outputs a std of about sqrt(2) = 1.41: the
    # ReLU added in step 5 zeroes the negative half, which brings it back down.
    print(f"output mean {hidden.mean():.3f}, std {hidden.std():.3f}  "
          "(expected about 0 and 1.41)")
    # Number of numbers this layer will learn: 64*32 weights + 32 biases.
    print(f"learnable parameters: {W1.size + b1.size}")

    # --- Self-check ---
    assert np.allclose(out_tiny, [[5.5, 2.0, -1.0]]), (
        "tiny example is wrong - check the formula X @ W + b"
    )
    assert W1.shape == (64, 32), f"W1 should be (64, 32), got {W1.shape}"
    assert b1.shape == (32,), f"b1 should be (32,), got {np.shape(b1)}"
    assert np.all(b1 == 0), "biases should start at 0"
    assert abs(W1.std() - np.sqrt(2 / 64)) < 0.02, "W1 spread is off - check init"
    assert hidden.shape == (5961, 32), f"output should be (5961, 32), got {hidden.shape}"
    # Same seed -> same weights: the run is reproducible.
    W_again, _ = init_linear(64, 32, np.random.default_rng(seed=0))
    assert np.array_equal(W1, W_again), "same seed must give the same weights"
    print("\nStep 4 OK")
