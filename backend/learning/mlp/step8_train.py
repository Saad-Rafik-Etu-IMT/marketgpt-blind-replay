"""MLP learning sprint - Step 8: training (gradient descent, thousands of times).

Context
-------
Step 7 showed that ONE step downhill lowers the loss. Training is just that
step, repeated many times:

    parameter = parameter - learning_rate * gradient

Vocabulary
----------
- Mini-batch: instead of computing gradients on all 5961 train examples for
  every step, we use a small random group (64 examples). Each step is ~90x
  cheaper, gives a slightly noisy but good-enough direction, and we get ~94
  steps per pass over the data instead of 1. This is *stochastic gradient
  descent* (SGD).
- Epoch: one full pass over the training set (every example used once).
  We train for several epochs.
- Learning rate: the size of each step. Too small -> learning crawls. Too big
  -> steps overshoot and the loss jumps around or explodes. 0.05 works here.
- Shuffling: before each epoch, the training examples are visited in a new
  random order, so batches mix different periods. This is NOT lookahead:
  every example is already a fixed (past window, next-day label) pair built
  in step 2, and validation/test examples are never in the batches.

Overfitting and early stopping
------------------------------
Training loss measures how well the model fits data it has SEEN. What we
care about is data it has NOT seen: the validation loss. Typical pattern:

    epochs ->   train loss keeps going down (the model memorizes noise)
                validation loss goes down, then back UP   <- overfitting

So after every epoch we measure the validation loss and keep a copy of the
parameters from the BEST validation epoch ("early stopping"). That is the
model we will evaluate in step 9.

Important: because validation was used to CHOOSE the epoch, the validation
score is now slightly optimistic. This is exactly why the project keeps a
separate test set, used only once at the very end.

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step8_train.py
"""

import sys
from pathlib import Path

import numpy as np

# Make steps 1-7 and the project's code importable (see step 2).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from step1_returns import CSV_PATH, load_returns  # noqa: E402
from step2_windows import make_xy  # noqa: E402
from step3_normalize import fit_normalizer, normalize  # noqa: E402
from step5_forward import init_mlp, mlp_forward  # noqa: E402
from step6_loss import cross_entropy  # noqa: E402
from step7_backprop import mlp_backward  # noqa: E402

from src.data.split import build_examples, load_split_config  # noqa: E402

# Training settings ("hyperparameters"). They are chosen by looking at the
# VALIDATION loss, never the test set.
SEED = 0
HIDDEN = 32
LEARNING_RATE = 0.05
BATCH_SIZE = 64
EPOCHS = 40


def train(X_train, y_train, X_val, y_val, *, hidden=HIDDEN, learning_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE, epochs=EPOCHS, seed=SEED):
    """Train the MLP with mini-batch SGD and early stopping on validation loss.

    Returns (best_params, history):
    - best_params: parameters from the epoch with the lowest validation loss;
    - history: one dict per epoch with train_loss, val_loss, val_accuracy.
    """
    params = init_mlp(n_in=X_train.shape[1], n_hidden=hidden, n_out=2, seed=seed)
    # A separate generator for shuffling, also seeded: the whole run is
    # reproducible (same seed -> exactly the same result).
    rng = np.random.default_rng(seed)

    history = []
    best_val_loss = np.inf
    best_params = None

    for epoch in range(1, epochs + 1):
        # New random visiting order of the training examples for this epoch.
        order = rng.permutation(len(y_train))

        # Walk through the shuffled examples, one mini-batch at a time.
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]   # indices of this batch

            # 1. Forward pass on THIS BATCH ONLY (X_train[batch]).
            probs, cache = mlp_forward(X_train[batch], params)

            # 2. Backward pass: gradients for this batch (labels y_train[batch]).
            grads = mlp_backward(cache, y_train[batch], params)
            # 3. Gradient descent: move every parameter a small step downhill.
            for name in params:
                params[name] = params[name] - learning_rate * grads[name]

        # End of epoch: measure on the FULL train and validation sets.
        train_loss = cross_entropy(mlp_forward(X_train, params)[0], y_train)
        val_probs = mlp_forward(X_val, params)[0]
        val_loss = cross_entropy(val_probs, y_val)
        val_accuracy = float((val_probs.argmax(axis=1) == y_val).mean())
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_accuracy": val_accuracy})

        # 4. Early stopping: is this the best validation loss so far?
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # .copy() matters: without it we would store references to the
            # arrays that the next epochs keep modifying, not a snapshot.
            best_params = {name: value.copy() for name, value in params.items()}

    return best_params, history


def print_curves(history: list[dict], width: int = 50) -> None:
    """Text chart of the loss curves (T = train, V = validation, left = lower)."""
    losses = [h[k] for h in history for k in ("train_loss", "val_loss")]
    low, high = min(losses), max(losses)
    best = min(history, key=lambda h: h["val_loss"])["epoch"]

    def column(value):
        return round((value - low) / (high - low) * (width - 1))

    print(f"{'epoch':>5} {'train':>7} {'val':>7} {'val acc':>8}   "
          f"{low:.3f} (lower = better) {'':>{width - 31}}{high:.3f}")
    for h in history:
        line = [" "] * width
        line[column(h["train_loss"])] = "T"
        line[column(h["val_loss"])] = "V"   # V drawn last: wins if they overlap
        marker = "  <- best validation" if h["epoch"] == best else ""
        print(f"{h['epoch']:>5} {h['train_loss']:>7.4f} {h['val_loss']:>7.4f} "
              f"{h['val_accuracy']:>8.3f}   |{''.join(line)}|{marker}")


def load_datasets():
    """Steps 1-3: normalized train and validation sets (test stays locked)."""
    returns = load_returns(CSV_PATH)
    config = load_split_config()
    examples = build_examples(list(returns.index.date), config)
    values = returns.to_numpy()
    X_train, y_train = make_xy(values, examples.train, config.lookback)
    X_val, y_val = make_xy(values, examples.validation, config.lookback)
    mean, std = fit_normalizer(X_train)
    return normalize(X_train, mean, std), y_train, normalize(X_val, mean, std), y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_datasets()

    print(f"Training: {len(y_train)} examples, batches of {BATCH_SIZE} "
          f"-> {int(np.ceil(len(y_train) / BATCH_SIZE))} steps per epoch, "
          f"{EPOCHS} epochs, learning rate {LEARNING_RATE}, seed {SEED}\n")
    best_params, history = train(X_train, y_train, X_val, y_val)
    print_curves(history)

    best = min(history, key=lambda h: h["val_loss"])
    print(f"\nBest epoch {best['epoch']}: validation loss {best['val_loss']:.4f}, "
          f"validation accuracy {best['val_accuracy']:.3f}")
    # What to notice in the chart:
    # - T keeps moving left (train loss falls) almost every epoch;
    # - V moves left at first, then drifts back right: overfitting. The model
    #   is learning patterns of 1993-2016 that do not hold in 2017-2019.
    # - Is the best validation result actually GOOD? Compare it with simple
    #   rules that ignore the inputs: that is step 9.

    # --- Self-check ---
    first, last = history[0], history[-1]
    assert len(history) == EPOCHS, "one history entry per epoch expected"
    assert last["train_loss"] < first["train_loss"] < 0.9020, (
        "train loss should go down - check the forward/backward calls use the "
        "batch, and the update SUBTRACTS learning_rate * gradient"
    )
    assert best_params is not None, "best_params never saved - check the early-stopping test"
    best_val_loss = cross_entropy(mlp_forward(X_val, best_params)[0], y_val)
    assert abs(best_val_loss - best["val_loss"]) < 1e-12, (
        "best_params must be the parameters of the lowest validation loss epoch "
        "(test: val_loss < best_val_loss)"
    )
    again, _ = train(X_train, y_train, X_val, y_val, epochs=2)
    once, _ = train(X_train, y_train, X_val, y_val, epochs=2)
    assert all(np.array_equal(again[k], once[k]) for k in again), (
        "same seed must give exactly the same model"
    )
    print("\nStep 8 OK")
