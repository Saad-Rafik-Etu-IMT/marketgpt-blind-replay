"""MLP learning sprint - Step 9: honest evaluation (did it learn anything real?).

Context
-------
Step 8 ended with "validation accuracy 58.1%". Sounds good... but compared to
what? A number on its own means nothing. This step asks three questions:

1. Does the MLP beat simple rules that IGNORE the inputs?
   - "Always up":        predict up every day (accuracy baseline);
   - "Constant up-rate": always say P(up) = train up-share, 53.6%
                         (loss baseline, from step 6).
   If the MLP cannot beat these, its 64 inputs added nothing.

2. Is the difference bigger than luck?
   Validation has only 628 examples. Even a model with NO real skill gets
   a slightly different accuracy on every sample of days. The *standard
   error* measures that luck:

       standard_error = sqrt(p * (1 - p) / n)

   With p = 0.57, n = 628 -> about 0.02 (2 percentage points). A rough rule:
   a difference smaller than ~2 standard errors (~4 points here) is not
   distinguishable from luck.

3. Does the result survive a different random seed?
   Seed 0 is just one draw: other seeds give other starting weights and
   batch orders. If the "edge" vanishes with another seed, it was luck.
   (The project's benchmark protocol asks for at least 3 seeds.)

Why we still don't touch the test set
-------------------------------------
Even here, we evaluate on VALIDATION only. The test set (2020+) is reserved
for the final, once-only evaluation of the whole project (F7). Looking at it
now, then changing anything, would turn it into a second validation set.

How to run (from the repository root):
    .venv\\Scripts\\python.exe backend\\learning\\mlp\\step9_evaluate.py
"""

import sys
from pathlib import Path

import numpy as np

# Make steps 1-8 importable (see step 2).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from step5_forward import mlp_forward  # noqa: E402
from step6_loss import cross_entropy  # noqa: E402
from step8_train import load_datasets, train  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def accuracy(predicted: np.ndarray, y: np.ndarray) -> float:
    """Share of examples where the predicted class equals the true class."""
    # 1. Compare element by element, then take the share of True values.
    #    Hint: (a == b) gives True/False per example; .mean() treats True as 1.
    return float((predicted == y).mean())


def standard_error(p: float, n: int) -> float:
    """Typical size of pure-luck variation of an accuracy p measured on n examples."""
    # 2. Apply sqrt(p * (1 - p) / n). Hint: np.sqrt(...)
    return float(np.sqrt(p * (1 - p) / n))


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_datasets()
    n = len(y_val)

    # --- Baselines that ignore the inputs ---
    # 3. "Always up": one prediction per validation example, all equal to 1.
    #    Hint: np.ones(n, dtype=int)
    always_up = np.ones(n, dtype=int)
    base_acc = accuracy(always_up, y_val)

    # "Constant up-rate": the same probabilities for every day, learned from
    # TRAIN only (using the validation up-rate here would be cheating).
    up_rate = y_train.mean()
    constant_probs = np.tile([1 - up_rate, up_rate], (n, 1))
    base_loss = cross_entropy(constant_probs, y_val)

    se = standard_error(base_acc, n)
    print(f"Validation: {n} examples")
    print(f"  always up        accuracy {base_acc:.3f}")
    print(f"  constant {up_rate:.1%} up loss     {base_loss:.4f}")
    print(f"  luck margin: standard error {se:.3f} -> differences under "
          f"~{2 * se:.1%} are not distinguishable from luck\n")

    # --- The MLP, trained with several seeds ---
    print(f"{'seed':>4} {'val loss':>9} {'val acc':>8} {'acc - always up':>16} "
          f"{'predicts up':>12}")
    results = []
    for seed in SEEDS:
        params, _ = train(X_train, y_train, X_val, y_val, seed=seed)
        probs = mlp_forward(X_val, params)[0]
        predicted = probs.argmax(axis=1)
        result = {
            "seed": seed,
            "loss": cross_entropy(probs, y_val),
            "acc": accuracy(predicted, y_val),
            "up_share": float(predicted.mean()),
        }
        results.append(result)
        print(f"{seed:>4} {result['loss']:>9.4f} {result['acc']:>8.3f} "
              f"{result['acc'] - base_acc:>+16.3f} {result['up_share']:>11.0%}")

    mean_loss = np.mean([r["loss"] for r in results])
    mean_acc = np.mean([r["acc"] for r in results])
    print(f"{'mean':>4} {mean_loss:>9.4f} {mean_acc:>8.3f} {mean_acc - base_acc:>+16.3f}")

    # --- Verdict ---
    print("\nVerdict:")
    print(f"  loss:     MLP {mean_loss:.4f} vs constant {base_loss:.4f} "
          f"-> difference {base_loss - mean_loss:+.4f} (tiny)")
    print(f"  accuracy: MLP {mean_acc:.3f} vs always up {base_acc:.3f} "
          f"-> difference {mean_acc - base_acc:+.3f}, well inside the luck margin")
    print("  -> no measurable edge: on validation, the MLP is not distinguishable")
    print("     from 'always predict up'. It mostly predicts up, too.")

    # What to take away:
    # - This is the EXPECTED outcome. Next-day direction of a big index fund
    #   is close to unpredictable from its own past returns; if a 2000-weight
    #   network beat it easily, everyone would be doing it.
    # - Seed 0 looked like +0.8 points. Other seeds are BELOW the baseline.
    #   One lucky seed is not a result: always compare several seeds.
    # - The pipeline is still a success: correct data, no lookahead, verified
    #   gradients, reproducible training, honest evaluation. That is exactly
    #   what a baseline (project F6) must be, so that MarketGPT can be
    #   compared against it fairly later.

    # --- Self-check ---
    assert accuracy(np.array([1, 0, 1, 1]), np.array([1, 1, 1, 0])) == 0.5, (
        "accuracy is wrong - share of positions where predicted == true"
    )
    assert abs(standard_error(0.5, 100) - 0.05) < 1e-12, "standard error formula is wrong"
    assert abs(base_acc - 0.5732484076433121) < 1e-12, (
        "always-up accuracy is wrong - predict 1 for every validation example"
    )
    assert abs(base_loss - 0.6851572009343467) < 1e-12, "constant baseline loss changed"
    assert abs(results[0]["acc"] - 0.5812) < 1e-3, "seed 0 should reproduce step 8"
    assert abs(mean_acc - base_acc) < 2 * se, "MLP vs always-up should be within luck"
    print("\nStep 9 OK - MLP learning sprint complete")
