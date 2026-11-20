"""Reproducible development campaign recipes; no protected-test reads."""

import argparse
from importlib.util import find_spec
from uuid import UUID, uuid4

from src.experiments.spec import ExperimentSpec


def campaigns(transformer: bool = False):
    definitions = [
        (
            "smoke",
            "Pipeline smoke test",
            "Training and checkpoint publication work end to end.",
            {},
        ),
        (
            "positive",
            "Synthetic positive control",
            "A learnable causal signal should beat a constant predictor.",
            {
                "sources": ["synthetic_signal"],
                "profile": "research",
                "epochs": 20,
                "presets": ["small"],
            },
        ),
        (
            "negative",
            "Synthetic negative control",
            "Random-walk inputs should not establish predictive superiority.",
            {
                "profile": "research",
                "epochs": 20,
                "presets": ["small"],
            },
        ),
        (
            "capacity",
            "Model capacity",
            "Additional hidden layers improve selection loss only if useful.",
            {
                "profile": "research",
                "presets": ["w1_baseline", "small", "medium", "large"],
            },
        ),
        (
            "features",
            "Feature family ablation",
            "Additional feature families improve performance compared with returns only.",
            {
                "profile": "research",
                "presets": ["small"],
                "feature_profiles": [
                    "returns_only_v1",
                    "returns_plus_trend",
                    "returns_plus_volatility",
                    "returns_plus_price_structure",
                    "returns_plus_volume",
                    "returns_plus_statistics",
                ],
            },
        ),
        (
            "optimization",
            "Bounded hyperparameter search",
            "Search on inner validation, with paired seeds.",
            {
                "profile": "research",
                "presets": ["small", "medium"],
                "search": {"budget": 6},
            },
        ),
        (
            "targets",
            "Multi-target study",
            "Joint direction and volatility learning can share representations.",
            {
                "profile": "research",
                "presets": ["small"],
                "targets": ["direction_binary_v1", "volatility_h5"],
            },
        ),
        (
            "confirmation",
            "Chronological confirmation",
            "Candidate performance is consistent across purged chronological folds.",
            {
                "profile": "confirmation",
                "presets": ["small"],
            },
        ),
    ]
    definitions += [
        (
            "horizon",
            "Multi-horizon returns",
            "Prediction quality changes with the economic horizon.",
            {
                "profile": "research",
                "presets": ["small"],
                "targets": ["expected_return_h1", "expected_return_h5"],
            },
        ),
        (
            "cost",
            "Cost-aware direction",
            "An explicit one-basis-point hurdle changes the classification problem.",
            {
                "profile": "research",
                "presets": ["small"],
                "targets": ["direction_binary_v1", "direction_above_1bp_v1"],
            },
        ),
    ]
    if transformer:
        definitions += [
            (
                "architectures",
                "MLP versus Transformer",
                "Compare model families on the same standardized return window.",
                {
                    "profile": "research",
                    "presets": ["small", "transformer_reference"],
                    "epochs": 8,
                },
            ),
            (
                "attention",
                "Transformer ablations",
                "Isolate position, dropout, weight decay and schedule effects.",
                {
                    "profile": "research",
                    "presets": [
                        "transformer_reference",
                        "transformer_no_position",
                        "transformer_no_dropout",
                        "transformer_no_decay",
                        "transformer_fixed_lr",
                    ],
                    "epochs": 8,
                },
            ),
        ]
    return [
        {
            "id": key,
            "title": title,
            "hypothesis": hypothesis,
            "spec": ExperimentSpec(name=title, auto_evaluate=True, **fields).model_dump(
                mode="json"
            ),
        }
        for key, title, hypothesis, fields in definitions
    ]


def main():
    from src.config import get_settings
    from src.experiments.runner import write_spec
    from src.experiments.worker import main as run_worker

    choices = {
        recipe["id"]: recipe for recipe in campaigns(find_spec("torch") is not None)
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=list(choices))
    parser.add_argument(
        "--output", type=UUID, default=None, help="Optional new campaign UUID"
    )
    parser.add_argument("--source", action="append", help="Override dataset(s)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"])
    args = parser.parse_args()
    payload = dict(choices[args.recipe]["spec"])
    if args.source:
        payload["sources"] = args.source
    if args.epochs is not None:
        payload["epochs"] = args.epochs
    if args.device:
        payload["device"] = args.device
    spec = ExperimentSpec.model_validate(payload)
    output = get_settings().data_dir / "experiments" / str(args.output or uuid4())
    print(f"Campaign: {output.name}", flush=True)
    write_spec(output, spec)
    return run_worker([str(output)])


if __name__ == "__main__":
    raise SystemExit(main())
