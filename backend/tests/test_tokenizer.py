"""MO3 numeric tokenizer tests: synthetic data only, no market downloads."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.ai.tokenizer import NumericReturnTokenizer, TokenizerError

LOOKBACK = 64


def sample(offset: float = 0.0) -> list[float]:
    return [offset + (index - 32) / 1_000 for index in range(LOOKBACK)]


def test_fit_creates_expected_number_of_quantile_bins() -> None:
    train = [sample(offset) for offset in (0.0, 0.01, 0.02, 0.03)]
    tokenizer = NumericReturnTokenizer(num_bins=32).fit(train)

    assert tokenizer.is_fitted
    assert tokenizer.vocab_size == 32
    assert len(tokenizer.bin_edges) == 31
    assert all(a < b for a, b in zip(tokenizer.bin_edges, tokenizer.bin_edges[1:]))


def test_encode_preserves_single_sample_and_batch_shapes() -> None:
    tokenizer = NumericReturnTokenizer().fit([sample(), sample(0.01)])

    encoded_sample = tokenizer.encode(sample())
    encoded_batch = tokenizer.encode([sample(), sample(0.01)])

    assert isinstance(encoded_sample, list)
    assert len(encoded_sample) == LOOKBACK
    assert len(encoded_batch) == 2
    assert all(len(row) == LOOKBACK for row in encoded_batch)


def test_token_ids_are_in_effective_vocabulary() -> None:
    tokenizer = NumericReturnTokenizer().fit([sample(), sample(0.01)])
    encoded = tokenizer.encode([sample(-10), sample(10)])
    flat = [token for row in encoded for token in row]

    assert min(flat) >= 0
    assert max(flat) < tokenizer.vocab_size


def test_extreme_values_are_clipped_to_first_or_last_bin() -> None:
    tokenizer = NumericReturnTokenizer(num_bins=4).fit([sample()])
    values = [-1_000.0, *([0.0] * (LOOKBACK - 2)), 1_000.0]

    encoded = tokenizer.encode(values)

    assert encoded[0] == 0
    assert encoded[-1] == tokenizer.vocab_size - 1


def test_duplicate_quantiles_reduce_effective_vocab_without_invalid_ids() -> None:
    constant = [[0.001] * LOOKBACK, [0.001] * LOOKBACK]
    tokenizer = NumericReturnTokenizer(num_bins=32).fit(constant)

    assert tokenizer.vocab_size == 1
    assert tokenizer.encode(constant) == [[0] * LOOKBACK, [0] * LOOKBACK]


def test_fit_requires_a_two_dimensional_training_matrix() -> None:
    with pytest.raises(TokenizerError, match="2D training matrix"):
        NumericReturnTokenizer().fit(sample())


@pytest.mark.parametrize(
    "bad_value", [True, math.nan, math.inf, -math.inf, "0.01", None]
)
def test_fit_rejects_non_finite_or_non_numeric_values(bad_value: object) -> None:
    train = [sample()]
    train[0][10] = bad_value  # type: ignore[list-item]

    with pytest.raises(TokenizerError, match="finite number"):
        NumericReturnTokenizer().fit(train)


def test_encode_requires_exactly_64_returns() -> None:
    tokenizer = NumericReturnTokenizer().fit([sample()])

    with pytest.raises(TokenizerError, match="exactly 64"):
        tokenizer.encode([0.0] * 63)


def test_encode_before_fit_is_rejected() -> None:
    with pytest.raises(TokenizerError, match="not fitted"):
        NumericReturnTokenizer().encode(sample())


def test_validation_or_test_values_do_not_change_fitted_bins() -> None:
    train = [sample(), sample(0.01)]
    tokenizer = NumericReturnTokenizer().fit(train)
    fitted_edges = tokenizer.bin_edges

    # Encoding extreme holdout values must not refit or mutate train-only bins.
    tokenizer.encode([sample(-100), sample(100)])

    assert tokenizer.bin_edges == fitted_edges


def test_save_and_load_preserve_encoding(tmp_path: Path) -> None:
    tokenizer = NumericReturnTokenizer().fit([sample(), sample(0.01)])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    restored = NumericReturnTokenizer.load(path)

    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.encode(sample()) == tokenizer.encode(sample())


def test_from_dict_rejects_unsorted_or_duplicate_edges() -> None:
    payload = {
        "format_version": 1,
        "num_bins": 4,
        "lookback": LOOKBACK,
        "bin_edges": [0.1, 0.1],
    }

    with pytest.raises(TokenizerError, match="strictly increasing"):
        NumericReturnTokenizer.from_dict(payload)


def test_from_dict_rejects_unknown_format_and_extra_fields() -> None:
    payload = {
        "format_version": 99,
        "num_bins": 4,
        "lookback": LOOKBACK,
        "bin_edges": [],
    }
    with pytest.raises(TokenizerError, match="format_version"):
        NumericReturnTokenizer.from_dict(payload)

    payload["format_version"] = 1
    payload["unexpected"] = True
    with pytest.raises(TokenizerError, match="missing or unexpected"):
        NumericReturnTokenizer.from_dict(payload)


def test_load_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(TokenizerError, match="cannot load"):
        NumericReturnTokenizer.load(path)


def test_serialized_state_is_valid_json(tmp_path: Path) -> None:
    tokenizer = NumericReturnTokenizer().fit([sample()])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["lookback"] == LOOKBACK

