"""The language provider may order evidence but cannot change it."""

import io
import json

import pytest

from src.services import local_explanation

FACTS = ["Probability: 70%.", "Target: cash.", "Measured sensitivity: 5%."]


@pytest.mark.parametrize(
    "sentences,accepted",
    [
        ([FACTS[2], FACTS[0], FACTS[1]], True),
        ([FACTS[0], FACTS[0], FACTS[2]], False),
        (FACTS[:2], False),
        (["A different prediction.", *FACTS[1:]], False),
        (None, False),
    ],
)
def test_ollama_requires_verbatim_evidence(monkeypatch, sentences, accepted):
    monkeypatch.delenv("MRTL_GGUF_MODEL", raising=False)
    monkeypatch.setenv("MRTL_OLLAMA_MODEL", "local-test")
    reply = json.dumps({"response": json.dumps({"sentences": sentences})})
    monkeypatch.setattr(
        local_explanation, "urlopen", lambda *a, **kw: io.StringIO(reply)
    )
    result = local_explanation.explain_facts(FACTS)
    assert result["source"] == ("local-test" if accepted else "Measured template")
    assert result["facts"] == (sentences if accepted else FACTS)


def test_provider_failure_preserves_measured_facts(monkeypatch):
    monkeypatch.delenv("MRTL_OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("MRTL_GGUF_MODEL", "local.gguf")

    def unavailable(*args):
        raise OSError("model unavailable")

    monkeypatch.setattr(local_explanation, "verbalize", unavailable)
    assert local_explanation.explain_facts(FACTS)["facts"] == FACTS


def test_gguf_uses_same_evidence_contract(monkeypatch):
    monkeypatch.delenv("MRTL_OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("MRTL_GGUF_MODEL", "local.gguf")
    monkeypatch.setattr(local_explanation, "verbalize", lambda *args: FACTS[::-1])
    assert local_explanation.explain_facts(FACTS)["source"] == "local.gguf"
    monkeypatch.setattr(local_explanation, "verbalize", lambda *args: FACTS[:1])
    assert local_explanation.explain_facts(FACTS)["source"] == "Measured template"
