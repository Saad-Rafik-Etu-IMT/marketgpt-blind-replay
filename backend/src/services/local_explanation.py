"""Local constrained verbalization. All permitted statements have measured evidence."""

import itertools
import json
import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.request import Request, urlopen

lock = Lock()


@lru_cache(maxsize=1)
def load(path: str) -> Any:
    # The optional binding is deliberately loaded only when configured.
    return import_module("llama_cpp").Llama(
        model_path=path, n_ctx=512, n_threads=2, verbose=False
    )


def verbalize(path: str, facts: list[str]) -> list[str]:
    grammar_class = import_module("llama_cpp").LlamaGrammar

    # The model selects the order of the three evidence sentences. JSON quoting
    # escapes decimal points, punctuation and whitespace as literal grammar text.
    candidates = {
        " ".join(order): list(order) for order in itertools.permutations(facts)
    }
    grammar = grammar_class.from_string(
        "root ::= " + " | ".join(json.dumps(x) for x in candidates), verbose=False
    )
    with lock:
        result = load(path)(
            "Explain this validation decision in three sentences. Evidence: "
            + " ".join(facts)
            + "\nExplanation:",
            max_tokens=190,
            temperature=0,
            grammar=grammar,
        )
    output = result["choices"][0]["text"].strip()
    if output not in candidates:
        raise ValueError("Incomplete explanation")
    return candidates[output]


def _matches_facts(sentences: object, facts: list[str]) -> bool:
    return (
        isinstance(sentences, list)
        and all(isinstance(sentence, str) for sentence in sentences)
        and sorted(sentences) == sorted(facts)
    )


def explain_facts(facts: list[str]) -> dict[str, str | list[str]]:
    """Let a local model reorder measured facts, never invent new evidence."""
    model = os.getenv("MRTL_OLLAMA_MODEL", "")
    gguf = os.getenv("MRTL_GGUF_MODEL")
    if gguf:
        try:
            generated = verbalize(gguf, facts)
            if _matches_facts(generated, facts):
                return dict(
                    source=Path(gguf).name,
                    check="Constrained generation; all three facts verified",
                    facts=generated,
                )
        except (ImportError, OSError, ValueError, RuntimeError):
            pass
    if model:
        try:
            prompt = (
                "Write a concise decision explanation using these sentences verbatim. Order them by importance. Return JSON with a sentences array containing each sentence exactly once. "
                + json.dumps(facts)
            )
            body = json.dumps(
                dict(
                    model=model,
                    prompt=prompt,
                    stream=False,
                    format="json",
                    options=dict(temperature=0, num_predict=180),
                )
            ).encode()
            with urlopen(
                Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=body,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=30,
            ) as res:
                sentences = json.loads(json.load(res)["response"])["sentences"]
            if _matches_facts(sentences, facts):
                return dict(
                    source=model,
                    check="All cited sentences matched measured facts",
                    facts=sentences,
                )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return dict(
        source="Measured template",
        check=(
            "Local generation unavailable or failed fact validation"
            if model or gguf
            else "Local model is not configured"
        ),
        facts=facts,
    )
