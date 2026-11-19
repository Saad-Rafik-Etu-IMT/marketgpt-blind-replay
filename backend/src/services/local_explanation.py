"""Local constrained verbalization. All permitted statements have measured evidence."""

import itertools
import json
from functools import lru_cache
from threading import Lock

lock = Lock()


@lru_cache(maxsize=1)
def load(path):
    from llama_cpp import Llama

    return Llama(model_path=path, n_ctx=512, n_threads=2, verbose=False)


def verbalize(path, facts):
    from llama_cpp import LlamaGrammar

    # The model selects the order of the three evidence sentences. JSON quoting
    # escapes decimal points, punctuation and whitespace as literal grammar text.
    candidates = {
        " ".join(order): list(order) for order in itertools.permutations(facts)
    }
    grammar = LlamaGrammar.from_string(
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
