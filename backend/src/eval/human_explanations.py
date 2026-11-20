"""Score an anonymous, counterbalanced study of explanation comprehension."""

from collections import defaultdict
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participant: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,48}$")
    case: int = Field(ge=1)  # One-based case number as shown in the cockpit.
    format: Literal["facts", "llm"]
    source: str = Field(min_length=1, max_length=120)
    fallback: bool = False
    probability_percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    block: int = Field(ge=1, le=8)
    preference: Literal["facts", "llm", "equal"] | None = None


class Responses(BaseModel):
    model_config = ConfigDict(extra="forbid")
    responses: list[Response] = Field(min_length=1, max_length=5000)


def score_responses(responses: Responses, cases):
    scores = defaultdict(lambda: defaultdict(list))
    seen = set()
    preferences = {}
    counts = {"facts": 0, "llm": 0}
    totals = {"facts": 0, "llm": 0}
    fallback = 0
    for response in responses.responses:
        identity = (response.participant, response.case)
        if identity in seen:
            raise ValueError("Each participant may see a case only once")
        seen.add(identity)
        if not 1 <= response.case <= len(cases):
            raise ValueError("Case is outside the selected run")
        if response.format == "facts" and response.fallback:
            raise ValueError("Only an LLM-assigned case can have a fallback")
        case = cases[response.case - 1]
        sensitivity = [abs(v) for v in case["occlusion"]]
        probability_ok = abs(response.probability_percent - 100 * case["p"]) <= 0.051
        block_ok = max(sensitivity) - sensitivity[response.block - 1] < 1e-12
        score = (int(probability_ok) + int(block_ok)) / 2
        scores[response.participant][response.format].append(score)
        counts[response.format] += 1
        totals[response.format] += int(probability_ok) + int(block_ok)
        fallback += int(response.fallback)
        if response.preference is not None:
            if (
                response.participant in preferences
                and preferences[response.participant] != response.preference
            ):
                raise ValueError("Conflicting preference for one participant")
            preferences[response.participant] = response.preference
    pairs = [
        dict(
            participant=participant,
            facts=mean(formats["facts"]),
            llm=mean(formats["llm"]),
        )
        for participant, formats in sorted(scores.items())
        if "facts" in formats and "llm" in formats
    ]
    return dict(
        participants=len(scores),
        pairs=pairs,
        incomplete=len(scores) - len(pairs),
        counts=counts,
        correct=totals,
        fallback=fallback,
        preference={
            name: list(preferences.values()).count(name)
            for name in ("facts", "llm", "equal")
        },
    )
