# Remaining work plan (proposed, 2026-11-17)

This plan comes from an audit of `main` at `a936d3a` on 2026-11-17 (see the
[README audit section](../README.md#audit-2026-11-17-verified-state-and-gaps)).
It is a **proposal**: task ownership describes future responsibility, not completed
work. Names, skills and availability are unknown, so the six-member allocation is
provisional and meant to be adjusted. No durations are given, and no deadline or
grading requirement is assumed.

## Guiding principles

- **The learning objective comes first:** (1) an MLP built and understood by the
  students; (2) a small Transformer time-series classifier; (3) a rigorous
  experimental comparison; (4) LLM-generated natural-language explanations;
  (5) evaluation of predictions and explanations; (6) a cockpit to inspect
  experiments and results. Finance is the test bed, not the goal. Do not grow
  this into a more complex trading platform, and do not rebuild what already works.
- **MarketGPT is a small Transformer classifier**, not a language model. The
  explanation LLM is a separate component.
- **Keep these stages distinct:** predicted class (under a versioned label
  definition), class probabilities, the target policy (Long/Flat), the submitted
  order, and the execution outcome (fill or rejection).
- **Students write the Deep Learning and LLM work themselves:** features and
  labels, tokenization, the MLP and the Transformer (embeddings, attention,
  architecture), losses and training loops, optimization and regularization
  experiments, attribution and its interpretation, prompts and generation
  experiments, and the analysis and conclusions.
- **An AI assistant (Claude, under review) may help with:** environment and
  scripts, interfaces and adapters, logging and artifact export, frontend
  integration, test infrastructure, local LLM transport (timeouts, cache) and
  documentation.

Owner legend: **S** = student, **A** = AI assistant under student review,
**J** = joint. Priority: **must** / **useful** / **optional**. Size S/M/L is
relative and uncertain until the team's level is known.

## Blocking decisions

| ID | Decision | Recommendation (proposal) | Blocks |
| --- | --- | --- | --- |
| DEC-1 | Label and class names | Binary `next_day_up_v1`: `up` if the next adjusted log return is > 0, otherwise `not_up` (matches the W1 sprint and the binary backend) | All models, cockpit contract v2 |
| DEC-2 | Input window | 64 returns for **both** models, so the architecture comparison uses identical inputs. **This changes** the "20 returns → 16 ReLU → 2" MLP baseline recorded in [benchmark.md](benchmark.md) and must be approved explicitly | MLP, Transformer, tokenizer |
| DEC-3 | Transformer framework | PyTorch on CPU (free). Students write attention and blocks by hand rather than using `nn.TransformerEncoder`. Pure NumPy is possible, but backprop through attention is expensive. Requires a dependency/CI PR | Transformer, training |
| DEC-4 | Local LLM | Model, licence and hardware to choose; Ollama is not installed on the audited machine. Measure CPU latency | LLM transport, prompts |
| DEC-5 | Corporate-action policy | See [benchmark.md](benchmark.md#blocker-for-real-data-benchmark-results-corporate-actions). Blocks **only** real-data trading results, not classification experiments | Trading runner results |
| DEC-6 | Ablation list | Four ablations: positional encoding, dropout, learning-rate schedule, weight decay. Shared reference config, same three seeds. **This replaces** the README F12 list (SGD/Adam, warmup, LayerNorm, dropout, weight decay, size). The budget is an internal target, to be confirmed by a pilot | Experiment campaign |

## Minimal integration contracts

These reuse existing contracts where possible: `split.build_examples`,
`BlindObservation` (64 returns), `DecisionRecord`/`ExecutionOutcome` schema v2 and
the cockpit contract v1 as a starting point. They are proposals, to be settled
with DEC-1/DEC-2.

| ID | Contract | Minimal content | Known incompatibility |
| --- | --- | --- | --- |
| IC1 | Label `next_day_up_v1` | `y = 1` iff the adjusted log return of t+1 is > 0 (horizon 1, per [split.json](../backend/config/split.json)); classes `["not_up", "up"]` | Cockpit v1 needs `long/flat/short`; class names currently reuse position names |
| IC2 | Predictor | `predict_proba(X: (n, 64) float) -> (n, 2)` in `[not_up, up]` order; preprocessing loaded from the checkpoint; no ticker or date input | Window size (DEC-2) |
| IC3 | Target policy `threshold-v1` | `long` if `p_up >= θ`, else `flat`; θ = 0.5 or chosen on validation, then recorded | `BinaryProbabilities(long, flat)` in [player.py](../backend/src/ai/player.py) names positions. Proposal: keep the API and document `long = p_up` under `threshold-v1` |
| IC4 | Dataset identity | `dataset_id = sha256(dataset_fingerprint, split_config_sha256, label_version, feature_version)`; normalization statistics and tokenizer bins fitted on train only, saved with the checkpoint | — |
| IC5 | Run and checkpoint (`run.json`) | `run_id`, `model`, `config`, `seed`, `dataset_id`, `label_version`, git commit, `history[{epoch, train_loss, val_loss, val_acc}]`, `best_epoch`, checkpoint path and SHA-256 (the file stays out of git) | Maps to cockpit `ResearchRun` |
| IC6 | Saved prediction | `run_id`, `sample_id`, `split`, `decision_offset`, 64 inputs, probabilities, predicted class; the label is stored separately in the report | — |
| IC7 | Sensitivity | `method: occlusion-v1`, mask size and mask value, baseline `p_up`, `Δp_up` per bar segment | — |
| IC8 | Fact sheet | `facts[{id, bar <= 0, kind, value, unit, text}]`, every number computed by code | Cockpit v1 has only `{id, bar, text}` |
| IC9 | Explanation | `text`, `llm_model`, `prompt_version`, `references`, `checks[{name, passed, detail}]`, `fallback`, `latency_ms` | Close to cockpit v1 |
| IC10 | Evaluation report | Per run and split: n, accuracy, macro-F1, log loss, Brier, ECE, baselines, CI method, per-seed values | Cockpit v1 has only `macroF1` |
| IC11 | Cockpit bundle v2 | v1 plus generic binary classes, `labelVersion`, `policy`, execution stages and sensitivities | **v1 requires a `short` probability**, so a v2 is needed |

## Backlog by scope

### Data and experimental protocol

| ID | Goal and reason | Start → deliverable | Files | Acceptance | Deps | Prio | Owner | Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP1 | Freeze the label, class names and policy (IC1, IC3); without them no two models are comparable | Contradictory docs → accepted protocol note | `docs/training_protocol.md` (new), [benchmark.md](benchmark.md) | Written decision; label and policy versions appear in records | DEC-1, DEC-2 | must | J (students decide) | S |
| DP2 | Multi-ETF training dataset built from the F2 splits | Logic exists only in `backend/learning/` (SPY, hard-coded shapes) → `src/data/dataset.py`: X (n, 64), y, `dataset_id` | `backend/src/data/` | Synthetic tests: no window crosses its split; label taken from t+1; stable `dataset_id` | DP1 | must | S (features/labels) + A (loading, hashing) | M |
| DP3 | Normalization fitted on train only and persisted | `learning/mlp/step3` → serializable fit/transform | same | Test: changing validation/test data leaves the statistics unchanged | DP2 | must | S | S |
| DP4 | F8 model/player safety tests | Partial → predictor never sees future bars; train-only fitting; sealed test manifest refused outside final evaluation | `backend/tests/` | Tests fail when a leak is injected | DP2, MO2 | must | J | M |
| DP5 | Corporate-action policy | Documented blocker → chosen and tested policy. **An explicit, announced execution-rule change**: new execution policy version or benchmark profile | `services/`, `core/`, `docs/` | Synthetic ex-dividend tests | DEC-5 | useful (must for real-data trading results) | J | L |

### Models and training

| ID | Goal and reason | Start → deliverable | Files | Acceptance | Deps | Prio | Owner | Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MO1 | Rule-based baselines: buy-and-hold, MA 20/50 crossover | None → `Player` classes | `backend/src/ai/baselines.py` | Synthetic tests; `probabilities=None` | — | must | A | S |
| MO2 | **Product MLP** (NumPy, ported from the W1 sprint): forward, backprop, training, early stopping, save/load, `predict_proba` | Learning scripts → tested module and checkpoint | `backend/src/ai/mlp.py` | Gradient check in pytest; same seed gives the same history; follows IC2/IC5 | DP2, DP3 | must | S | M |
| MO3 | **Numeric tokenizer**: ~32 quantile bins fitted on train | None → `fit/encode` with saved bins | `backend/src/ai/tokenizer.py` | Train-only fit; extreme values handled; ids in `[0, V)` | DP2 | must | S | M |
| MO4 | **Transformer**: token and positional embeddings, hand-written self-attention, blocks, two-class head | None → model plus shape/mask tests | `backend/src/ai/marketgpt/model.py` | Synthetic-tensor tests; parameter count reported; overfits one mini-batch | MO3, DEC-3 | must | S | L |
| MO5 | Transformer training loop: loss, optimizer, LR schedule, weight decay, early stopping, seeds | None → `train_marketgpt` writing IC5 | `backend/src/ai/marketgpt/train.py` | Deterministic on CPU at a fixed seed; validation only | MO4, IN1 | must | S | M |
| MO6 | Predictor → `Player` adapter (policy IC3) | None → `ModelPlayer` | `backend/src/ai/model_player.py` | Tested through `decide_and_submit` on a synthetic scenario | MO2, DP1 | must | A | S |
| MO7 | LSTM, attention heatmaps | — | — | — | MO4 | optional | S | M |

### Experiments, evaluation and reproducibility

| ID | Goal and reason | Start → deliverable | Files | Acceptance | Deps | Prio | Owner | Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IN1 | Run logging; writes `run.json` and predictions (IC5, IC6) | None → shared utility | `backend/src/eval/artifacts.py` | Every run records commit, seed, `dataset_id` and checkpoint hash | DP2 | must | A | S |
| EV1 | Validation classification metrics (accuracy, macro-F1, log loss, Brier) against *always up* and a constant prior; block-bootstrap CIs | Only in `step9` → module and IC10 report | `backend/src/eval/classification.py` | Metrics checked on hand-computed cases; CI method documented | MO2 | must | S | M |
| EV2 | F13 calibration: reliability diagram, ECE | None | same and cockpit | Figure and ECE per run and seed | EV1 | must | S | S |
| EV3 | **Pilot**, then the MLP vs Transformer comparison and four ablations (DEC-6) × three seeds on a shared reference config | None → table and curves | `backend/config/experiments/`, scripts | Pilot confirms the budget; everything selected on validation; CIs reported | MO2, MO5, EV1 | must | S | L |
| EV4 | F7 runner: scenarios → sessions → players → records → final equity, drawdown, Sharpe, exposure, rejections | Building blocks exist → runner on `development` scenarios | `backend/src/eval/runner.py` | benchmark-v2 profile verified; dataset fingerprint checked; identical rules for all players | MO1, MO6 | useful (must for trading results) | A | L |
| EV5 | Protected final evaluation: one command, test set read **once** | None → script and usage log | `backend/src/eval/`, `docs/` | Reproducible command; configuration frozen before the test is opened | EV3 (+ EV4, DP5) | must | J | M |

### Explanation generation and evaluation

| ID | Goal and reason | Start → deliverable | Files | Acceptance | Deps | Prio | Owner | Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EX1 | Occlusion sensitivity (IC7) | None → generic function over `predict_proba` | `backend/src/ai/attribution.py` | Tests on a toy model with known sensitivity; visible bars only | MO2 (enough to start) | must | S | M |
| EX2 | Fact-sheet builder (IC8); numbers computed by code | None | `backend/src/ai/facts.py` | Every fact traces to a computed value | EX1 | must | J | S |
| EX3 | Local LLM transport: Ollama client, timeout, cache keyed by decision and input identity, fake client for offline tests | None | `backend/src/ai/llm_client.py` | Offline tests; a timeout produces the template fallback | DEC-4 | must | A | M |
| EX4 | Prompts and generation experiments (versioned prompts) | None | `backend/src/ai/prompts/` | At least two prompt versions compared on the same fact sheets | EX2, EX3 | must | S | M |
| EX5 | Validator and template fallback: citations on visible bars only, numbers equal to facts within tolerance, direction, no identifying information | None | `backend/src/ai/validator.py` | Adversarial test set; fallback always labelled | EX2 | must | J (students design rules) | M |
| EX6 | Explanation evaluation: fallback rate, numeric agreement, top-3 vs three random bars occlusion, blind human rating of LLM vs template | None | `backend/src/eval/explanations.py` | Report with n and limitations; no causal claim | EX4, EX5 | must | S | M |
| EX7 | F15 blind vs dated experiment | — | — | — | EX4 | optional | S | M |

### Cockpit and integration

| ID | Goal | Start → deliverable | Files | Acceptance | Deps | Prio | Owner | Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CK1 | Binary contract v2 (IC11); v1 bundles get a clear rejection message | v1 three-class → v2 | [contract.ts](../frontend/lib/research/contract.ts), `demo.ts`, tests | Existing vitest cases kept or adapted without weakening | DP1 | must | A | M |
| CK2 | Backend exporter: runs, predictions, reports, explanations → bundle v2 | None | `backend/src/eval/export.py` | A bundle exported from MO2 passes the v2 parser (cross-checked with an example file) | IN1, CK1 | must | A | M |
| CK3 | Run comparison view (MLP/Transformer, ablations, calibration) | Deferred → view | `frontend/components/research/` | Shows seeds and CIs; no metric recomputed in the UI | CK2, EV1 | must | J | M |
| CK4 | Real execution stages in Inspector (target, order, fill/rejection, `rejection_reason`) | Shown as "Unavailable" | same | Fed only from `DecisionRecord`/`ExecutionOutcome` | EV4 | useful | A | S |

### Documentation and defense

| ID | Goal | Files | Acceptance | Prio | Owner | Size |
| --- | --- | --- | --- | --- | --- | --- |
| DO1 | Keep README status and evidence consistent | README, [benchmark.md](benchmark.md) | Every status links to evidence | must | A | S |
| DO2 | Training-protocol and explanation-validation-protocol documents | `docs/` | Reviewed by another member | must | J | S |
| DO3 | Environment: explicit numpy and PyTorch CPU in the manifest/lock and CI; reproduction command | `pyproject.toml`, lock, CI | CI green with the new dependencies; coverage gate unchanged | must | A | S |
| DO4 | Demo under three minutes, backup video, report, two rehearsals | — | Every member can explain the whole pipeline | must | J | M |

## Proposed six-member allocation

This is aligned with the existing pairs where possible: A = Members 1 + 6,
B = Members 2 + 3, C = Members 4 + 5. **Proposed change:** occlusion attribution
moves from pair B to Member 5 (pair C). Every member owns a DL/LLM learning
contribution plus a secondary contribution in another scope.

| Member | Main scope | Tasks (secondary) | DL/LLM learning | Implements personally | Can delegate to the AI assistant | Inputs from | Deliverable | Defense demo | Reviewer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Data, labels, integrity | DP1, DP2, DP3, DP4 (EV4 owner, DP5 preparation) | Label definition, class imbalance, leakage, train-only preprocessing | Windows and labels, normalization, leakage tests | CSV loading, `dataset_id` hashing, runner skeleton | 2, 3 (model needs), 4 (policy) | `dataset.py`, protocol note | Perturbing future/test data changes neither X nor statistics | 6 |
| 2 | MLP and classification metrics | MO2, EV1, MO1 review (CK3) | Backprop, initialization, early stopping, metrics, bootstrap | NumPy MLP, training, metrics and CIs | Checkpoint I/O, IN1 wiring, CK3 components | 1 | First real run, IC10 report | Curves, comparison with *always up*, CIs | 3 |
| 3 | Tokenizer and Transformer | MO3, MO4 (MO7 optional) | Quantization, embeddings, positional encoding, multi-head attention, LayerNorm and residuals | All of MO3 and MO4 | Shape tests, synthetic fixtures, DO3 | 1, DEC-3 | `tokenizer.py`, `model.py` | 64 returns → tokens → probabilities | 2, handover to 4 |
| 4 | Optimization, ablations, calibration | MO5, EV3, EV2 (CK2 review) | Optimizers, LR schedules, weight decay, dropout, seed variance, calibration | Training loop, ablation configs, analysis | Config launcher, logging, export | 3, 2 | MLP vs Transformer table, 4 × 3-seed ablations, reliability diagrams | Ablation table and calibration in the cockpit | 5 |
| 5 | Attribution, facts, validator | EX1, EX2, EX5 (CK1, CK4) | Occlusion and its limits (not causal), automated-check design | Occlusion, fact selection, validator rules | IC7/IC8 serialization, contract v2, display | 2 (MLP checkpoint), then 4 | Sensitivities, fact sheets, validator and template | Most sensitive bars → facts → a failing check → template fallback | 4, 6 |
| 6 | LLM generation and explanation evaluation | EX4, EX6, EX3 with the AI assistant (DO4 coordination, EX7 optional) | Prompting, decoding, hallucination, blind human-evaluation protocol | Prompts, generation experiments, EX6 protocol and analysis | Ollama client, timeouts, cache, fake client | 5, DEC-4 | Versioned explanations, explanation report | Live or cached generation → checks → ratings | 1, 5 |

**File conflicts and parallel-work risks**

- `backend/src/ai/` is shared: one file per task (see the paths above). Do not
  change `player.py` without agreement.
- `frontend/lib/research/contract.ts` is a single file: one CK1 PR at a time.
- The README status table is a conflict hotspot. Each member updates their own
  row; DO1 consolidates.
- `pyproject.toml` and the lock: a single DO3 PR adds PyTorch.
- [split.json](../backend/config/split.json), the scenario manifests and the
  benchmark profiles are **frozen**: do not edit them.
- Member 4 depends on Members 3 and 1, and can start the pilot protocol on the
  MLP (MO2) while MO4 is in progress.

## Milestones and critical path

**Prerequisites (milestone 0):** DEC-1 to DEC-4, DP1 and DO3, plus drafts of IC1–IC11.

**Critical path:** DP1 → DP2/DP3 → MO3 → MO4 → MO5 → EV3 pilot → EV3 campaign →
EV5 (test read once) → DO4.

| Milestone | Visible result | Parallel work | Exit criterion |
| --- | --- | --- | --- |
| 1. MLP trained, evaluated and shown in the cockpit | Real MLP run with curves and report | MO2, IN1, CK1, CK2, MO1; Member 3 starts MO3 | A bundle exported from MO2 loads in the cockpit; tests green |
| 2. Transformer compared with the MLP | Same config and same seeds | MO4, MO5, EV1, CK3; Member 5 starts EX1 on the MLP checkpoint | At least three seeds per model; CIs; validation-only selection |
| 3. A real LLM explanation with facts and checks | Facts → LLM → checks → fallback | EX1–EX5 (can start at milestone 1), EV2 | Local generation works; the validator rejects the adversarial cases |
| 4. Reproducible final evaluation | Ablations, calibration, explanation evaluation, final test | EV3, EV6, EX6; EV4 (+ DP5 if trading results are reported) | One command reproduces the tables; config frozen before the test is opened |
| 5. Demo and defense | Demo under three minutes, backup video | DO1, DO2, DO4 | Two rehearsals; every member explains the whole pipeline |

The LLM track does not wait for the Transformer: the MLP checkpoint is enough to
develop it, and it is replayed on the Transformer afterwards.

## First actions (at most three per member)

| Member | Actions |
| --- | --- |
| 1 | Write the DP1 proposal (label, classes, policy) for a team decision; specify `dataset_id` and the DP2 inputs/outputs; write the DP2/DP3 leakage tests first |
| 2 | List what moves from `learning/mlp` into `src/ai/mlp.py`; define the MLP checkpoint format per IC5; write the gradient-check test |
| 3 | Argue DEC-3; design the quantile bins (V, extremes); sketch the shape flow (tokens → embeddings → attention → head) |
| 4 | Propose the reference config and confirm DEC-6; define the pilot protocol (budget, seeds 0/1/2); specify `run.json` (IC5) with the AI assistant for IN1 |
| 5 | Define `occlusion-v1` (mask size and value); list the fact kinds (IC8); write ten adversarial validator cases |
| 6 | Propose DEC-4 (model, licence, CPU latency measurement); draft prompt v0 on a synthetic fact sheet; outline the blind human-rating protocol |

Ready for the AI assistant once DEC-1 is settled: DO3, CK1, IN1, MO1.

## Constraints that stay in force

No future data in decisions; preprocessing fitted on train only; validation for
development and a protected test set read once; no data, databases, weights or
secrets in git; no paid services; no silent execution-rule changes; no lowered
coverage gate or deleted tests. Measured sensitivities do not prove causal
reasoning, and automated checks do not guarantee that generated text is fully true.
LSTM, multiplayer, training orchestration, attention heatmaps and the LLM
blind/dated experiment stay optional unless explicitly decided otherwise.
