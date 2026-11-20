# Laboratory architecture

Baseline `ef5ba786802c0f208c1178f9b802db4fadf3a83a` already includes T1–T7.
The prior [audit](ARCHITECTURE_AUDIT.md) remains a historical source.

| Responsibility | Implementation |
|---|---|
| Causal data, features, targets | `src/features`, `src/datasets`, `src/targets` |
| Campaign specification and bounded search | `src/experiments/spec.py`, `estimate.py`, `campaigns.py` |
| Lifecycle and worker | `api/routes_experiments.py`, `experiments/worker.py`, `runner.py` |
| Model registration | `ai/mlp/checkpoint.py`, `experiments/models.py` |
| Statistical evaluation | `src/evaluation` |
| Actual replay | `experiments/replay.py` → existing `eval/execution.py` → broker/portfolio/risk |
| Holdout boundary and exports | `experiments/holdout.py`, `reporting.py` |
| UI | `app/laboratory/page.tsx`, LaboratoryTraining and LaboratoryResults components |

SQLite remains the simulation ledger; campaign metadata and checkpoints use
the existing atomic filesystem artifacts. No new distributed infrastructure,
database migrations, live execution adapter or external service was introduced.
Old study and imported-report APIs remain compatible and independently usable.
