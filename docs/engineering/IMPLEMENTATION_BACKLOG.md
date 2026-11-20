# Implementation reconciliation

Statuses describe real code at delivery, not planned future functionality.
The baseline already included the T1–T7 MLP, data and evaluation foundations.

| Milestone | Priority | Status | Evidence / remaining work |
|---|---|---|---|
| M0 Scientific protocol | P0 | COMPLETED | Nested fold fit/selection, frozen data/split, corrected legacy wording, prospective plan |
| M1 Data and features | P0 | COMPLETED (core) | Main T1–T4; source previews, correlation and manifests. External context group H and variable window feature families remain deferred |
| M2 Models & baselines | P1 | COMPLETED (core) | Configurable MLP, registered reference Transformer/ablations, constant/logistic/ridge and depth-one tree, cost-aware head |
| M3 Persistent scheduler | P0 | COMPLETED (bounded local) | Waiting queue, cancellation, per-run resume, progress; exact epoch resume, parallel/multi-process scheduling and remote UI manager deferred |
| M4 Optimization | P1 | COMPLETED (bounded grid/random) | Stable candidate IDs and budget; successive halving and Bayesian optimization deferred |
| M5 Walk-forward | P0 | COMPLETED (expanding) | Nested fold selection refits target states/scalers; rolling and unseen-asset folds deferred |
| M6 Evaluation & registry | P1 | COMPLETED (core) | Model manifests, date-block intervals, baseline metrics, ROC/PR, calibration, ablation, costs; dedicated champion promotion workflow deferred |
| M7 Campaign automation | P1 | COMPLETED (12 recipes) | UI and CLI; automatic inner-validation report, configurable source sets; not every research idea is a turnkey recipe |
| M8 Replay | P1 | COMPLETED (binary model) | Real deterministic broker and risk/ledger; other predictive heads not automatically executable, risk-aware utility policies deferred |
| M9 Cockpit | P1 | COMPLETED | Six synchronized laboratory views, root redirect, legacy screens preserved; no destructive deletion |
| M10 Explainability | P2 | COMPLETED (measured core) | Feature shifts and recorded decision fact sheet/optional local LLM; legacy human evaluation retained, generalized rating workflow deferred |
| M11 Development campaign | P0 | COMPLETED (synthetic controls) | Five-seed positive/noise studies; broad real-market experiments still need local source data and time budget |
| M12 Final holdout | P0 | BLOCKED by protocol | Prospective lock and once-only predictive CLI implemented/tested on fixture; genuine future period and explicit human authorization required |
| M13 Report and defense | P0 | COMPLETED (exports/documents) | Markdown, CSV, JSON, existing SVG scientific plots; final research conclusions await real campaigns and holdout |

Next priorities: run a predeclared broad real-data development campaign; compare
baselines and model families on the same return profile; only retain additional
features supported by evidence. Freeze final evaluation after candidate
selection and before its prospective period starts. No production/live scope.
