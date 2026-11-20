# Validation report

Baseline main: backend 994 passed, 8 skipped (optional Torch not installed), 93.79% coverage; frontend 35 passed; type checking and lint clean. Baseline tree not modified in the original working copy.

Run from the frontend: `npm run typecheck`, `npm run lint`, `npm test`, `npm run build`, `npm run e2e`. Browser validation uses real worker-generated synthetic evidence, no mocked predictions. Run backend: `python -m pytest backend/tests` (root) and `ruff check backend`.

The targeted laboratory browser test passed including creation, convergence, evaluation, sensitivity, replay reveal, measured facts, reload, presentation and mobile width checks. Browser screenshots are produced as `laboratory.png`, `presentation.png` and `laboratory-mobile.png` under the Playwright test output and attached to successful pull-request CI. The final suite results are recorded in the PR.

Limitations: optional Transformer dependency unavailable in the baseline environment; no actual real-money execution, real-data acquisition, external LLM or GPU telemetry was tested. The dashboard is a front-end presentation of recorded development replay, not a security boundary preventing a user from inspecting locally saved development artifacts.
