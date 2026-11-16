# Research cockpit: first milestone

This milestone implements the shared shell and one thin, connected workflow:
Experiments → saved prediction in Inspector → explanation/facts → Evaluation
failure → original Inspector snapshot. It is not a training or evaluation system.
Replay keeps its original order, portfolio, risk and execution behavior.

## Data sources and lifetime

- Real mode is the default. With no imported bundle, research views say No Data.
- The Experiments empty state reads the existing instrument-summary API. This is
  operator metadata, not a training dataset or a player observation.
- "Load synthetic demo" is an explicit action. One deterministic, validated
  bundle supplies all three pages; a persistent banner labels it synthetic.
  Its loss curve was not produced by training. Its explanation is authored
  example text, not an LLM response.
- Import accepts JSON files up to 2 MiB. Parsing and storage happen only in
  browser memory. Nothing is uploaded; no filesystem path is accepted or followed.
  Navigation retains the bundle; reload removes it. Real mode clears it.
- A failed import clears old evidence and reports the error. Changing sources
  invalidates pending imports. There is no fallback to synthetic data.
- Import origin is an exporter declaration, not independently verified provenance.
  An imported bundle declaring `synthetic` keeps the synthetic banner.
- Query parameters `run`, `sample`, and `report` identify selections, not data.
  A recipient must load the same bundle. Invalid/cross-run selections show a
  warning rather than another sample's explanation.

## Proposed integration contract, version 1

This is **new frontend integration design**, not an existing backend/exporter
format. The [contract and runtime parser](../frontend/lib/research/contract.ts)
are authoritative. The [synthetic example](../frontend/lib/research/demo.ts)
is a complete construction example. Future exporters should have separate
adapters rather than imposing their native format on presentation components.

Envelope:

```typescript
{
  schemaVersion: 1;
  origin: "experimental" | "synthetic";
  runs: ResearchRun[];
  samples: ResearchSample[];
  reports: ResearchReport[];
}
```

| Record | Required fields |
| --- | --- |
| Run | `id`, `model`, `dataset`, `split`, `checkpoint`, `seed`, `lossName`, `curves` |
| Curve point | `step`; `trainingLoss`, `validationLoss` (null/omitted means unavailable) |
| Sample | `id`, `runId`, `decisionBar`, `predicted`, `probabilities`, `inputs`; optional/null `explanation` |
| Input | `bar`, `feature`, finite `value`, `unit` |
| Explanation | `text`, `template`, `modelVersion`, `promptVersion`, `facts`, `references`, `checks` |
| Fact | `id`, `bar`, `text` |
| Recorded check | `name`, boolean `passed`, `detail` |
| Report | `id`, `runId`, `split`, `cutoffBar`, `sampleCount`, optional/null `macroF1`, `failures` |
| Failure | `sampleId`, `observed`, `outcomeBar`, `detail` |

Validation:

- Only schema version 1 is accepted; unsupported versions explain which export
  is needed. Invalid JSON is rejected before record parsing.
- Maximum 20 runs, 500 samples, 20 reports; 2,000 curve points/run;
  512 inputs/sample; 64 facts/references/checks/explanation; 500 failures/report.
- IDs are unique within record collections, 1–100 ASCII letters, numbers,
  dots, underscores or hyphens. Other strings are nonempty and at most 4,000 characters.
- Numeric fields must be finite; bar/step/seed/count fields are safe integers.
  Curves have nonnegative losses and strictly increasing steps.
- Probability keys are `long`, `flat`, `short`, each in [0, 1], summing to 1
  within 0.000001. They are model probabilities, not probability of profit.
- Samples and reports reference existing runs. Failures reference existing
  samples from the report's run; their observed class differs from the prediction.
  Unique listed failures cannot outnumber evaluated samples.
- Inputs cannot exceed the decision bar. Facts must reference supplied visible
  input bars. Explanation references must name supplied fact IDs.
- Outcomes must follow the decision and not exceed the report cutoff.
- Known fields are reconstructed by the parser; extra fields are discarded.
  Content is rendered as React text, never injected HTML. These shape checks do
  not verify the truth of arbitrary explanation prose or metric calculations.

## Scientific boundaries

Training/validation curves are separate from saved retrospective evaluation.
The example failure uses a next-bar label with a ±0.1% flat band; its single
sample does not supply meaningful aggregate performance, so Macro-F1 is unavailable.
No metrics, labels, returns or attribution values are calculated in components.
Missing losses break the curve and appear as unavailable, not zero.

Recorded explanation checks are exporter-reported, not browser verification of
reasoning. No live generation, latency measurement, fallback service, human
assessment or occlusion attribution is implemented. The Inspector does not
invent target positions, risk decisions, orders or fills from class predictions.
Retrospective outcomes stay in Evaluation, outside saved decision-time inputs.

No test report is loaded. Historical test results, when supported later, must
carry timestamp/commit and must not be presented as current health.
Imported data is not supplied to Replay hooks, orders or player observations.
Replay remains non-anonymized, with its existing temporal boundaries.

## Validation commands

From `frontend`, with dependencies and Playwright Chromium already installed:

```powershell
npm run typecheck
npm run lint
npm test
$env:MRTL_NEXT_DIST_DIR = ".next-verify"
npm run build
Remove-Item Env:MRTL_NEXT_DIST_DIR
npm run e2e
```

`MRTL_NEXT_DIST_DIR` optionally isolates Next build output when another process
owns `.next`; default operation is unchanged. Playwright uses `.next-e2e`,
ports 3100/8100, its isolated synthetic database, and refuses existing servers.
Do not stop another user's service to free a port or build directory.

Research E2E captures all three pages at 1440px and 390px in both themes.
Screenshots are in ignored `frontend/test-results` subdirectories. Assertions
cover navigation, source separation, imports/reload, keyboard entry and overflow.
The original synthetic Replay E2E remains in the suite.

On the installed Node 20.7.0, Playwright's ESM loader fails before discovery
(`ERR_INVALID_ARG_TYPE`, `parentURL`). For this CommonJS-configured project,
`$env:PW_DISABLE_TS_ESM = "1"` before `npm run e2e` uses Playwright's existing
CommonJS TypeScript loader without changing dependencies or skipping tests.
A current Node 20 patch is recommended as in the main README.

## Deferred

Run comparison/filtering, comprehensive reproducibility drawers, F1 provenance
import, token inspection, attribution, live LLM integration, aggregate evaluation,
calibration, human assessments and financial research artifacts remain future
increments. Token editing, attention maps, assisted hints, confidence strips,
test-threshold tuning, training and job management are not included.
