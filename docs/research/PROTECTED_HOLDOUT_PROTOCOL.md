# Protected prospective evaluation

The historical notebook examined dates inside the old sealed period. The
laboratory never treats those previously examined dates as a fresh final test.

After development, choose a completed real-data candidate and freeze an
explicit future start/end in Overview. The project-wide
`experiments/final_holdout.json` and campaign `holdout_plan.json` record the
candidate, checkpoint manifest hash, immutable spec and development data
fingerprints, exact code revision and prospective period. The start must be
after creation. A second lock is refused.

After the end date, with complete local data and explicit authorization:

```bash
python -m src.experiments.holdout data/experiments/<uuid> --authorize-id <frozen-id>
```

The CLI checks the exact plan and checkpoint. Before opening protected input
it atomically creates an exclusive consumed marker. It verifies development
preprocessing identity and scores the saved predictor and development-fitted
baselines, without training the candidate or promoting it. An attempt is
consumed even if later processing fails; no automatic retry. Results are
written locally to `holdout_result.json`; the HTTP interface only exposes
plan/consumed/result availability, not the protected scores.

This is procedural protection for a trusted local owner, not cryptographic
security against a filesystem administrator. The current final evaluator
is predictive, one registered candidate; a multi-candidate final broker
benchmark requires separate design and review before locking. **No real
holdout was frozen or evaluated during development of the laboratory.**
