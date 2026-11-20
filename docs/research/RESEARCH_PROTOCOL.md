# Research protocol

- Use anonymous point-in-time numerical inputs. Asset/date metadata is used for
  chronology and reporting, not supplied as predictor inputs.
- Training fits all preprocessing and target state on inner-train. Inner
  validation retains checkpoints and may guide development search. Historical
  validation accesses are logged; repeated examination is development evidence.
- Confirmation builds expanding chronological outer folds within training.
  Each fold carves an earlier purged selection set, refits its scaler and target
  encoder, retains an epoch without reading the outer fold, then scores the
  outer fold. The outer fit does not inherit the globally retained epoch.
- Paired seeds keep configuration comparisons comparable. Market dates, not
  individual correlated asset observations, define bootstrap blocks.
- An uninterrupted final holdout requires a newly prospectively frozen period.
  The historical notebook examined the original sealed period; see
  [the training audit](../training_audit.md). No final test is read by campaigns.

Define the label horizon independently of next-close execution snapshots.
Use [protected holdout protocol](PROTECTED_HOLDOUT_PROTOCOL.md) before a final
evaluation. Cross-asset holdouts and rolling rather than expanding folds remain
future additions.
