# Recorded development results — synthetic controls

Executed two local NumPy campaigns, five seeds `0..4`, small MLP, permanent
64-return profile, eight maximum epochs; scored only inner-validation
(1,098 decisions across two synthetic assets, 549 dates). Neither campaign
read protected test data. The positive control deliberately has strong
autocorrelation; the negative control is a drift/noise random walk.

| Control | Ensemble accuracy | ROC AUC | Interpretation |
|---|---:|---:|---|
| Positive signal | 78.87% | 0.86947 | Model detects a deliberately strong learnable feature |
| Noise | 51.73% | 0.52280 | No measurable superiority over the frequency baseline |

The positive-control ensemble lost to logistic regression on log loss; the
paired 95% date-block interval for model minus logistic loss was approximately
[+0.00674, +0.03111]. The noise control's difference versus class frequency
was approximately +0.00424, interval [−0.00401, +0.01237]. These are
development diagnostics, not evidence of financial alpha. Real-data
historical results from the prior protocol are in [training_audit.md](../training_audit.md).

Reproduce with the positive and negative recipes (`--epochs 8`). Report
configuration, seeds, model/baseline loss and uncertainty, not accuracy alone.
No new downloaded-market research campaign or genuine final benchmark was
executed in this implementation.
