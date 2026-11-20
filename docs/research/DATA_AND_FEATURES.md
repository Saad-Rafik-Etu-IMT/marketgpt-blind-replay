# Data and features

The implemented feature registry, causal engine and feature families are
documented in [features.md](../features.md); [datasets.md](../datasets.md)
describes chronological selection, manifests and train-only scaling.

The local F1 source config lists 15 ETFs. Downloaded CSVs remain local, subject
to the approved educational data policy. The UI reports source coverage and
provenance status, previews development features, distributions, training-only
Spearman correlation of a representative scalar subset, feature catalogue and
saved partition counts. Full external market-context group H is not available.

New campaigns persist `data_lock.json`: per-asset canonical development frame
fingerprints and split boundaries. A resumed campaign or evaluator refuses
changed data; manifests also hash exact feature/label arrays. Protected-period
prices are excluded from feature previews, fitting, and development hashes.

Controls: `synthetic` is a seeded drift/noise random walk; `synthetic_signal`
has strong causal return autocorrelation, intended only to check learnability.
They cannot be mixed with downloaded assets.
