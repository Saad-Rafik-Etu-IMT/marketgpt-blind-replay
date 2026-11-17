F5 player interface ([player.py](player.py)): a player maps a blind observation to a
binary long/flat target with optional long/flat probabilities. Rule-based baseline
players (F6 / MO1) are in [baselines.py](baselines.py): buy-and-hold and an MA 20/50
crossover on a relative price path, both without probabilities. The product MLP,
MarketGPT and explanation code are planned and not implemented here.
See [benchmark](../../../docs/benchmark.md#stage-2-players-and-execution-f5).
