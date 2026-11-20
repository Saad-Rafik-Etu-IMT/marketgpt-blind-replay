# Saved validation studies

Start the normal local launcher and open **Research studies**. Choose the synthetic source or a CSV produced by the F1 pipeline in `backend/data/raw/yfinance/`. A study saves its configuration, checkpoints, results and worker log under the configured data directory, in `studies/<uuid>/`. The browser can be closed during training. Reopen a study from the list or bookmark its URL. The normal single-process FastAPI launcher accepts one study at a time.

On Windows, use `.\start.ps1 -Install -Research` once, then `.\start.ps1`.

The existing NumPy MLP runs with the backend dependencies. To enable the reference Transformer, install CPU PyTorch into the **same Python environment used by the launcher**. The tested Linux CPU runtime used torch 2.15.0.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Select **MLP+Transformer**, enter seeds such as `0,1,2`, and enable four ablations for a paired campaign. Every model sees the same adjusted-close windows. The encoder uses PyTorch's standard self-attention implementation, two heads, a 16-dimensional embedding, mean pooling and a binary output. The ablations remove positional embeddings, dropout, weight decay or the cosine schedule one at a time. Each run restores the lowest-validation-loss weights. The comparison estimates validation performance conditional on training and model selection. Final sealed test evaluation is a separate experiment.

## Historical validation example

For the repository's configured historical sources, run `python -m pip install -e ".[data]"` and `python -m src.data.download` from `backend/` in the project environment. Reopen **Research studies** and choose the downloaded ticker.

A local SPY verification run used 1,258 daily bars from 2015–2019, 439 training windows, seed 0 and two epochs. Both models completed broker replay on the same 627 validation decisions. The always-up baseline predicted 359 correctly.

| Predictor | Correct decisions | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| MLP | 324 / 627 | 51.7% | 0.505 |
| Reference Transformer | 300 / 627 | 47.8% | 0.475 |
| Always up | 359 / 627 | 57.3% | 0.364 |

Both models fell below the always-up baseline on accuracy in this run. Macro-F1 gives equal weight to the up and non-up classes, which explains how both models can score higher on macro-F1 while making fewer correct predictions overall. The short single-seed run exercises the workflow. The project’s generalization claim still requires a separately evaluated holdout.

## Inspect and demonstrate

### Recorded AI competition

Open **Replay → AI competition** or use **Watch AI competition** on a saved
study. Replay automatically opens the most recent completed trace-enabled study;
use **Change study** to select another study, seed or available Transformer variant.
For a four-player
comparison, train a fresh study with both MLP and Transformer (CPU PyTorch is
required). A one-model study still has the two rule-based strategies. Older saved
studies remain readable, but need to be rerun to record engine events.

The worker writes the engine's actual `DecisionRecord`, `ExecutionOutcome` and
`PortfolioSnapshotRead` for each session. The timeline contains the exact OHLC
bars inserted into the simulator, including its declared accounting convention.
The competition API joins the AI runs only when market data, profile, accounting,
decision windows and baseline results match. Each rule-based baseline is shown
once. No browser-side trading or retraining is needed during playback.

- The slider is the number of revealed next-session closes. At session 0 the
  initial snapshot and first available decision are visible. Advancing one session
  reveals the actual next-open fill (if any) and next-close portfolio snapshot.
- Play, pause, restart, seek and automatic pause on disagreement or fill use
  the same cursor for all players. Select a player and click an engine stage to
  inspect the corresponding fields. Inspect the previous revealed decision to
  explain the execution outcome. Prices and PnL remain hidden until their
  session is revealed in this UI.
- **Overview** shows market fills and portfolio equity, **Decision** contains
  the engine inspector and explanations, and **Performance** contains PnL,
  drawdown, exposure and probability plots. Playback stays at the same session
  when switching views; auto-pause switches are grouped under **Auto-pause**.
- The price plot uses simulated closes and exact execution prices (▲ buy, ▼
  sell). Equity, cumulative net change from initial equity, drawdown, market
  exposure and probability plots display only the revealed prefix. An AI's
  probability at the current close is available at decision time. The raw
  operator API serves the complete validation artifact; this viewer is for
  presentation, not an adversarial blind test. Use the isolated jury listener
  for a real vote-before-reveal challenge.
- **Manual simulation** still exposes the original interactive replay controls.
- New scientific figures have **SVG** download and expand controls. Expand a
  chart for a large presentation view and close it with Escape or Close. The files are scalable
  vector figures with a white background, axes, titles and series labels for
  reports or slides.

### Scientific figures

The Experiment tab pairs each Transformer variant with its own reference seed,
reports missing pairs, and plots the learning curves of the two reference models
with identical axes and the first minimum-validation-loss checkpoint marked.
The Evaluation tab computes five equal-width reliability bins from the currently
selected period, including probabilities of exactly 1 in the final bin. Exact
bin values and navigable cases remain available. Confusion counts are also shown
as a heatmap. Inspector compares the eight blocks using masked-minus-original
probabilities in percentage points. This is a measured intervention on the inputs.

The Data tab uses a single saved timeline with the study's own split configuration
and fingerprint. It displays returns and their distribution only through
validation; the test boundary is configuration metadata. The observed calendar
gap and the required embargo in bars are separately identified.

### Collecting explanation evaluation responses

In **Research studies → Explanations**, select the exact validation run used for
the human experiment. In Replay, the explanation format selector offers the same
facts as a table or as constrained LLM output. Before collecting responses, counterbalance the assignment
of distinct cases to fact-table and LLM explanation conditions. Use the same
underlying facts for both. Ask each participant to give the displayed rising
probability (percent, one decimal) and identify the eight-return block with the
largest absolute measured sensitivity (block 1 is oldest, block 8 newest). Record
preference after they have seen both formats. Use anonymous participant codes.

Import a JSON object shaped as `{"responses": [...]}`. Each response specifies
`participant`, `case` (one-based), `format` (`facts` or `llm`), `source` (actual
model or fact sheet), `fallback` (boolean), `probability_percent` (0–100), `block`
(1–8), and optional `preference` (`facts`, `llm`, `equal`). The import replaces
the selected run's previous response file. Scores use the saved prediction and
occlusion values, count the two questions equally, and accept tied strongest
blocks. The paired plot averages scores within each participant and assigned
format before calculating the mean paired difference. It reports incomplete
pairs and LLM fallbacks, and shows preference independently. No human scores
appear before actual responses are imported.

1. **Data** shows partition dates, sample counts, class balance, fitted scaling and quantile boundaries.
2. **Experiment** groups results by model and variant across seeds. Select an individual run to see its loss curves and model details.
3. **Evaluation** filters the confusion matrix by month and links errors to their exact input windows. Calibration and the paired bootstrap interval cover the complete validation period. The dataset comparison links completed runs across studies.
4. **Inspector** measures sensitivity by replacing eight returns with the training mean, rerunning the predictor, and recording the probability difference. This measures sensitivity to the specified intervention.
5. **Blind replay** draws a case on the server. A vote is committed before the outcome and broker records are returned. **Portfolio** displays exposure, costs, rejections, equity and maximum drawdown. For historical research studies only, a synthetic total-return instrument scales each day's OHLC by adjusted close / close, anchored to the initial close. This assumes reinvestment and synthetic index units. Simulated fills are index prices. The raw-price operator replay retains its own price convention.
6. **Presentation** opens a timed walkthrough and links recorded contributors to commits by area. Git history covers recorded changes only.

## Jury-only listener

For a jury browser that must be isolated from operator data, give it access only to a jury listener. Select a completed study UUID and its exact run key, for example `Transformer-reference-0`. In a separate backend process using the same data directory:

```bash
MRTL_JURY_ONLY=1 MRTL_JURY_STUDY=<uuid> MRTL_JURY_RUN=Transformer-reference-0 python -m uvicorn src.main:app --port 8001
```

Set `NEXT_PUBLIC_JURY_API_BASE_URL=http://localhost:8001` when starting or building the frontend, then open `/jury`. Set environment variables with PowerShell's `$env:NAME="value"` on Windows. This listener registers only the jury challenge and vote API. Repeating a vote returns the same outcome and changing a committed vote is rejected. The operator API and filesystem are trusted local surfaces. This deployment does not provide authentication or isolate a jury member who also has operator access.

## Local explanation generation

Set `MRTL_OLLAMA_MODEL` to an installed model name in a running local Ollama service on port 11434. Alternatively install `llama-cpp-python` and set `MRTL_GGUF_MODEL` to a compatible local model file. Environment settings belong to the backend. **Explain from measured facts** uses constrained verbalization of the measured prediction, position target and strongest occlusion. Choose Accessible or Scientific language; once execution is revealed, Replay also offers an outcome explanation grounded in the actual fill and equity change. Rule-based players have target explanations without fabricated probabilities. The generator orders the supplied facts and the result identifies its source. Failed validation or an unavailable runtime returns a labelled deterministic template. No model file or real market dataset is bundled.
