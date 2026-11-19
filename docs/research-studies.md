# Saved validation studies

Start the normal local launcher and open **Research studies**. Choose the synthetic source or a CSV produced by the F1 pipeline in `backend/data/raw/yfinance/`. A study saves its configuration, checkpoints, results and worker log under the configured data directory, in `studies/<uuid>/`. The browser can be closed during training. Reopen a study from the list or bookmark its URL. The normal single-process FastAPI launcher accepts one study at a time.

On Windows, use `.\start.ps1 -Install -Research` once, then `.\start.ps1`.

The existing NumPy MLP runs with the backend dependencies. To enable the reference Transformer, install CPU PyTorch into the **same Python environment used by the launcher**. The tested Linux CPU runtime used torch 2.15.0.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Select **MLP+Transformer**, enter seeds such as `0,1,2`, and enable four ablations for a paired campaign. Every model sees the same adjusted-close windows. The encoder uses PyTorch's standard self-attention implementation, two heads, a 16-dimensional embedding, mean pooling and a binary output. The ablations remove positional embeddings, dropout, weight decay or the cosine schedule one at a time. Each run restores the lowest-validation-loss weights. The comparison estimates validation performance conditional on training and model selection. Final sealed test evaluation is a separate experiment.

## Inspect and demonstrate

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

Set `MRTL_OLLAMA_MODEL` to an installed model name in a running local Ollama service on port 11434. Alternatively install `llama-cpp-python` and set `MRTL_GGUF_MODEL` to a compatible local model file. Environment settings belong to the backend. **Explain from measured facts** uses constrained verbalization of the predicted probability, position target and strongest measured occlusion. The result identifies its source. Failed validation or an unavailable runtime returns a labelled deterministic template. No model file or real market dataset is bundled.
