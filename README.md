# experiments-in-language-model-architecture-alternatives

Small, self-contained experiments with **non-Transformer sequence architectures**. The shared idea (called *KRE-Form1* here) replaces self-attention with two stages:

1. **Unit discovery** — a convolutional backbone segments the input into a small set of learned "units" (spans) instead of attending over every raw position.
2. **Scalar unit weighting** — an MLP assigns each unit a single scalar weight per query position; the model prediction is the weighted sum of units, projected to the output space.

Two instantiations are implemented: one on synthetic physiological time series, one on raw text.

---

## Experiment 1 — EKG forecasting (`EKG Time Series.py`)

**Task:** given a 1600-sample synthetic EKG window (16 beats), predict the next beat (100 samples).

**Architecture (`KREForm1`, 64,421 params):**
- `UnitDiscoveryCNN` — 4 Conv1d layers (kernels 7/5/5/5, final stride 25) mapping the window to 16 unit vectors of dim 64.
- `UnitWeightingHead` — MLP → scalar weight per unit; weighted sum (`einsum`) → linear head outputs the 100-sample forecast. The weights are returned for inspection, so you can see which beats the model leaned on.

**Data generator (fully synthetic, no downloads):**
- Beats composed from Gaussian P/Q/R/S/T waves with per-beat jitter.
- Anomaly injection: premature ventricular contractions, atrial-fibrillation runs, motion artefacts, post-PVC compensatory beats.
- Noise models: baseline wander, EMG muscle noise, 50 Hz powerline interference.
- `make_window_v3` exposes difficulty modes (`morphology`, `pvc_chain`, `af_run`, `unaligned`, `all`) for stress testing.

**Training:** AdamW, lr 3e-4, 3000 steps, batch 32, MSE loss, CUDA-if-available.

**Results** (logged in `series.txt`):

| Run | Final loss (MSE) |
| --- | --- |
| Main training | ~0.0059 (from 0.0818) |
| Stress: `pvc_chain` | 0.00000 |
| Stress: `af_run` | 0.00428 |
| Stress: `unaligned` | 0.00702 |
| Stress: `morphology` | 0.03865 |
| Stress: `all` | 0.04468 |

  
```bash
python "EKG Time Series.py"   # visualize -> train -> stress tests
```

Standalone helpers: `save_model()` / `load_model()` for checkpoint reuse.

---

## Experiment 2 — Language modelling without attention (`txt 1m.py`)

**Task:** next-token prediction over a small Wikipedia corpus, with segmentation learned end-to-end.

**Architecture (`KREForm1BPERelPos`, 3,569,418 params):**
- `SoftSegCNN` — token embeddings through 4 dilated Conv1d layers (dilations 1/2/4/8) producing per-position **boundary probabilities** `b` and **feature vectors** `f`.
- `soft_unit_formation` — differentiable segmentation: boundary probabilities define "stay" probabilities along the sequence; candidate spans (length 1–8, starts chosen by top-k lowest boundary scores) get log-probabilities, softmaxed into mixing coefficients `alpha`. Units are span-summed feature vectors with expected end positions — everything is a smooth function of `b`, so gradients flow through the segmentation itself. A boundary-rate statistic is returned for regularization.
- `RelativePositionWeightMLP` — for each prediction position, encodes unit position and relative distance (unit → query) separately, concatenates with the unit vector, and regresses a scalar weight.
- Prediction = weighted sum of units → linear vocab head. A boundary-rate regularizer (`boundary_reg=0.5`) pulls the average boundary rate toward a target of 0.2, i.e. it biases the model toward forming units a few tokens long rather than degenerating to one token per unit or one giant unit.

**Data:**
- `wikids.txt` — ~3.98M characters of cleaned wikitext fetched from the Wikipedia API (15 classical-music articles), markup/ref/heading-stripped by `clean_wikitext()`.
- `wikiDS_bpe.json` — BPE tokenizer (HuggingFace `tokenizers` format, vocab 13,000); 894,281 tokens after encoding.

**Training:** 50,000 steps, seq_len 256, batch 8, lr 3e-4, cross-entropy + boundary regularizer, CUDA.

**Results** (logged in `text`, hardware: RTX PRO 4500):
- Cross-entropy: 57.5 (init) → **6.03** @ step 50,000
- Validation perplexity: 1915 @ step 500 → **414** @ step 50,000
- Boundary rate: 0.484 → **0.187** (target 0.2 — segmentation converges as intended)
- Samples remain incoherent at this scale (small corpus, 3.6M-param model) — the point of the run is that the segmentation + weighting mechanism trains stably end-to-end, not sample quality.

Outputs: checkpoint bundle (model + tokenizer + loss logs), `kre_bpe_relpos_weights.png` — a per-(unit, position) weight heatmap plus unit locations, for inspecting what the model attends to.

```bash
python "txt 1m.py"   # build corpus -> train 50k steps -> sample -> weight heatmap
```

---

## Files

| File | What it is |
| --- | --- |
| `EKG Time Series.py` | Experiment 1: synthetic EKG generator + KRE-Form1 training + stress tests |
| `txt 1m.py` | Experiment 2: corpus build, BPE encoding, KRE-Form1-BPE-RelPos LM training + sampling + weight visualization |
| `series.txt` | Training log — EKG experiment (main run + stress modes) |
| `text` | Training log — language modelling run (loss, perplexity, boundary rate, samples) |
| `wikids.txt` | Cleaned Wikipedia corpus (~4M chars) |
| `wikiDS_bpe.json` | Trained BPE tokenizer (vocab 13,000, `tokenizers` v1.0 format) |

## Requirements

```bash
pip install torch numpy matplotlib tokenizers
```

Both scripts auto-select CUDA when available and fall back to CPU; each is a single self-contained file (no package layout, no config files).

## Status / honest caveats

- Research scratchpad, not a library: file names have spaces, checkpoints and PNGs are written to the working directory, and defaults are baked into `__main__` blocks.
- Baselines (e.g. same-size Transformer or TCN) are **not** included yet — reported numbers are absolute, not relative.
- The EKG task is synthetic; the LM task is tiny. Both runs demonstrate that the unit-discovery + scalar-weighting mechanism learns, not that it beats attention.
