# A Distribution-Free Framework for Rewrite-Based Human-Text Detection via Knockoff Filtering

This repository contains the code for the paper:

> **A Distribution-Free Framework for Rewrite-Based Human-text Detection via Knockoff Filtering**  

## Overview

We propose a statistical calibration framework that converts arbitrary rewrite-based LLM-text detectors into detectors with **finite-sample false discovery rate (FDR) guarantees**, without retraining. Our key insight is that rewrite-based detection implicitly constructs knockoff samples, enabling LLM-generated text detection to be formulated as a multiple hypothesis testing problem.

The framework:
1. Generates a knockoff by rewriting each text with an LLM
2. Computes a signed comparison statistic between the original and the knockoff
3. Applies a knockoff filter to select texts classified as human-written while controlling FDR at a user-chosen level `q`

We evaluate on **three detection methods** (L2D, ImBD, Likelihood), **19 domains**, and **four source LLMs** (GPT-3.5-Turbo, GPT-4o, Gemini-1.5-Pro, Llama-3-70B).

Some of the code is adopted from Zhou et al. (2026) using his implemetnation of L2D and ImBD.

## Repository Structure

```
.
├── scripts/
│   ├── detect_l2d.py           # L2D detection and rewrite generation
│   ├── detect_ImBD.py          # ImBD detection
│   ├── detect_likelihood.py    # Likelihood-based detection
│   ├── detect_knockoff.py      # Apply knockoff filter to pre-computed scores
│   ├── knockoff_filter.py      # Core FDR control algorithm (threshold + selection)
│   ├── rewrite_machine.py      # LLM rewrite sampler (gemma-9b-instruct)
│   ├── model.py                # Model loading utilities
│   ├── metrics.py              # ROC/PR curve computation
│   ├── helper.py               # BSpline basis functions
│   ├── utils.py                # Data loading utilities
│   ├── AdaDist/                # L2D scoring model (AUC fine-tuned on gemma-9b)
│   │   ├── model.py
│   │   ├── model_fast.py
│   │   ├── engine.py
│   │   └── dataset.py
│   └── ImBD/                   # ImBD scoring model (SPO fine-tuned on gemma-9b)
│       ├── spo.py
│       ├── engine.py
│       ├── dataset.py
│       └── utils_spo.py
├── exp_diverse/
│   ├── data/                   # Raw text data (human + AI, 19 domains × 4 models)
│   └── results/                # Detection outputs and knockoff statistics
├── exp_diverse.sh              # Run ImBD and L2D detection across all domains/models
├── exp_knockoff.sh             # Apply knockoff filter (L2D, ImBD, Likelihood)
├── summarize_knockoff.py       # Generate Tables 1 & 2 from the paper
├── cross_domain_null_mean_avg2.py      # Cross-domain null mean transfer analysis
├── cross_domain_null_mean_overall.py   # Aggregate cross-domain results by model
├── plot_cross_domain_null_mean_avg2.py # Generate Figure 3 plots
├── setup.sh                    # Environment setup (NLTK data, model caching)
├── DockerFile                  # CUDA 12.6 + Python 3.11 container
└── docker-compose.yml          # Docker Compose with GPU support
```

## Setup

### Option 1: Docker (recommended)

```bash
# Build and start the container (mounts current directory to /workspace)
docker compose up -d
docker compose exec knockoff bash
```

The container is based on `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` with PyTorch 2.7.0 and all dependencies pre-installed. Set `HF_TOKEN` in your environment if you need access to gated Hugging Face models.

### Option 2: Local environment

```bash
pip install -r requirements.txt

# Download NLTK data and cache Gemma models
bash setup.sh

# Add to your shell profile:
export NLTK_DATA="$HOME/nltk_data"
```

**Key dependencies:** PyTorch, Transformers, PEFT (LoRA), scipy, scikit-learn, matplotlib.

### Data

The dataset used in the experiments is from [Hao et al., 2025](https://aclanthology.org/2025.acl-long.343/) — 19 domains with 200 human-written and 200 LLM-generated texts per domain, covering GPT-3.5-Turbo, GPT-4o, Gemini-1.5-Pro, and Llama-3-70B-Instruct. Place the data files under `exp_diverse/data/` in the format:

```
exp_diverse/data/{Domain}_{Model}.raw_data.json
```

Each file is a JSON with:
```json
{
  "original": ["human text 1", "human text 2", ...],
  "sampled":  ["ai text 1",    "ai text 2",    ...]
}
```

## Running Experiments

### Step 1 — Generate rewrites and compute detection scores

```bash
bash exp_diverse.sh
```

This runs ImBD and L2D detection across all 19 domains and 4 source models, saving `.imbd.json` and `.l2d.json` files to `exp_diverse/results/`. The ImBD model is loaded from a checkpoint trained on 500 samples with SPO (β=0.05, lr=1e-4). The L2D model is loaded from `mamba413/L2D` on Hugging Face. Both use `gemma-9b-instruct` as the rewrite backbone (top-p=0.96, temperature=0.7, K=4 rewrites per text).

### Step 2 — Apply the knockoff filter

```bash
bash exp_knockoff.sh
```

This applies the knockoff filter to all three detection methods at target FDR levels `q ∈ {0.05, 0.1, 0.2, 0.3, 0.5}`:

| Method | Signed statistic | Requires |
|--------|-----------------|----------|
| L2D | `f(T_i, R_i)` from AdaDist model | `.l2d.json` |
| ImBD | `g(T_i) − g(R_i)` from SPO model | `.imbd.json` + rewrite file |
| Likelihood | `g(R_i) − g(T_i)` under gemma-1b | raw data + rewrite file |

Output files per `{Domain}_{Model}`:
- `.knockoff_l2d.json` — FDR/power at each q level
- `.knockoff_imbd.json` — FDR/power at each q level
- `.knockoff_likelihood.json` — FDR/power at each q level

### Step 3 — Summarize results (Tables 1 & 2)

```bash
python summarize_knockoff.py
```

Prints Table 1 (symmetry diagnostics: frac+, KS p-value) and Table 2 (empirical FDR and detection power), both in plain text and LaTeX.

### Step 4 — Cross-domain null mean transfer

```bash
python cross_domain_null_mean_avg2.py   # per-domain results, averaged over source domains
python cross_domain_null_mean_overall.py # aggregate by model
python plot_cross_domain_null_mean_avg2.py  # Figure 3 plots
```

This evaluates the cross-domain transfer setting: for each target domain, the null mean is borrowed from the 18 other domains, improving symmetry and power when target-domain calibration data are limited.

## Method Details

### Knockoff Filter

For a collection of signed statistics `{s_i}`, the knockoff threshold is:

```
τ = min { c > 0 : (#{s_i ≤ −c} + 1) / max(#{s_i ≥ c}, 1) ≤ q }
```

Texts with `s_i > τ` are classified as human-written. Under the symmetry assumption (Assumption 1 in the paper), this procedure controls FDR ≤ q in finite samples (Theorem 1).

The core implementation is in [scripts/knockoff_filter.py](scripts/knockoff_filter.py).

### Symmetry Assumption

The FDR guarantee requires that for AI-generated text, `s_i` is approximately symmetric around zero. We assess this empirically using:
- **frac+**: fraction of null (AI-text) statistics with `s_i > 0` (should be ≈ 0.5)
- **KS p-value**: Kolmogorov–Smirnov test for symmetry around zero

When symmetry does not hold (e.g., for L2D without centering), we apply a **null mean correction**: subtract the empirical mean of AI-text scores before thresholding. The cross-domain setting borrows this mean from held-out domains.

## Results Summary

Cross-domain null mean transfer (averaged across 19 domains):

| Method | GPT-3.5T Power | GPT-4o Power | Gemini Power | Llama-70B Power |
|--------|---------------|--------------|-------------|----------------|
| L2D | 0.833 | 0.825 | 0.941 | 0.901 |
| ImBD | 0.682 | 0.649 | 0.903 | 0.790 |
| Likelihood | 0.220 | 0.284 | 0.440 | 0.471 |

At target FDR level q = 0.2. FDR for L2D and ImBD ranges from 0.16–0.28 across models (close to nominal). Likelihood achieves meaningful power with near-zero FDR inflation.

## References

- Barber and Candès (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*.
- Candès et al. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *JRSS-B*.
- Chen et al. (2025). Imitate before detect. *AAAI 2025*. (ImBD)
- Zhou et al. (2026). Learn-to-distance. *ICLR 2026*. (L2D)
- Hao et al. (2025). Learning to rewrite. *ACL 2025*. (Dataset)
