# PRISM Language Modeling

This directory contains the causal language modeling implementation of PRISM
and its LM baselines. It is organized as a companion implementation under the
top-level `prism/` repository:

```text
llm/
├── train.py
├── requirements.txt
├── models/
│   ├── common.py
│   ├── gdn.py
│   ├── efla.py
│   ├── pgdn.py
│   ├── prism.py
│   └── prism_ablations.py
├── scripts/
│   ├── run_main_comparison.sh
│   └── run_prism_ablations.sh
├── data/
│   └── README.md
└── evaluation/
    └── README.md
```

The paper's main released experiments are in `../rec/`. This directory keeps
the same PRISM design principles - Write-Forget Decoupling, an input-anchored
solver path, and Rank-L accumulation - but adapts them to causal LM training.

## Models

| Model | Description |
|-------|-------------|
| `gdn` | Gated Delta Net baseline |
| `efla` | Exponential Forgetting Linear Attention baseline |
| `pgdn` | Preconditioned GDN baseline |
| `prism` | PRISM full LM block |
| `prism_r2` | Shared solver key across steps |
| `prism_r3` | No retained residual allocation |
| `prism_r4` | Step 0 reuses the base key |
| `prism_r5` | All solver steps reuse the base key |

The `prism_r2` to `prism_r5` variants are LM companion ablations. They are not
the same names as the recommendation RQ3 ablations in `../rec/`.

## Setup

```bash
pip install -r requirements.txt
```

The default tokenizer is GPT-2 via `transformers.GPT2Tokenizer`. If
`ft_local/tokenizer_gpt2/` exists, `train.py` uses it instead.

## Data

`train.py` first looks for pre-tokenized Arrow files under `data/train`,
`data/validation`, and `data/test`. If they are missing, it falls back to the
public WikiText-103 raw dataset for a lightweight sanity run.

For the larger SlimPajama-style setup, prepare Arrow files as described in
`data/README.md`.

## Training

Single model:

```bash
python train.py --models prism --config large_130m --epochs 10
```

Multiple models:

```bash
python train.py --models gdn efla pgdn prism --config large_130m --epochs 10
```

Distributed:

```bash
torchrun --nproc_per_node=4 train.py \
  --models gdn efla pgdn prism \
  --config large_130m \
  --epochs 10 \
  --batch_size 8 \
  --grad_accum_steps 2
```

Scripted main comparison:

```bash
bash scripts/run_main_comparison.sh
```

PRISM LM ablations:

```bash
bash scripts/run_prism_ablations.sh
```

## Evaluation

Perplexity evaluation is built into `train.py`:

```bash
python train.py \
  --models prism \
  --config large_130m \
  --eval_only \
  --ckpt_dir checkpoints/prism
```

The checkpoint files written by `train.py` are raw PyTorch state dictionaries.
Downstream evaluation with `lm-evaluation-harness` requires exporting or wrapping
the model in a HuggingFace-compatible interface first. See
`evaluation/README.md` for details.

## Reference LM Results

The original language-modeling package included the following reference numbers
for a large-scale LM run. They are kept here as LM experiment context and should
not be confused with the recommendation tables in `../PRISM.pdf`.

| Model | Params | Wiki PPL | LMB PPL | LMB ACC | Avg ACC |
|-------|--------|----------|---------|---------|---------|
| GDN | 132.03M | 35.19 | 28.82 | 18.8 | 36.9 |
| EFLA | 132.03M | 35.51 | 28.50 | 18.6 | 38.1 |
| PGDN | 132.16M | 35.68 | 28.01 | 18.6 | 38.3 |
| PRISM | 138.06M | 34.68 | 27.00 | 19.8 | 40.1 |

## Citation

Use the citation in the repository root README.
