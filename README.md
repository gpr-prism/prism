# PRISM

This repository contains the open-source code for **PRISM: Parallel Residual
Iterative Sequence Model**.

The code is organized into two self-contained tracks:

```text
prism/
├── PRISM.pdf                  # paper snapshot
├── rec/                       # recommendation experiments from the paper
└── llm/                       # causal language modeling implementation
```

## What Is Included

| Directory | Purpose | Main entry points |
|-----------|---------|-------------------|
| `rec/` | Sequential recommendation, retrieval, ablations, and synthetic probing used by the paper. | `train_link_prediction.py`, `evaluate_node_retrieval.py`, `test_syntheticdata.py` |
| `llm/` | A causal language modeling implementation of the same PRISM idea and LM baselines. | `train.py`, `scripts/run_main_comparison.sh`, `scripts/run_prism_ablations.sh` |

Large datasets, model checkpoints, tokenizer caches, and generated logs are not
bundled. The READMEs under `rec/` and `llm/` describe how to prepare the
corresponding data.

## Paper Alignment

The paper's main empirical results are the recommendation benchmarks and
mechanistic synthetic probing. Those map to `rec/`. The `llm/` directory is a
language modeling adaptation of PRISM that keeps the same core design principles
but uses LM-specific training code and companion ablations.

| Paper concept | `rec/` implementation | `llm/` implementation |
|---------------|-----------------------|-----------------------|
| Write-Forget Decoupling | `rec/models/prism.py`, `PRISMBlock._gated_linear_attention_with_injection` | `llm/models/prism.py`, `PRISMBlock._recurrence` |
| Input-Anchored Loop Unrolling | ShortConv anchor, gain predictor, residual loop, per-step key/value projections | ShortConv anchor plus fused low-rank solver projections |
| Rank-L accumulation | `step_k_proj`, `step_v_proj`, `step_delta`, `solver_steps` | solver step list with retained residual allocation |
| Paper RQ3 ablations | `prism_ablate_l1`, `prism_ablate_no_nonlinear`, `prism_ablate_no_shortconv`, `prism_ablate_no_gain`, `prism_hybrid4` | companion LM ablations `prism_r2` to `prism_r5` |
| Synthetic probing | `rec/test_syntheticdata.py` | not included |

This split avoids implying that the LM result tables are part of the
recommendation paper tables. Use `PRISM.pdf` as the reference for the paper
claims, and use `llm/README.md` for the language modeling setup.

## Quick Start

Recommendation track:

```bash
cd rec
pip install -r requirements.txt
python train_link_prediction.py --model_name prism --dataset_name Amazon_movies
```

Language modeling track:

```bash
cd llm
pip install -r requirements.txt
python train.py --models prism --config large_130m --epochs 1
```

The LM command falls back to WikiText-103 when no pre-tokenized data is present.
For the larger SlimPajama-style setup, prepare Arrow files under `llm/data/` as
described in `llm/data/README.md`.

## Citation

If you use this code, please cite:

```bibtex
@article{jiang2026prism,
  title   = {{PRISM}: Parallel Residual Iterative Sequence Model},
  author  = {Jie Jiang and Ke Cheng and Xin Xu and Mengyang Pang and Tianhao Lu
             and Jiaheng Li and Yue Liu and Yuan Wang and Jun Zhang
             and Huan Yu and Zhouchen Lin},
  journal = {arXiv preprint arXiv:2602.10796},
  year    = {2026}
}
```
