# PRISM Recommendation Experiments

This directory contains the recommendation-track code for **PRISM: Parallel
Residual Iterative Sequence Model**. It maps to the main empirical sections of
`../PRISM.pdf`: recommendation benchmarks, RQ3 ablations, and synthetic probing.

## Key Files

| File | Purpose |
|------|---------|
| `models/prism.py` | Full PRISM recommendation model |
| `models/prism_ablate_l1.py` | RQ3 ablation: single refinement step |
| `models/prism_ablate_no_nonlinear.py` | RQ3 ablation: remove solver non-linearity |
| `models/prism_ablate_no_shortconv.py` | RQ3 ablation: remove ShortConv anchor |
| `models/prism_ablate_no_gain.py` | RQ3 ablation: remove gain predictor |
| `models/prism_hybrid4.py` | Hybrid PRISM + MoM architecture |
| `train_link_prediction.py` | Link prediction training |
| `evaluate_node_retrieval.py` | Retrieval evaluation with Hits@K/NDCG/AUC |
| `test_syntheticdata.py` | Mechanistic synthetic probing |

The TTT and TTTv2 baselines are implemented directly in `models/TTT.py` and
`models/TTTv2.py`; the old vendored JAX TTT repository is intentionally not
included in this cleaned release tree.

## Setup

```bash
pip install -r requirements.txt
```

The `GDeltanet` baseline additionally requires the optional
`flash-linear-attention` package, which provides the `fla` module. It is not
installed by default because it depends on GPU/kernel build details.

## Data Layout

Run commands from this directory. Processed datasets are expected under:

```text
DyLink_Datasets/<dataset_name>/
├── edge_list.csv
├── entity_text.csv
├── relation_text.csv
├── e_feat.npy          # optional, required when --use_feature Bert
└── r_feat.npy          # optional, required when --use_feature Bert
```

Supported dataset names include `Amazon_books`, `Amazon_elec`,
`Amazon_movies`, and `Yelp`.

## Data Preparation

1. Download raw data.
   - Amazon Review Data: https://snap.stanford.edu/data/amazon/productGraph/
   - Yelp Open Dataset or the project-specific Yelp source used in your runs.

2. Edit the raw file paths at the bottom of `dataset_preprocess.py`.

3. Run:

```bash
python dataset_preprocess.py
```

4. Optional text features:

```bash
python get_pretrained_embeddings.py
```

`get_pretrained_embeddings.py` expects a local encoder under `llm_model/` by
default. Either place a compatible HuggingFace model there or edit
`pretrained_model_name` in the script.

## Train PRISM

```bash
python train_link_prediction.py \
  --dataset_name Amazon_movies \
  --model_name prism \
  --num_layers 2 \
  --num_heads 2 \
  --channel_embedding_dim 64 \
  --num_neighbors 20 \
  --num_epochs 10 \
  --gpu 0 \
  --use_feature Bert
```

## Retrieval Evaluation

```bash
python evaluate_node_retrieval.py \
  --dataset_name Amazon_movies \
  --model_name prism \
  --num_layers 2 \
  --num_heads 2 \
  --channel_embedding_dim 64 \
  --num_neighbors 20 \
  --gpu 0 \
  --use_feature Bert
```

## RQ3 Ablations

Set `--model_name` to one of:

```text
prism_ablate_l1
prism_ablate_no_nonlinear
prism_ablate_no_shortconv
prism_ablate_no_gain
prism_hybrid4
```

Example:

```bash
python train_link_prediction.py \
  --dataset_name Amazon_elec \
  --model_name prism_ablate_l1 \
  --num_layers 2 \
  --num_heads 2 \
  --channel_embedding_dim 64 \
  --num_neighbors 20 \
  --num_epochs 10 \
  --gpu 0 \
  --use_feature Bert
```

## Synthetic Probing

```bash
python test_syntheticdata.py
```

## Outputs

Generated checkpoints and logs are ignored by the root `.gitignore`:

```text
saved_models/<model>/<dataset>/...
logs/<model>/<dataset>/...
```

Use the citation in the repository root README.
