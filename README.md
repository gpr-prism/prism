# PRISM (Parallel Residual Iterative Sequence Model)

This repository provides the codebase for **PRISM (Parallel Residual Iterative Sequence Model)**. PRISM bridges the gap between efficient linear recurrences and high-fidelity iterative solvers via **Write-Forget Decoupling** and **Input-Anchored Loop Unrolling**. The implementation aligns with the paper and includes ablation variants for RQ3.

## Key Ideas
- **Write-Forget Decoupling**: keep forgetting linear and state-independent while allocating capacity to high-rank writing.
- **Input-Anchored Loop Unrolling**: use a short convolution to anchor residuals and a learned predictor to approximate multi-step refinement in parallel.
- **Rank Accumulation**: expand update rank beyond rank-1 within a single step.

## Repository Layout
- `models/prism.py`: PRISM implementation (uses `PRISMBlock`).
- `models/prism_ablate_l1.py`: w/o iterative refinement (L=1).
- `models/prism_ablate_no_nonlinear.py`: w/o solver non-linearity.
- `models/prism_ablate_no_shortconv.py`: w/o ShortConv anchor.
- `models/prism_ablate_no_gain.py`: w/o gain predictor (constant step size).
- `train_link_prediction.py`: link prediction training.
- `evaluate_node_retrieval.py`: retrieval evaluation (Hits@k).
- `train_edge_classification.py`: edge classification training.
- `evaluate_edge_classification.py`: edge classification evaluation.
- `utils/load_configs.py`: shared CLI configuration.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
Place datasets under:
```
DyLink_Datasets/<dataset_name>/
```
Expected files:
- `edge_list.csv`
- `entity_text.csv`
- `relation_text.csv`

Supported datasets in code:
```
Amazon_books, Amazon_elec, Amazon_movies,
Enron, GDELT, Googlemap_CT, ICEWS1819,
Stack_elec, Stack_english, Stack_ubuntu, Yelp
```

### Text Embeddings
If using text features (`--use_feature Bert`), generate embeddings:
```bash
python get_pretrained_embeddings.py
```
(Optional) Qwen embeddings:
```bash
python get_pretrained_embeddings_qwen.py
```

## Training and Evaluation
### Link Prediction (Train)
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

### Retrieval Evaluation (Hits@k)
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

### Edge Classification (Train)
```bash
python train_edge_classification.py \
  --dataset_name GDELT \
  --model_name prism \
  --num_layers 2 \
  --num_heads 2 \
  --channel_embedding_dim 64 \
  --num_neighbors 20 \
  --num_epochs 10 \
  --gpu 0 \
  --use_feature Bert
```

### Edge Classification (Eval)
```bash
python evaluate_edge_classification.py \
  --dataset_name GDELT \
  --model_name prism \
  --num_layers 2 \
  --num_heads 2 \
  --channel_embedding_dim 64 \
  --num_neighbors 20 \
  --gpu 0 \
  --use_feature Bert
```

## PRISM Ablations (RQ3)
Set `--model_name` to one of:
- `prism_ablate_l1`
- `prism_ablate_no_nonlinear`
- `prism_ablate_no_shortconv`
- `prism_ablate_no_gain`

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

## Notes
- PRISM uses `PRISMBlock` in `models/prism.py`.
- The solver step count is controlled via `--num_experts` (mapped to `solver_steps`).
- Checkpoints: `saved_models/<model>/<dataset>/...`
- Logs: `logs/<model>/<dataset>/...`

## Citation
If you use this code, please cite the PRISM paper (ICML 2026 submission).
