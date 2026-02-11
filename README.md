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
- `evaluate_models_utils.py`: shared evaluation utilities for train/eval.
- `utils/load_configs.py`: shared CLI configuration.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
### 1. Download Raw Data
Please download the source data from the following links:

- **Amazon Review Data:**
  - Source: [SNAP Amazon Data](https://snap.stanford.edu/data/amazon/productGraph/)
  - Instructions: Download the specific category files. You will need both the **5-core** (reviews) and **metadata** files for your target category (e.g., *Books*, *Electronics*, *Movies and TV*).
- **Yelp Dataset:**
  - Source: [Yelp Data (Google Drive)](https://drive.google.com/drive/folders/1QFxHIjusLOFma30gF59_hcB19Ix3QZtk)

### 2. Preprocess Data
Before running the model, you need to format the raw data into the expected directory structure.

1. Open `dataset_preprocess.py` and modify the file paths to point to your downloaded source files (Amazon 5-core/metadata or Yelp data).
2. Run the preprocessing script:

```bash
python dataset_preprocess.py
```

This will generate the standard dataset files (`edge_list.csv`, `entity_text.csv`, `relation_text.csv`) in the following directory structure:

```text
DyLink_Datasets/<dataset_name>/
├── edge_list.csv
├── entity_text.csv
└── relation_text.csv
```

**Supported Datasets:** `Amazon_books`, `Amazon_elec`, `Amazon_movies`, `Yelp`

### 3. Feature Extraction
To obtain semantic features for the nodes, you need to generate pretrained embeddings.

- **Text Features (BERT):**
  Run the following script to get BERT-encoded feature vectors for the text in the dataset:
  ```bash
  python get_pretrained_embeddings.py
  ```

- **Multimodal Features (Optional):**
  If you wish to experiment with multimodal settings, you can use Qwen to generate embeddings:
  ```bash
  python get_pretrained_embeddings_qwen.py
  ```

## Synthetic Data
To generate and run experiments on synthetic datasets, simply execute the following script:

```bash
python test_syntheticdata.py
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
If you use this code, please cite the PRISM paper.
