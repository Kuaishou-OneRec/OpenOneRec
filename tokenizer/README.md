# Residual K-Means Tokenizer

A residual K-means model for vector quantization. It encodes continuous embeddings into discrete codes through hierarchical clustering.

## Files

- `res_kmeans.py` - Model definition
- `train_res_kmeans.py` - Training script
- `infer_res_kmeans.py` - Inference script

## Installation

```bash
pip install torch numpy pandas pyarrow faiss tqdm
```

## Usage

### Training

```bash
python train_res_kmeans.py \
    --data_path ./data/embeddings.parquet \
    --model_path ./checkpoints \
    --n_layers 3 \
    --codebook_size 8192 \
    --dim 4096
```

**Arguments:**
- `--data_path`: Path to parquet file(s) with `embedding` column
- `--model_path`: Directory to save the model
- `--n_layers`: Number of residual layers (default: 3)
- `--codebook_size`: Size of each codebook (default: 8192)
- `--dim`: Embedding dimension (default: 4096)
- `--seed`: Random seed (default: 42)

### Inference

```bash
python infer_res_kmeans.py \
    --model_path ./checkpoints/model.pt \
    --emb_path ./data/embeddings.parquet \
    --output_path ./output/codes.parquet
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--emb_path`: Path to parquet file with `pid` and `embedding` columns
- `--output_path`: Output path (default: `{emb_path}_codes.parquet`)
- `--batch_size`: Inference batch size (default: 10000)
- `--device`: Device to use (default: cuda if available)
- `--n_layers`: Number of layers to use (default: all)

**Input format:** Parquet with columns `pid`, `embedding`

**Output format:** Parquet with columns `pid`, `codes`
