import os
import argparse
import random
import numpy as np
import torch
import pyarrow.parquet as pq
from tqdm import tqdm
from res_kmeans import ResKmeans


def read_train_data(path, max_data_size, emb_dim):
    """Read training data from local parquet files"""
    dataset = pq.ParquetDataset(path)

    fragments = list(dataset.fragments)
    random.shuffle(fragments)
    print(f"Total files: {len(fragments)}")

    embeddings = []
    current_size = 0

    for fragment in tqdm(fragments, desc="Reading files"):
        table = fragment.to_table(columns=['embedding'])
        if table.num_rows == 0:
            continue

        emb_chunk = table['embedding'].to_numpy(zero_copy_only=False)
        if emb_chunk.dtype == 'object':
            emb_chunk = np.vstack(emb_chunk)

        emb_chunk = emb_chunk[:, :emb_dim].astype(np.float32)
        embeddings.append(emb_chunk)
        current_size += len(emb_chunk)

        if current_size >= max_data_size:
            print(f"Reached max_data_size: {current_size}")
            break

    result = np.concatenate(embeddings, axis=0)[:max_data_size]
    print(f"Final shape: {result.shape}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Train ResKmeans')
    parser.add_argument('--data_path', type=str, required=True, help='training data path')
    parser.add_argument('--model_path', type=str, required=True, help='model save path')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--codebook_size', type=int, default=8192, help='codebook size')
    parser.add_argument('--dim', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--niter', type=int, default=20, help='kmeans iterations')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    embeddings = read_train_data(args.data_path, args.max_data_size, args.dim)

    # Create and train model
    model = ResKmeans(
        n_layers=args.n_layers,
        codebook_size=args.codebook_size,
        dim=args.dim,
    )
    model.train_kmeans(torch.tensor(embeddings))

    # Save model
    os.makedirs(args.model_path, exist_ok=True)
    save_path = os.path.join(args.model_path, "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
