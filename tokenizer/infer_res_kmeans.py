import argparse
import torch
import numpy as np
import pandas as pd
from res_kmeans import ResKmeans


def load_embeddings(emb_path):
    """Load parquet file with pid and embedding columns"""
    df = pd.read_parquet(emb_path)
    pids = df['pid'].tolist()
    emb = torch.tensor(np.stack(df['embedding'].values), dtype=torch.float32)
    return pids, emb


def main():
    parser = argparse.ArgumentParser(description='ResKmeans Inference')
    parser.add_argument('--model_path', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--emb_path', type=str, required=True, help='embedding file path')
    parser.add_argument('--output_path', type=str, default=None, help='output path (default: emb_path + _codes.parquet)')
    parser.add_argument('--batch_size', type=int, default=10000, help='inference batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_layers', type=int, default=None, help='number of layers to use (default: all layers)')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    if isinstance(checkpoint, ResKmeans):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        # Restore from state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Infer model parameters
        n_layers = sum(1 for k in state_dict.keys() if k.startswith('centroids.'))
        first_centroid = state_dict['centroids.0']
        codebook_size, dim = first_centroid.shape

        model = ResKmeans(n_layers=n_layers, codebook_size=codebook_size, dim=dim)
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Unknown checkpoint format")

    model = model.to(args.device)
    model.eval()
    print(f"Model loaded: n_layers={model.n_layers}, codebook_size={model.codebook_size}, dim={model.dim}")

    # Load embeddings
    print(f"Loading embeddings from {args.emb_path}")
    pids, emb = load_embeddings(args.emb_path)
    print(f"Embeddings shape: {emb.shape}, num pids: {len(pids)}")

    # Inference
    print("Encoding...")
    all_codes = []
    with torch.no_grad():
        for i in range(0, len(emb), args.batch_size):
            batch = emb[i:i + args.batch_size].to(args.device)
            codes = model.encode(batch, n_layers=args.n_layers)
            all_codes.append(codes.cpu())
            if (i // args.batch_size) % 10 == 0:
                print(f"  Processed {min(i + args.batch_size, len(emb))}/{len(emb)}")

    all_codes = torch.cat(all_codes, dim=0)
    print(f"Output codes shape: {all_codes.shape}")

    # Save results to parquet
    output_path = args.output_path or args.emb_path.rsplit('.', 1)[0] + '_codes.parquet'
    df_out = pd.DataFrame({
        'pid': pids,
        'codes': all_codes.numpy().tolist()
    })
    df_out.to_parquet(output_path, index=False)
    print(f"Codes saved to {output_path}")

    # Compute reconstruction loss
    print("\nComputing reconstruction loss...")
    with torch.no_grad():
        sample_size = min(10000, len(emb))
        sample_emb = emb[:sample_size].to(args.device)
        sample_codes = all_codes[:sample_size].to(args.device)
        reconstructed = model.decode(sample_codes)
        loss_info = model.calc_loss(sample_emb, reconstructed)
        print(f"Reconstruction loss (MSE): {loss_info['loss']:.6f}")
        print(f"Relative loss: {loss_info['rel_loss']:.6f}")


if __name__ == '__main__':
    main()
