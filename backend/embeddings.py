"""
embeddings.py — Encode all unique cleaned posts with all-MiniLM-L6-v2 and save to disk.

Reads:  data/train_clean.parquet, data/dev_clean.parquet, data/test_clean.parquet
Writes: data/post_embeddings.npy  (shape: N x 384, float32)
        data/post_index.pkl       (pd.Series mapping integer position → post text)

Run data_pipeline.py first if the cleaned parquets don't exist yet.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    print("Loading cleaned splits...")
    splits = ['data/train_clean.parquet', 'data/dev_clean.parquet', 'data/test_clean.parquet']
    frames = [pd.read_parquet(p) for p in splits]

    all_posts = pd.concat([
        *(df['first_post_clean'] for df in frames),
        *(df['second_post_clean'] for df in frames),
    ]).unique().tolist()
    print(f"  Unique posts to embed: {len(all_posts):,}")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(
        all_posts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    np.save('data/post_embeddings.npy', embeddings)
    pd.Series(all_posts).to_pickle('data/post_index.pkl')

    print("Saved data/post_embeddings.npy")
    print("Saved data/post_index.pkl")
    print("\nDone.")
