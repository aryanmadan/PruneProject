"""
feature_engineering.py — Compute cosine similarity, length ratio, and Jaccard features.

Reads:  data/train_clean.parquet, data/dev_clean.parquet, data/test_clean.parquet
        data/post_embeddings.npy, data/post_index.pkl
Writes: data/features_train.parquet, data/features_dev.parquet, data/features_test.parquet

Output columns: cosine_similarity, length_ratio, jaccard_similarity, label_binary

Run data_pipeline.py and embeddings.py first if their outputs don't exist yet.
"""

import numpy as np
import pandas as pd


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def load_embeddings(emb_path: str, index_path: str):
    embeddings = np.load(emb_path)
    post_index: pd.Series = pd.read_pickle(index_path)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    text_to_idx = {text: i for i, text in enumerate(post_index)}
    return embeddings_norm, text_to_idx


def compute_features(df: pd.DataFrame, embeddings_norm: np.ndarray, text_to_idx: dict) -> pd.DataFrame:
    idx1 = df['first_post_clean'].map(text_to_idx).to_numpy()
    idx2 = df['second_post_clean'].map(text_to_idx).to_numpy()

    emb1 = embeddings_norm[idx1]
    emb2 = embeddings_norm[idx2]
    cosine_sim = (emb1 * emb2).sum(axis=1)

    len1 = df['first_post_clean'].str.len().to_numpy()
    len2 = df['second_post_clean'].str.len().to_numpy()
    min_len = np.minimum(len1, len2).astype(float)
    max_len = np.maximum(len1, len2).astype(float)
    max_len[max_len == 0] = 1
    length_ratio = min_len / max_len

    jaccard = np.fromiter(
        (jaccard_similarity(a, b)
         for a, b in zip(df['first_post_clean'], df['second_post_clean'])),
        dtype=np.float32,
        count=len(df),
    )

    return pd.DataFrame({
        'cosine_similarity': cosine_sim.astype(np.float32),
        'length_ratio': length_ratio.astype(np.float32),
        'jaccard_similarity': jaccard,
        'label_binary': df['label_binary'].to_numpy(),
    })


if __name__ == '__main__':
    print("Loading embeddings and post index...")
    embeddings_norm, text_to_idx = load_embeddings(
        'data/post_embeddings.npy',
        'data/post_index.pkl',
    )
    print(f"  Embeddings shape: {embeddings_norm.shape}")
    print(f"  Unique posts indexed: {len(text_to_idx):,}")

    splits = [
        ('train', 'data/train_clean.parquet', 'data/features_train.parquet'),
        ('dev',   'data/dev_clean.parquet',   'data/features_dev.parquet'),
        ('test',  'data/test_clean.parquet',  'data/features_test.parquet'),
    ]

    for name, src, dst in splits:
        print(f"\nProcessing {name}...")
        df = pd.read_parquet(src)
        features = compute_features(df, embeddings_norm, text_to_idx)
        features.to_parquet(dst, index=False)

        pos = int(features['label_binary'].sum())
        total = len(features)
        print(f"  {total:,} rows saved → {dst}")
        print(f"  Labels: {pos:,} duplicate ({pos/total:.1%}) / {total-pos:,} not ({(total-pos)/total:.1%})")

    print("\nDone.")
