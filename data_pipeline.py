"""
data_pipeline.py — Load raw SODD parquets, clean text, convert labels, save cleaned splits.

Reads:  data/SODD_train.parquet.gzip, data/SODD_dev.parquet.gzip, data/SODD_test.parquet.gzip
Writes: data/train_clean.parquet, data/dev_clean.parquet, data/test_clean.parquet

Columns in output: first_post_clean, second_post_clean, label_binary
"""

import re
import pandas as pd
from bs4 import BeautifulSoup


def remove_html_and_code(text: str) -> str:
    soup = BeautifulSoup(text, 'html.parser')
    for tag in soup.find_all(['code', 'pre']):
        tag.decompose()
    return soup.get_text()

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def clean_post(text: str) -> str:
    return normalize_text(remove_html_and_code(text))

def to_binary_label(label: int) -> int:
    return 1 if label in [0, 1] else 0


def load_and_clean(path: str, sample_n: int | None, drop_null_authors: bool) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=42)
    if drop_null_authors:
        df = df.dropna(subset=['first_author', 'second_author'])

    df = df[df['label'] != 4].copy()
    df['label_binary'] = df['label'].apply(to_binary_label)

    df['first_post_clean'] = df['first_post'].apply(clean_post)
    df['second_post_clean'] = df['second_post'].apply(clean_post)

    return df[['first_post_clean', 'second_post_clean', 'label_binary']]


if __name__ == '__main__':
    splits = [
        ('data/SODD_train.parquet.gzip', 'data/train_clean.parquet', 100_000, True),
        ('data/SODD_dev.parquet.gzip',   'data/dev_clean.parquet',   100_000, False),
        ('data/SODD_test.parquet.gzip',  'data/test_clean.parquet',  None,    False),
    ]

    for src, dst, sample_n, drop_nulls in splits:
        split = dst.split('/')[-1].replace('_clean.parquet', '')
        print(f"Processing {split}...")
        df = load_and_clean(src, sample_n, drop_nulls)
        df.to_parquet(dst, index=False)
        pos = df['label_binary'].sum()
        total = len(df)
        print(f"  {total:,} rows saved → {dst}")
        print(f"  Labels: {pos:,} duplicate ({pos/total:.1%}) / {total-pos:,} not ({(total-pos)/total:.1%})")

    print("\nDone.")
