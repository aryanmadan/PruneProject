"""
dashboard.py — Streamlit app for live duplicate detection between two tasks.

Reads:  data/best_model.pkl
Uses:   all-MiniLM-L6-v2 (sentence-transformers), clean_post() from data_pipeline.py

Run:    streamlit run dashboard.py
"""

import numpy as np
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from data_pipeline import clean_post
from feature_engineering import jaccard_similarity


@st.cache_resource
def load_model():
    return joblib.load("data/best_model.pkl")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def compute_features(text_a: str, text_b: str):
    embedder = load_embedder()
    clean_a = clean_post(text_a)
    clean_b = clean_post(text_b)

    emb_a, emb_b = embedder.encode([clean_a, clean_b], convert_to_numpy=True)

    cos_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10))

    len_a, len_b = len(clean_a), len(clean_b)
    max_len = max(len_a, len_b, 1)
    length_ratio = min(len_a, len_b) / max_len

    jaccard = jaccard_similarity(clean_a, clean_b)

    return np.array([[cos_sim, length_ratio, jaccard]]), cos_sim


st.title("Duplicate Task Detector")

post_a = st.text_area("Task A", height=150)
post_b = st.text_area("Task B", height=150)

if st.button("Check for Duplicate"):
    if not post_a.strip() or not post_b.strip():
        st.warning("Please enter text for both tasks.")
    else:
        features, cos_sim = compute_features(post_a, post_b)
        model = load_model()

        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0]

        if prediction == 1:
            st.error(f"**Duplicate** (confidence: {confidence[1]:.1%})")
        else:
            st.success(f"**Not a duplicate** (confidence: {confidence[0]:.1%})")

        st.metric("Cosine Similarity", f"{cos_sim:.4f}")
