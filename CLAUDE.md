# Prune - Support Ticket Deduplication System

## Project Overview
Prune is an AI-powered duplicate detection system that automatically identifies duplicate and similar support tickets, transforming manual similarity analysis into an intelligent, automated classification system with a user-friendly dashboard and active learning capabilities.

## Current Status
Classifier trained and saved (data/best_model.pkl). Training/intermediate datasets and the synthetic-data generation script have been removed; the system now runs on the saved model with the active-learning loop collecting new labeled data.

## Project History & Key Decisions
- Originally built as a Trello task deduplication tool
- Trained classifier on Stack Overflow Duplicate Dataset (SODD) — caused domain mismatch
- SODD taught the model "duplicate = similar wording" but real tasks can be semantically equivalent with different vocabulary
- Decision: generate synthetic task pairs from SODD duplicate pairs using Gemini API
- Synthetic data will be in-domain (short task titles) and fix the mismatch

## ML Approach
- Embeddings: all-MiniLM-L6-v2 (sentence-transformers, 384-dimensional)
- Features: cosine similarity, length ratio, Jaccard similarity
- Classifier: Logistic Regression
- Training data: synthetically generated task pairs (source parquet since removed; trained model saved as data/best_model.pkl)
- Deployment: cosine similarity thresholding at 0.75 as baseline, classifier improves over time
- Active learning: confirm/dismiss loop collects real labeled data for future retraining

## File Structure
- data_pipeline.py — data loading and preprocessing
- embeddings.py — generates sentence embeddings
- feature_engineering.py — computes cosine similarity, length ratio, Jaccard
- train_classifier.py — trains Logistic Regression classifier
- dashboard.py — Streamlit dashboard for duplicate review

## Data
- data/best_model.pkl — trained Logistic Regression classifier (only remaining data artifact)
- Source datasets (SODD parquets, synthetic_tasks.parquet, cleaned/feature parquets, embeddings) have been deleted now that the model is trained

## Key Principles
- Recall is the priority metric — missing a duplicate is worse than a false alarm
- Confirm/dismiss loop is the ML story — system improves with real user feedback
- Synthetic data is legitimate and clearly documented as such

## Tech Stack
- Python, Streamlit, sentence-transformers, scikit-learn
- Gemini API (google-generativeai) for synthetic data generation
- Hardware: MacBook M3 8GB — sample to 100k rows max for memory

## Project Path
/Users/aryan/Documents/PruneProject/
