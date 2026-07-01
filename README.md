# Prune

**Stop doing the same work twice.**

Prune is an ML-powered duplicate task detection system that automatically identifies similar and duplicate work items across project management platforms. It transforms manual backlog review into an intelligent, automated classification system.

## What it does

- Connects to Trello, Jira, or Asana boards and pulls all tasks
- Converts each task title into a semantic vector using `all-MiniLM-L6-v2` sentence embeddings
- Computes pairwise cosine similarity across all task combinations
- Runs a trained Logistic Regression classifier to flag duplicate pairs
- Presents flagged pairs with similarity scores and confidence levels
- Logs user feedback (Confirm/Dismiss) to an active learning loop that improves the model over time

## ML Pipeline

1. **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` converts task titles into 384-dimensional semantic vectors
2. **Features** — Three features per pair: cosine similarity, length ratio, Jaccard similarity
3. **Classifier** — Logistic Regression trained on the Stack Overflow Duplicity Dataset (SODD)
4. **Active Learning** — User feedback is logged to `feedback.csv` for future retraining

## Tech Stack

- **Backend** — FastAPI, sentence-transformers, scikit-learn
- **Frontend** — React + Vite, Space Grotesk font
- **Integrations** — Trello API, Jira REST API, Asana API

## Running locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```
