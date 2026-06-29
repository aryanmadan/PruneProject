"""
api.py — FastAPI backend for Prune duplicate task detection.

Endpoints:
  POST /check    — predict whether two tasks are duplicates
  POST /trello   — fetch all card titles from a Trello board
  POST /feedback — log a confirmed/dismissed label to data/feedback.csv

Loads on startup:
  data/best_model.pkl (joblib)
  all-MiniLM-L6-v2 (sentence-transformers)

Run:  uvicorn api:app --port 8000   (or: python api.py)
"""

import base64
import csv
import os

import numpy as np
import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from data_pipeline import clean_post
from feature_engineering import jaccard_similarity

MODEL_PATH = "data/best_model.pkl"
FEEDBACK_PATH = "data/feedback.csv"
EMBEDDER_NAME = "all-MiniLM-L6-v2"

app = FastAPI(title="Prune API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded on startup.
model = None
embedder = None


@app.on_event("startup")
def load_resources():
    global model, embedder
    model = joblib.load(MODEL_PATH)
    embedder = SentenceTransformer(EMBEDDER_NAME)


# --- Request models ---------------------------------------------------------

class CheckRequest(BaseModel):
    task_a: str
    task_b: str


class TrelloRequest(BaseModel):
    api_key: str
    token: str
    board_id: str


class JiraRequest(BaseModel):
    domain: str
    email: str
    api_token: str
    project_key: str


class AsanaRequest(BaseModel):
    access_token: str
    project_id: str


class FeedbackRequest(BaseModel):
    task_a: str
    task_b: str
    label: int


class ScanTask(BaseModel):
    id: str
    name: str


class ScanRequest(BaseModel):
    tasks: list[ScanTask]
    threshold: float = 0.75


# --- Feature computation (mirrors dashboard.py / feature_engineering.py) -----

def compute_features(text_a: str, text_b: str):
    clean_a = clean_post(text_a)
    clean_b = clean_post(text_b)

    emb_a, emb_b = embedder.encode([clean_a, clean_b], convert_to_numpy=True)

    cos_sim = float(
        np.dot(emb_a, emb_b)
        / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10)
    )

    len_a, len_b = len(clean_a), len(clean_b)
    max_len = max(len_a, len_b, 1)
    length_ratio = min(len_a, len_b) / max_len

    jaccard = jaccard_similarity(clean_a, clean_b)

    return np.array([[cos_sim, length_ratio, jaccard]]), cos_sim


# --- Endpoints --------------------------------------------------------------

SIMILARITY_FLOOR = 0.60


@app.post("/check")
def check(req: CheckRequest):
    features, cos_sim = compute_features(req.task_a, req.task_b)

    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    is_duplicate = prediction == 1
    confidence = float(proba[1] if is_duplicate else proba[0])

    # Similarity floor: a pair below the cosine floor can't be a duplicate, no
    # matter what the classifier says. Flag the override so the UI can explain
    # it. similarity/confidence stay as computed.
    overridden = is_duplicate and cos_sim < SIMILARITY_FLOOR
    if overridden:
        is_duplicate = False

    return {
        "similarity": cos_sim,
        "is_duplicate": is_duplicate,
        "confidence": confidence,
        "overridden": overridden,
    }


@app.post("/scan")
def scan(req: ScanRequest):
    """Pairwise duplicate scan over a list of tasks.

    Embeds every title once, computes all pairwise cosine similarities in a
    single vectorized step, then runs the classifier only on pairs above the
    cosine threshold. O(n) embeddings instead of O(n^2) /check round-trips.
    """
    tasks = req.tasks
    n = len(tasks)
    if n < 2:
        return {"tasks_scanned": n, "pairs": []}

    cleaned = [clean_post(t.name) for t in tasks]

    embeddings = embedder.encode(cleaned, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    unit = embeddings / norms

    # Full pairwise cosine matrix, keep upper triangle (i < j).
    sim_matrix = unit @ unit.T
    i_idx, j_idx = np.triu_indices(n, k=1)
    cos_sims = sim_matrix[i_idx, j_idx]

    keep = cos_sims > req.threshold
    i_keep = i_idx[keep]
    j_keep = j_idx[keep]
    cos_keep = cos_sims[keep]

    if len(i_keep) == 0:
        return {"tasks_scanned": n, "pairs": []}

    lengths = np.array([len(c) for c in cleaned], dtype=float)
    len_a = lengths[i_keep]
    len_b = lengths[j_keep]
    max_len = np.maximum(np.maximum(len_a, len_b), 1.0)
    length_ratio = np.minimum(len_a, len_b) / max_len

    jaccard = np.array(
        [jaccard_similarity(cleaned[i], cleaned[j]) for i, j in zip(i_keep, j_keep)],
        dtype=float,
    )

    features = np.column_stack([cos_keep, length_ratio, jaccard])
    preds = model.predict(features)
    probas = model.predict_proba(features)

    pairs = []
    for k in range(len(i_keep)):
        i, j = int(i_keep[k]), int(j_keep[k])
        is_dup = bool(preds[k] == 1)
        confidence = float(probas[k][1] if is_dup else probas[k][0])
        pairs.append({
            "id": f"{tasks[i].id}-{tasks[j].id}",
            "task_a": tasks[i].name,
            "task_b": tasks[j].name,
            "similarity": float(cos_keep[k]),
            "is_duplicate": is_dup,
            "confidence": confidence,
        })

    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return {"tasks_scanned": n, "pairs": pairs}


# Sample software-project tasks with intentional near-duplicates, used by the
# "Load Demo Board" feature so the Trello flow can be demoed without real creds.
DEMO_TASKS = [
    {"id": "1", "name": "Fix login button not responding"},
    {"id": "2", "name": "Login button is broken"},
    {"id": "3", "name": "Add dark mode to settings page"},
    {"id": "4", "name": "Implement dark mode in settings"},
    {"id": "5", "name": "Write unit tests for auth module"},
    {"id": "6", "name": "Set up CI/CD pipeline"},
    {"id": "7", "name": "Configure automated deployment pipeline"},
    {"id": "8", "name": "Fix broken image uploads"},
    {"id": "9", "name": "Image upload feature is not working"},
    {"id": "10", "name": "Update README with setup instructions"},
    {"id": "11", "name": "Improve onboarding documentation"},
    {"id": "12", "name": "Refactor database connection logic"},
    {"id": "13", "name": "Optimize database queries for performance"},
    {"id": "14", "name": "Add password reset flow"},
    {"id": "15", "name": "Implement forgot password functionality"},
]


@app.get("/demo")
def demo():
    return {"tasks": DEMO_TASKS}


@app.post("/trello")
def trello(req: TrelloRequest):
    auth = {"key": req.api_key, "token": req.token}
    cards_url = f"https://api.trello.com/1/boards/{req.board_id}/cards"
    board_url = f"https://api.trello.com/1/boards/{req.board_id}"

    try:
        resp = requests.get(cards_url, params={**auth, "fields": "name"}, timeout=15)
        resp.raise_for_status()
        board_resp = requests.get(
            board_url, params={**auth, "fields": "name"}, timeout=15
        )
        board_resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Trello API error: {exc}")

    cards = resp.json()
    tasks = [{"id": c["id"], "name": c["name"]} for c in cards]
    board_name = board_resp.json().get("name", "")
    return {"tasks": tasks, "board_name": board_name}


@app.post("/jira")
def jira(req: JiraRequest):
    url = f"https://{req.domain}/rest/api/3/search"
    params = {"jql": f"project={req.project_key}", "maxResults": 100}
    token = base64.b64encode(f"{req.email}:{req.api_token}".encode()).decode()
    headers = {"Authorization": f"Basic {token}", "Accept": "application/json"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Jira API error: {exc}")

    issues = resp.json().get("issues", [])
    tasks = [
        {"id": i["key"], "name": i.get("fields", {}).get("summary", "")}
        for i in issues
    ]
    # Jira has no board name; the project key is the natural label.
    return {"tasks": tasks, "board_name": req.project_key}


@app.post("/asana")
def asana(req: AsanaRequest):
    base = "https://app.asana.com/api/1.0"
    headers = {"Authorization": f"Bearer {req.access_token}"}
    tasks_url = f"{base}/projects/{req.project_id}/tasks"
    project_url = f"{base}/projects/{req.project_id}"

    try:
        resp = requests.get(
            tasks_url, params={"opt_fields": "name,gid"}, headers=headers, timeout=15
        )
        resp.raise_for_status()
        project_resp = requests.get(
            project_url, params={"opt_fields": "name"}, headers=headers, timeout=15
        )
        project_resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Asana API error: {exc}")

    data = resp.json().get("data", [])
    tasks = [{"id": t["gid"], "name": t.get("name", "")} for t in data]
    board_name = project_resp.json().get("data", {}).get("name", req.project_id)
    return {"tasks": tasks, "board_name": board_name}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_PATH)

    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["task_a", "task_b", "label"])
        writer.writerow([req.task_a, req.task_b, req.label])

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000)
