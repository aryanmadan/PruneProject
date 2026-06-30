"""
train_classifier.py — Train and evaluate duplicate-detection classifiers.

Reads:  data/features_train.parquet, data/features_dev.parquet, data/features_test.parquet
Writes: data/best_model.pkl

Trains Logistic Regression and Random Forest on the train split, evaluates both
on dev (optimising for recall), then evaluates the winner on the test split and
saves it with joblib.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import joblib

FEATURE_COLS = ["cosine_similarity", "length_ratio", "jaccard_similarity"]
LABEL_COL = "label_binary"


def load_split(path: str):
    df = pd.read_parquet(path)
    return df[FEATURE_COLS], df[LABEL_COL]


def evaluate(name: str, model, X, y):
    preds = model.predict(X)
    print(f"\n{'=' * 40}")
    print(f"{name}")
    print(f"{'=' * 40}")
    print(classification_report(y, preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, preds))
    return recall_score(y, preds)


if __name__ == "__main__":
    X_train, y_train = load_split("data/features_train.parquet")
    X_dev, y_dev = load_split("data/features_dev.parquet")
    X_test, y_test = load_split("data/features_test.parquet")

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    dev_recalls = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        dev_recalls[name] = evaluate(f"{name} — Dev", model, X_dev, y_dev)

    best_name = max(dev_recalls, key=dev_recalls.get)
    print(f"\nBest model by recall: {best_name} (recall={dev_recalls[best_name]:.4f})")

    best_model = models[best_name]
    evaluate(f"{best_name} — Test", best_model, X_test, y_test)

    joblib.dump(best_model, "data/best_model.pkl")
    print(f"\nSaved best model to data/best_model.pkl")
