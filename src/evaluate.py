# src/evaluate.py
import joblib
import pandas as pd
from utils import compute_metrics

def main():
    df = pd.read_csv("data/test.csv")
    bundle = joblib.load("models/best_model.joblib")

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    X = vectorizer.transform(df["text"].values)
    y_true = df["label"].values
    y_pred = model.predict(X)

    metrics = compute_metrics(y_true, y_pred)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # CI/CD gate
    if metrics["f1"] < 0.7:
        raise ValueError("F1 below threshold!")

if __name__ == "__main__":
    main()