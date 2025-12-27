# src/train_compare.py
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from utils import build_vectorizer, compute_metrics

MODELS = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVM": LinearSVC()
}

def main():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train_text = train_df["text"].values
    y_train = train_df["label"].values

    X_test_text = test_df["text"].values
    y_test = test_df["label"].values

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    mlflow.set_experiment("Sentiment-ML-Comparison")

    best_f1 = 0.0
    best_bundle = None

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)

            mlflow.log_params({
                "model": name,
                "vectorizer": "TF-IDF",
                "ngram": "1-2"
            })

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            import tempfile, joblib, os

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.joblib")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="model")

            print(f"{name} → F1: {metrics['f1']:.4f}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_bundle = {
                    "vectorizer": vectorizer,
                    "model": model,
                    "metrics": metrics,
                    "model_name": name
                }

    joblib.dump(best_bundle, "models/best_model.joblib")
    print(f"\nBest model: {best_bundle['model_name']} | F1={best_f1:.4f}")

if __name__ == "__main__":
    main()
