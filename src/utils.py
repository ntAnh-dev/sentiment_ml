# src/utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True
    )

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    }