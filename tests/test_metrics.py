# tests/test_metrics.py
from utils import compute_metrics

def test_compute_metrics_perfect():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0

def test_compute_metrics_worst():
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 1, 1]

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["f1"] == 0.0
