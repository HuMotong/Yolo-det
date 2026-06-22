from __future__ import annotations

from collections.abc import Iterable


def binary_metrics(
    truth: Iterable[str],
    predictions: Iterable[str],
) -> dict[str, float | int]:
    pairs = list(zip(truth, predictions, strict=True))
    tp = sum(
        actual == "positive" and predicted == "positive" for actual, predicted in pairs
    )
    tn = sum(
        actual == "negative" and predicted == "negative" for actual, predicted in pairs
    )
    fp = sum(
        actual == "negative" and predicted == "positive" for actual, predicted in pairs
    )
    fn = sum(
        actual == "positive" and predicted == "negative" for actual, predicted in pairs
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(pairs) if pairs else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
