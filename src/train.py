from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_utils import load_data, make_train_test_split, split_features_target
from src.pipelines import build_classification_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_logreg_pipeline.joblib"

FINAL_THRESHOLD = 0.40


def evaluate_binary_classifier(
    y_true,
    y_pred,
    y_proba,
) -> dict[str, float]:
    """Calculate binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main() -> None:
    """Train final pipeline, evaluate it on test set and save model bundle."""
    print("Loading data...")
    df = load_data(DATA_PATH)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print("\nBuilding final pipeline...")

    final_pipeline = build_classification_pipeline(
        model=LogisticRegression(
            C=10,
            class_weight="balanced",
            l1_ratio=0,
            max_iter=2000,
            random_state=42,
        ),
        use_scaler=True,
        use_indicators=True,
    )

    print("Training final pipeline...")
    final_pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= FINAL_THRESHOLD).astype(int)

    test_metrics = evaluate_binary_classifier(
        y_true=y_test,
        y_pred=y_test_pred,
        y_proba=y_test_proba,
    )

    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

    print(f"\nFinal threshold: {FINAL_THRESHOLD}")

    print("\nConfusion matrix:")
    print(test_confusion_matrix)

    print("\nMetrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=["No diabetes", "Diabetes"],
            zero_division=0,
        )
    )

    model_bundle = {
        "pipeline": final_pipeline,
        "threshold": FINAL_THRESHOLD,
        "model_name": "tuned_balanced_logistic_regression",
        "best_params": {
            "C": 10,
            "class_weight": "balanced",
            "l1_ratio": 0,
        },
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_confusion_matrix.tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_bundle, MODEL_PATH)

    print(f"\nSaved model bundle to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
    