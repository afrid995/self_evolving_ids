"""
train_known.py -- Train the XGBoost known-attack classifier.

Usage:
    python src/train_known.py

What it does:
  1. Loads preprocessed training and validation data from data/processed/
  2. Computes sample weights to handle class imbalance
  3. Trains an XGBoost multi-class classifier (with RandomForest fallback)
  4. Evaluates on validation AND test sets
  5. Prints per-class precision/recall/F1, confusion matrix, macro F1
  6. Saves the trained model to models/known_classifier.pkl

Expected result:
  - Macro F1 > 0.85 on NSL-KDD validation set
  - If not, the script prints tuning suggestions
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from src.config import (
    PROCESSED_DIR, MODELS_DIR,
    KNOWN_MODEL_PATH, XGBOOST_PARAMS,
    RANDOM_STATE,
)
from src.evaluate import (
    print_classification_report,
    print_confusion_matrix,
    compute_summary_metrics,
)


# ============================================================
# 1. Load data
# ============================================================
def load_processed_data() -> dict:
    """
    Load all preprocessed Parquet files and metadata.

    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, label_classes
    """
    print("  Loading processed data...")

    data = {}
    for name in ["X_train", "X_val", "X_test",
                  "y_train", "y_val", "y_test"]:
        path = PROCESSED_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"  [ERROR] Missing: {path}")
            print("  Run 'python src/preprocess.py' first!")
            sys.exit(1)
        data[name] = pd.read_parquet(path)

    # Flatten y columns to 1D arrays
    for y_name in ["y_train", "y_val", "y_test"]:
        data[y_name] = data[y_name]["label"].values

    # Load metadata
    feat_path = PROCESSED_DIR / "feature_names.txt"
    data["feature_names"] = feat_path.read_text().strip().split("\n")

    cls_path = PROCESSED_DIR / "label_classes.txt"
    all_classes = cls_path.read_text().strip().split("\n")

    # The training data excludes the zero-day class (R2L), so the
    # integer labels may be non-contiguous (e.g. [0,1,2,4]).
    # XGBoost requires contiguous labels [0..n_classes-1].
    # Remap them here.
    unique_train = np.unique(data["y_train"])
    old_to_new = {old: new for new, old in enumerate(unique_train)}
    data["label_remap"] = old_to_new  # keep for reverse lookup

    for y_name in ["y_train", "y_val", "y_test"]:
        data[y_name] = np.array([old_to_new[v] for v in data[y_name]])

    # Filter class names to only the classes present in training
    data["label_classes"] = [all_classes[i] for i in unique_train]
    data["all_classes"] = all_classes  # keep full list for later phases

    print(f"    Train: {len(data['X_train']):,} samples")
    print(f"    Val:   {len(data['X_val']):,} samples")
    print(f"    Test:  {len(data['X_test']):,} samples")
    print(f"    Classes: {data['label_classes']}")

    return data


# ============================================================
# 2. Compute sample weights for class imbalance
# ============================================================
def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights inversely proportional to class frequency.

    Minority classes get higher weight so the classifier pays more
    attention to them.  This is the standard 'balanced' approach.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)

    # weight_for_class = n_samples / (n_classes * count_of_class)
    class_weights = {
        cls: n_samples / (n_classes * cnt)
        for cls, cnt in zip(classes, counts)
    }

    sample_weights = np.array([class_weights[label] for label in y])

    print("  Class weights (for imbalance handling):")
    for cls, cnt in zip(classes, counts):
        pct = 100 * cnt / n_samples
        print(f"    Class {cls}: {cnt:>7,} samples ({pct:5.1f}%) "
              f"-> weight {class_weights[cls]:.3f}")

    return sample_weights


# ============================================================
# 3. Train classifier
# ============================================================
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    n_classes: int,
) -> object:
    """
    Train an XGBoost multi-class classifier.

    Falls back to RandomForestClassifier if XGBoost import fails.

    Returns the trained model.
    """
    try:
        import xgboost as xgb
        print("\n  [OK] Using XGBoost classifier")

        params = XGBOOST_PARAMS.copy()
        params["num_class"] = n_classes

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            verbose=True,
        )
        return model

    except ImportError:
        print("\n  [WARNING] XGBoost not available. Using RandomForest fallback.")
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        return model


# ============================================================
# 4. Evaluate model
# ============================================================
def evaluate_model(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    split_name: str,
) -> dict[str, float]:
    """
    Evaluate the classifier on a given split and print results.

    Returns summary metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluation on {split_name} set")
    print(f"{'='*60}")

    y_pred = model.predict(X)

    # Classification report
    print_classification_report(y, y_pred, class_names)

    # Confusion matrix
    print_confusion_matrix(y, y_pred, class_names)

    # Summary metrics
    metrics = compute_summary_metrics(y, y_pred)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    return metrics


# ============================================================
# 5. Main training pipeline
# ============================================================
def train_known_classifier() -> object:
    """
    Full training pipeline for the known-attack classifier.

    Returns the trained model.
    """
    print("=" * 60)
    print("  Phase 2: Training Known-Attack Classifier")
    print("=" * 60)

    # --- Load data ---
    print("\n[Step 1/4] Loading data...")
    data = load_processed_data()

    # --- Compute sample weights ---
    print("\n[Step 2/4] Computing class weights...")
    sample_weights = compute_sample_weights(data["y_train"])

    # --- Train ---
    print("\n[Step 3/4] Training classifier...")
    start_time = time.time()

    model = train_xgboost(
        X_train=data["X_train"].values,
        y_train=data["y_train"],
        sample_weights=sample_weights,
        n_classes=len(data["label_classes"]),
    )

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f} seconds")

    # --- Evaluate ---
    print("\n[Step 4/4] Evaluating model...")

    # Validation set
    val_metrics = evaluate_model(
        model, data["X_val"].values, data["y_val"],
        data["label_classes"], "Validation",
    )

    # Test set
    test_metrics = evaluate_model(
        model, data["X_test"].values, data["y_test"],
        data["label_classes"], "Test",
    )

    # --- Check quality gate ---
    target_f1 = 0.85
    if val_metrics["macro_f1"] >= target_f1:
        print(f"\n  [PASS] Macro F1 ({val_metrics['macro_f1']:.4f}) "
              f">= {target_f1} target")
    else:
        print(f"\n  [BELOW TARGET] Macro F1 ({val_metrics['macro_f1']:.4f}) "
              f"< {target_f1} target")
        print("  Suggestions:")
        print("    - Increase n_estimators (e.g., 300-500)")
        print("    - Try max_depth=8 or 10")
        print("    - Increase learning_rate to 0.15")
        print("    - Check if preprocessing produced clean data")

    # --- Save model ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, KNOWN_MODEL_PATH)
    print(f"\n  Model saved to: {KNOWN_MODEL_PATH}")

    # --- Top features ---
    print("\n  Top 10 most important features:")
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_names = data["feature_names"]
            indices = np.argsort(importances)[::-1][:10]
            for rank, idx in enumerate(indices, 1):
                print(f"    {rank:2d}. {feat_names[idx]:30s} "
                      f"{importances[idx]:.4f}")
    except Exception as e:
        print(f"    Could not extract feature importances: {e}")

    print("\n" + "=" * 60)
    print("  Phase 2 complete!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train_known_classifier()
