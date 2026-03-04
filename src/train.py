# ============================================================
# src/train.py — TRAINING (Baseline: Logistic Regression)
# Goal: train model on train split, evaluate on val split, save artifacts to runs/
# NOTE: No threshold tuning here (that belongs in eval.py).
# ============================================================

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# --- import data pipeline ---
from src.data import prepare_data, DataConfig


# -----------------------------
# utilities
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def predict_proba_positive(model: LogisticRegression, X: Any) -> np.ndarray:
    """
    Returns P(y=1).
    Works for sklearn LogisticRegression.
    """
    # LogisticRegression supports predict_proba
    proba = model.predict_proba(X)[:, 1]
    return proba.astype(float)


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, Any]:
    y_pred = (y_prob >= thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": thr,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


# -----------------------------
# training
# -----------------------------
def train_logreg(
    X_train: Any,
    y_train: np.ndarray,
    class_weight: str | None,
    C: float,
    max_iter: int,
    random_state: int,
) -> LogisticRegression:
    """
    Baseline model:
    - Logistic Regression is strong, fast, interpretable
    - Uses L2 regularization by default
    """
    model = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_on_val(model: LogisticRegression, X_val: Any, y_val: np.ndarray) -> Dict[str, Any]:
    y_prob = predict_proba_positive(model, X_val)

    auc = roc_auc_score(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)  # PR-AUC

    # report also at default 0.5 threshold
    thr_report = metrics_at_threshold(y_val, y_prob, thr=0.50)

    return {
        "val": {
            "roc_auc": float(auc),
            "avg_precision_pr_auc": float(ap),
            "default_threshold_report": thr_report,
        }
    }


# -----------------------------
# main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline churn model (Logistic Regression).")
    parser.add_argument("--raw-path", type=str, default="data/raw/telco.csv", help="Path to raw telco CSV.")
    parser.add_argument("--run-dir", type=str, default="runs", help="Root runs directory.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits/model.")

    # model knobs
    parser.add_argument("--class-weight", type=str, default="balanced", choices=["balanced", "none"],
                        help="Use 'balanced' to handle class imbalance, or 'none' to disable.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression.")
    parser.add_argument("--max-iter", type=int, default=2000, help="Max iterations for LogisticRegression.")

    args = parser.parse_args()

    # 1) Prepare data (leakage-free)
    cfg = DataConfig(
        raw_path=args.raw_path,
        random_state=args.random_state,
        # keep defaults for ratios + columns unless you explicitly change later
    )

    bundle = prepare_data(cfg)

    X_train = bundle["X_train_mat"]
    y_train = np.asarray(bundle["y_train"], dtype=np.int64)

    X_val = bundle["X_val_mat"]
    y_val = np.asarray(bundle["y_val"], dtype=np.int64)

    preprocessor = bundle["preprocessor"]
    feature_names = bundle.get("feature_names", None)

    # 2) Train model
    class_weight = None if args.class_weight == "none" else "balanced"

    model = train_logreg(
        X_train=X_train,
        y_train=y_train,
        class_weight=class_weight,
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    # 3) Evaluate on validation only (do NOT touch test here)
    metrics = evaluate_on_val(model, X_val, y_val)

    # 4) Save artifacts
    run_id = now_run_id()
    run_path = Path(args.run_dir) / run_id
    ensure_dir(run_path)

    # Save model + preprocessor
    joblib.dump(model, run_path / "model.joblib")
    joblib.dump(preprocessor, run_path / "preprocess.joblib")

    # Save metadata + metrics
    save_json(run_path / "config.json", asdict(cfg) | {
        "train_script": "src/train.py",
        "model_type": "logreg",
        "logreg": {
            "class_weight": args.class_weight,
            "C": args.C,
            "max_iter": args.max_iter,
        },
    })
    save_json(run_path / "metrics.json", metrics)

    # Save feature names (optional but useful)
    if feature_names is not None:
        (run_path / "feature_names.txt").write_text("\n".join(map(str, feature_names)), encoding="utf-8")

    print(f"[OK] Saved run to: {run_path}")
    print(f"[VAL] ROC-AUC: {metrics['val']['roc_auc']:.4f}")
    print(f"[VAL] PR-AUC:  {metrics['val']['avg_precision_pr_auc']:.4f}")
    print(f"[VAL] Default thr=0.50 | "
          f"Precision={metrics['val']['default_threshold_report']['precision']:.4f} "
          f"Recall={metrics['val']['default_threshold_report']['recall']:.4f}")


if __name__ == "__main__":
    main()