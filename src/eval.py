# ============================================================
# src/eval.py — EVALUATION + THRESHOLD TUNING (Decisioning)
# Goal:
#   - Load trained run artifacts (model + preprocess)
#   - Tune threshold on VAL using cost-based objective
#   - Final evaluation on TEST using chosen threshold
# Outputs:
#   runs/<run_id>/
#     threshold_report.csv
#     best_threshold.json
#     test_report.json
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


from src.data import DataConfig, load_raw_data, clean_data, build_X_y, split_data


# -----------------------------
# utils
# -----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def predict_proba_positive(model: Any, X: Any) -> np.ndarray:
    """
    Returns P(y=1).
    Works for sklearn LogisticRegression and similar models with predict_proba.
    """
    proba = model.predict_proba(X)[:, 1]
    return np.asarray(proba, dtype=float)


def confusion_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    """
    Returns (tn, fp, fn, tp)
    """
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_from_probs(y_true, y_prob, thr)

    y_pred = (y_prob >= thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def cost_from_confusion(fp: int, fn: int, c_fp: float, c_fn: float) -> float:
    return float(fp * c_fp + fn * c_fn)


# -----------------------------
# core eval
# -----------------------------
def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fp: float,
    c_fn: float,
    t_min: float = 0.05,
    t_max: float = 0.95,
    t_step: float = 0.01,
) -> pd.DataFrame:
    rows = []
    thr = t_min
    while thr <= t_max + 1e-12:
        m = metrics_at_threshold(y_true, y_prob, thr)
        total_cost = cost_from_confusion(m["fp"], m["fn"], c_fp=c_fp, c_fn=c_fn)
        m["total_cost"] = total_cost
        rows.append(m)
        thr = round(thr + t_step, 10)

    df = pd.DataFrame(rows)
    # Sort by cost, then by f1 as tie-break
    df = df.sort_values(["total_cost", "f1"], ascending=[True, False]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate run + tune threshold (cost-based).")
    parser.add_argument("--run", type=str, required=True, help="Path to run folder, e.g. runs/2026-03-03_172421")
    parser.add_argument("--raw-path", type=str, default="data/raw/telco.csv", help="Path to raw telco CSV.")
    parser.add_argument("--c-fp", type=float, default=1.0, help="Cost of false positive.")
    parser.add_argument("--c-fn", type=float, default=5.0, help="Cost of false negative (usually higher).")

    # threshold sweep grid
    parser.add_argument("--t-min", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=0.95)
    parser.add_argument("--t-step", type=float, default=0.01)

    args = parser.parse_args()

    run_path = Path(args.run)
    assert run_path.exists(), f"Run path not found: {run_path}"

    model_path = run_path / "model.joblib"
    prep_path = run_path / "preprocess.joblib"
    cfg_path = run_path / "config.json"

    assert model_path.exists(), f"Missing model artifact: {model_path}"
    assert prep_path.exists(), f"Missing preprocess artifact: {prep_path}"
    assert cfg_path.exists(), f"Missing config artifact: {cfg_path}"

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    cfg_saved = load_json(cfg_path)

    # Recreate the SAME split using saved config ratios + seed
    cfg = DataConfig(
        raw_path=args.raw_path,
        random_state=int(cfg_saved.get("random_state", 42)),
        train_ratio=float(cfg_saved.get("train_ratio", 0.70)),
        val_ratio=float(cfg_saved.get("val_ratio", 0.15)),
        test_ratio=float(cfg_saved.get("test_ratio", 0.15)),
    )

    df_raw = load_raw_data(cfg.raw_path)
    df_clean = clean_data(df_raw)
    X_df, y = build_X_y(df_clean, cfg.target_col, cfg.id_col)

    splits = split_data(
        X_df,
        y,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        random_state=cfg.random_state,
    )

    # Transform using the SAVED preprocessor (DO NOT refit)
    X_val_mat = preprocessor.transform(splits["X_val"])
    y_val = np.asarray(splits["y_val"], dtype=np.int64)

    X_test_mat = preprocessor.transform(splits["X_test"])
    y_test = np.asarray(splits["y_test"], dtype=np.int64)

    # Probability scores
    val_prob = predict_proba_positive(model, X_val_mat)
    test_prob = predict_proba_positive(model, X_test_mat)

    # Score metrics independent of threshold
    val_auc = roc_auc_score(y_val, val_prob)
    val_pr_auc = average_precision_score(y_val, val_prob)

    # Threshold sweep on VAL
    report = threshold_sweep(
        y_true=y_val,
        y_prob=val_prob,
        c_fp=args.c_fp,
        c_fn=args.c_fn,
        t_min=args.t_min,
        t_max=args.t_max,
        t_step=args.t_step,
    )

    best = report.iloc[0].to_dict()
    best_thr = float(best["threshold"])

    # Final TEST evaluation using chosen threshold
    test_thr_metrics = metrics_at_threshold(y_test, test_prob, best_thr)
    test_auc = roc_auc_score(y_test, test_prob)
    test_pr_auc = average_precision_score(y_test, test_prob)
    test_cost = cost_from_confusion(test_thr_metrics["fp"], test_thr_metrics["fn"], c_fp=args.c_fp, c_fn=args.c_fn)

    # Save artifacts
    report.to_csv(run_path / "threshold_report.csv", index=False)

    save_json(
        run_path / "best_threshold.json",
        {
            "chosen_on": "val",
            "c_fp": float(args.c_fp),
            "c_fn": float(args.c_fn),
            "val_roc_auc": float(val_auc),
            "val_pr_auc": float(val_pr_auc),
            "best_row": best,
        },
    )

    save_json(
        run_path / "test_report.json",
        {
            "evaluated_on": "test",
            "threshold_from_val": best_thr,
            "c_fp": float(args.c_fp),
            "c_fn": float(args.c_fn),
            "test_roc_auc": float(test_auc),
            "test_pr_auc": float(test_pr_auc),
            "test_threshold_metrics": test_thr_metrics,
            "test_total_cost": float(test_cost),
        },
    )

    # Print summary
    print(f"[OK] Loaded run: {run_path}")
    print(f"[VAL] ROC-AUC={val_auc:.4f}  PR-AUC={val_pr_auc:.4f}")
    print(f"[VAL] Best threshold (min cost)={best_thr:.2f}  cost={best['total_cost']:.2f}  "
          f"P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}")
    print(f"[TEST] ROC-AUC={test_auc:.4f} PR-AUC={test_pr_auc:.4f}")
    print(f"[TEST] Using thr={best_thr:.2f}  "
          f"P={test_thr_metrics['precision']:.4f} R={test_thr_metrics['recall']:.4f} "
          f"F1={test_thr_metrics['f1']:.4f} cost={test_cost:.2f}")
    print(f"[SAVED] {run_path / 'threshold_report.csv'}")
    print(f"[SAVED] {run_path / 'best_threshold.json'}")
    print(f"[SAVED] {run_path / 'test_report.json'}")


if __name__ == "__main__":
    main()