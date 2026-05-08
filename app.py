import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.caption("Team 22: predict which customers are likely to leave.")


# -----------------------------
# Sidebar: pick a trained run + threshold
# -----------------------------
RUNS_DIR = Path("runs")

if not RUNS_DIR.exists() or not any(RUNS_DIR.iterdir()):
    st.error("No trained runs found. Run `python -m src.train` first.")
    st.stop()

run_options = sorted(
    [p for p in RUNS_DIR.iterdir() if p.is_dir()],
    key=lambda p: p.name,
    reverse=True,
)
run_path = st.sidebar.selectbox("Run", run_options, format_func=lambda p: p.name)

# Try to load the run's chosen threshold; fall back to 0.5
default_thr = 0.5
best_thr_path = run_path / "best_threshold.json"
if best_thr_path.exists():
    try:
        default_thr = float(json.loads(best_thr_path.read_text())["threshold"])
    except Exception:
        pass

threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, default_thr, 0.01)
st.sidebar.write(f"Customers with P(churn) ≥ **{threshold:.2f}** are flagged.")


@st.cache_resource
def load_artifacts(run_dir: Path):
    model = joblib.load(run_dir / "model.joblib")
    preprocess = joblib.load(run_dir / "preprocess.joblib")
    return model, preprocess


model, preprocess = load_artifacts(run_path)


def score(df: pd.DataFrame) -> np.ndarray:
    """Run preprocess + model on a raw-shaped dataframe; return P(churn)."""
    X = preprocess.transform(df)
    return model.predict_proba(X)[:, 1]


# -----------------------------
# Tabs
# -----------------------------
tab_single, tab_batch, tab_metrics = st.tabs(
    ["Single customer", "Batch CSV", "Run metrics"]
)

# -----------------------------
# Single customer
# -----------------------------
with tab_single:
    st.subheader("Score one customer")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=80, value=12)
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["Yes", "No", "No phone service"]
        )

    with col2:
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox(
            "Online security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5
        )
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, value=float(monthly_charges * tenure)
        )

    if st.button("Predict", type="primary"):
        row = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )
        prob = float(score(row)[0])
        flagged = prob >= threshold

        c1, c2 = st.columns(2)
        c1.metric("Churn probability", f"{prob:.1%}")
        c2.metric("Decision", "AT RISK: outreach" if flagged else "Likely to stay")
        st.progress(prob)


# -----------------------------
# Batch CSV
# -----------------------------
with tab_batch:
    st.subheader("Score a CSV of customers")
    st.write(
        "Upload a CSV with the same columns as `data/raw/telco.csv` "
        "(the `Churn` column is optional and will be ignored)."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        # Match the cleaning step from src/data.py for TotalCharges
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            if "tenure" in df.columns:
                mask = (df["tenure"] == 0) & (df["TotalCharges"].isna())
                df.loc[mask, "TotalCharges"] = 0.0

        feature_df = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])

        try:
            proba = score(feature_df)
        except Exception as e:
            st.error(f"Could not score this file: {e}")
            st.stop()

        out = df.copy()
        out["churn_prob"] = proba
        out["flagged"] = (proba >= threshold).astype(int)

        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", f"{len(out):,}")
        c2.metric("Flagged at risk", f"{int(out['flagged'].sum()):,}")
        c3.metric("Flag rate", f"{out['flagged'].mean():.1%}")

        st.dataframe(
            out.sort_values("churn_prob", ascending=False),
            use_container_width=True,
            height=400,
        )

        st.download_button(
            "Download scored CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"scored_{run_path.name}.csv",
            mime="text/csv",
        )


# -----------------------------
# Run metrics
# -----------------------------
with tab_metrics:
    st.subheader(f"Metrics for `{run_path.name}`")

    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        val = metrics.get("val", {})
        c1, c2 = st.columns(2)
        c1.metric("Validation ROC-AUC", f"{val.get('roc_auc', float('nan')):.4f}")
        c2.metric("Validation PR-AUC", f"{val.get('avg_precision_pr_auc', float('nan')):.4f}")
        with st.expander("Full metrics.json"):
            st.json(metrics)

    test_path = run_path / "test_report.json"
    if test_path.exists():
        st.markdown("### Test set report (from `eval.py`)")
        st.json(json.loads(test_path.read_text()))

    thr_csv = run_path / "threshold_report.csv"
    if thr_csv.exists():
        st.markdown("### Threshold sweep")
        thr_df = pd.read_csv(thr_csv)
        st.line_chart(thr_df.set_index("threshold")[["total_cost"]])
        with st.expander("Full table"):
            st.dataframe(thr_df, use_container_width=True)
