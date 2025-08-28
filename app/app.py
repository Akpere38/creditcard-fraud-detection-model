import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Credit Card Fraud Detector — Streamlit App")
st.caption(
    "Use the default Kaggle dataset or upload your own transactions. Get per-transaction fraud probability, adjustable threshold, and interactive PR curve."
)


@st.cache_data(show_spinner=True)
def load_default_data(default_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load default dataset from CSV."""
    if not os.path.exists(default_path):
        raise FileNotFoundError(
            f"Default csv not found at '{default_path}'. Please place Kaggle 'creditcard.csv' there."
        )
    df = pd.read_csv(default_path)
    return df


def validate_columns(df: pd.DataFrame, require_target: bool = False) -> tuple[bool, str]:
    """Validate required columns in the DataFrame."""
    needed = {"Time", "Amount"}
    # The Kaggle set has V1..V28
    needed.update({f"V{i}" for i in range(1, 29)})
    if require_target:
        needed.add("Class")

    missing = [c for c in sorted(needed) if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    return True, ""


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame, has_target: bool = True):
    """Preprocess the DataFrame."""
    df = df.copy()
    scaler = RobustScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    if has_target:
        X = df.drop(columns=["Class"])
        y = df["Class"].astype(int)
    else:
        X, y = df, None
    return X, y


@st.cache_resource(show_spinner=True)
def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Train XGBoost classifier."""
    # Handle imbalance with scale_pos_weight = negatives / positives
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=spw,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def pr_curve_figure(y_true, y_prob, title="Precision-Recall Curve"):
    """Create a Precision-Recall curve figure."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode="lines", name=f"PR curve (AP={ap:.3f})")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(range=[0, 1]),
        height=400,
    )
    return fig


def confusion_matrix_figure(y_true, y_pred, title="Confusion Matrix"):
    """Create a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    z = cm
    x = ["Pred: Non-Fraud (0)", "Pred: Fraud (1)"]
    y = ["Actual: Non-Fraud (0)", "Actual: Fraud (1)"]
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y))
    fig.update_layout(title=title, height=360)
    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(
                x=x[j], y=y[i],
                text=str(cm[i, j]), showarrow=False,
                font=dict(color="white", size=14)
            )
    return fig

# -------------------------------
# Sidebar — Data Source
# -------------------------------
st.sidebar.header("Data Source")
source = st.sidebar.radio(
    "Choose dataset:",
    ("Use default Kaggle CSV", "Upload my CSV"),
    index=0,
)

uploaded_file = None

# --- Option 1: Use Default Dataset ---
if source == "Use default Kaggle CSV":
    try:
        df = load_default_data()
        st.success("Loaded default dataset: data/creditcard.csv")

        # Add download button for default dataset
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            label="📥 Download Default Dataset",
            data=csv_data,
            file_name="creditcard.csv",
            mime="text/csv",
        )

    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# --- Option 2: Upload Dataset ---
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload transactions CSV", 
        type=["csv"]
    )
    if uploaded_file is None:
        st.info("Please upload a CSV to proceed.")
        st.stop()
    else:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded.")

# -------------------------------
# Threshold control
# -------------------------------
st.sidebar.header("Threshold")
threshold = st.sidebar.slider(
    "Fraud classification threshold (on predicted probability of class=1)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01
) 

st.sidebar.markdown("Please download the credit card dataset from kaggle and upload it to continue: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
# Validate columns
has_target = "Class" in df.columns
ok, msg = validate_columns(df, require_target=has_target)
if not ok:
    st.error(msg)
    st.stop()

# Show basic info
with st.expander("🔎 Peek data (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Rows", f"{len(df):,}")
col_b.metric("Columns", f"{df.shape[1]:,}")
if has_target:
    fraud_rate = df["Class"].mean() * 100
    col_c.metric("Fraud Rate", f"{fraud_rate:.2f}%")
else:
    col_c.metric("Fraud Rate", "N/A (no labels)")
col_d.metric("Threshold", f"{threshold:.2f}")

# -------------------------------
# Train / Predict Workflow
# -------------------------------
X, y = preprocess(df, has_target=has_target)

# If labels are present, train and evaluate using a train/test split
if has_target:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = train_xgb(X_train, y_train)

    # Predictions on test set
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    st.subheader("📈 Evaluation (hold-out test set)")
    m1, m2 = st.columns(2)
    with m1:
        st.plotly_chart(pr_curve_figure(y_test, y_prob), use_container_width=True)
        st.caption(f"Average Precision (PR-AUC): **{pr_auc:.4f}** · ROC-AUC: **{roc_auc:.4f}**")
    with m2:
        st.plotly_chart(confusion_matrix_figure(y_test, y_pred), use_container_width=True)

    # Classification report
    with st.expander("📃 Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        rep_df = pd.DataFrame(report).T
        st.dataframe(rep_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}"}), use_container_width=True)

    # Show top highest-risk transactions in test set
    st.subheader("🔝 Highest-risk transactions (test set)")
    k = st.slider("Show top K by fraud probability", 5, 200, 20, 5)
    topk_idx = np.argsort(-y_prob)[:k]
    topk = X_test.iloc[topk_idx].copy()
    topk["Actual"] = y_test.iloc[topk_idx].values
    topk["Predicted"] = y_pred[topk_idx]
    topk["Fraud_Probability"] = y_prob[topk_idx]
    st.dataframe(topk, use_container_width=True)

    # Download predictions (test set)
    pred_out = X_test.copy()
    pred_out["Actual"] = y_test.values
    pred_out["Fraud_Probability"] = y_prob
    pred_out["Predicted"] = y_pred
    csv_bytes = pred_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download test-set predictions (CSV)",
        data=csv_bytes,
        file_name="fraud_test_predictions.csv",
        mime="text/csv",
    )

else:
    # No labels — just score the uploaded/default data
    st.subheader("🔮 Scoring (no labels in data)")
    st.info(
        "No 'Class' column found. The app will train on the default dataset (if available) and score your uploaded data."
    )
    # Train using default data with labels
    try:
        df_train = load_default_data()
        ok2, msg2 = validate_columns(df_train, require_target=True)
        if not ok2:
            st.error("Default training data invalid: " + msg2)
            st.stop()
        X_train_all, y_train_all = preprocess(df_train, has_target=True)
        model = train_xgb(X_train_all, y_train_all)
    except Exception as e:
        st.error(f"Cannot train model because default labeled dataset is unavailable. {e}")
        st.stop()

    # Score current data
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    st.caption("Scored your dataset. Showing top highest-risk transactions.")
    k = st.slider("Show top K by fraud probability", 5, 200, 20, 5)
    topk_idx = np.argsort(-y_prob)[:k]
    scored = X.iloc[topk_idx].copy()
    scored["Predicted"] = y_pred[topk_idx]
    scored["Fraud_Probability"] = y_prob[topk_idx]
    st.dataframe(scored, use_container_width=True)

    # Download scored results
    out = X.copy()
    out["Fraud_Probability"] = y_prob
    out["Predicted"] = y_pred
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download scored predictions (CSV)",
        data=csv_bytes,
        file_name="fraud_scored_predictions.csv",
        mime="text/csv",
    )

# -------------------------------
# Extras: Probability threshold exploration
# -------------------------------
st.markdown("---")
st.subheader("🧪 Threshold Exploration")
if has_target:
    # Explore how precision/recall change as threshold moves
    y_prob_all = y_prob  # from above evaluation branch
    precision, recall, thresh = precision_recall_curve(y_test, y_prob_all)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=thresh, y=precision[:-1], mode="lines", name="Precision"))
    fig2.add_trace(go.Scatter(x=thresh, y=recall[:-1], mode="lines", name="Recall"))
    fig2.update_layout(
        title="Precision & Recall vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Built with XGBoost, robust scaling, and stratified evaluation. Tip: lower the threshold to increase recall (catch more fraud) at the cost of more false positives."
)
