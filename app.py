import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from maintenance_rules import risk_category, maintenance_suggestions

st.set_page_config(page_title="Predictive Maintenance Intelligence", page_icon="🛠️", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_resource
def load_bundle():
    return joblib.load("model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("data_clean.csv"), json.load(open("metrics.json")), pd.read_csv("feature_importance.csv")

bundle = load_bundle()
pipe, THR, FEATURES = bundle["pipeline"], bundle["threshold"], bundle["features"]
data, metrics, importance = load_data()
LABELS = {"air_temp": "Air temperature [K]", "process_temp": "Process temperature [K]",
          "rpm": "Rotational speed [rpm]", "torque": "Torque [Nm]", "tool_wear": "Tool wear [min]"}

st.title("🛠️ AI-Powered Predictive Maintenance & Equipment Failure Intelligence")
st.caption("Decision-support tool trained on the synthetic AI4I 2020 dataset. Predictions are estimates, not guarantees.")

tab_pred, tab_an, tab_perf, tab_exp = st.tabs(["🔮 Predict", "📊 Machine Analytics", "📈 Model Performance", "🧠 Explainability"])

# ---------------- Predict ----------------
with tab_pred:
    st.sidebar.header("Machine operating conditions")
    rec = {}
    for col in ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]:
        lo, hi, med = float(data[col].min()), float(data[col].max()), float(data[col].median())
        step = 1.0 if col in ("rpm", "tool_wear") else 0.1
        rec[col] = st.sidebar.slider(LABELS[col], lo, hi, med, step)
    rec["type"] = st.sidebar.selectbox("Product type (L / M / H)", ["L", "M", "H"], index=1)

    prob = float(pipe.predict_proba(pd.DataFrame([rec])[FEATURES])[:, 1][0])
    level = risk_category(prob, THR)
    color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[level]

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure probability (risk score)", f"{prob:.1%}")
    c2.metric("Risk category", f"{color} {level}")
    c3.metric("Predicted label", "FAILURE RISK" if prob >= THR else "No failure expected")
    st.progress(min(prob, 1.0))
    st.caption(f"Risk bands: Low < {THR/2:.1%} ≤ Medium < {THR:.1%} ≤ High  (decision threshold tuned on validation data)")

    st.subheader("Suggested checks")
    for tip in maintenance_suggestions(rec, prob, THR):
        st.write(tip)

    with st.expander("Derived quantities"):
        st.write(f"Mechanical power: **{rec['torque'] * rec['rpm'] * 2 * np.pi / 60:,.0f} W**  |  "
                 f"Process − air temperature: **{rec['process_temp'] - rec['air_temp']:.1f} K**  |  "
                 f"Tool wear × torque: **{rec['tool_wear'] * rec['torque']:,.0f} min·Nm**")

# ---------------- Analytics ----------------
with tab_an:
    st.subheader("Historical machine-condition patterns")
    col = st.selectbox("Feature", list(LABELS), format_func=LABELS.get)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=data, x=col, hue="failure", common_norm=False, fill=True, palette=["#4C9F70", "#D1495B"], ax=ax)
        ax.axvline(rec[col], color="black", ls="--", label="current input"); ax.legend()
        ax.set_title(f"{LABELS[col]}: failure vs no failure"); st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=data.sort_values("failure"), x="rpm", y="torque", hue="failure", palette=["#BBBBBB", "#D1495B"], s=12, alpha=0.6, ax=ax)
        ax.scatter(rec["rpm"], rec["torque"], c="black", marker="X", s=150, label="current input")
        ax.legend(); ax.set_title("Torque vs rotational speed"); st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(data[list(LABELS) + ["failure"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix"); st.pyplot(fig)
    st.metric("Overall historical failure rate", f"{data['failure'].mean():.2%}")

# ---------------- Performance ----------------
with tab_perf:
    t = metrics["test"]
    st.subheader(f"Selected model: {metrics['selected_model']}  (threshold {metrics['threshold']:.3f})")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recall", f"{t['recall']:.3f}"); m2.metric("Precision", f"{t['precision']:.3f}")
    m3.metric("F1", f"{t['f1']:.3f}"); m4.metric("ROC-AUC", f"{t['roc_auc']:.3f}"); m5.metric("PR-AUC", f"{t['pr_auc']:.3f}")
    cm = np.array([[t["TN"], t["FP"]], [t["FN"], t["TP"]]])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred: OK", "Pred: Failure"], yticklabels=["Actual: OK", "Actual: Failure"])
    ax.set_title("Confusion matrix (test set)"); st.pyplot(fig)
    st.warning(f"Missed failures (false negatives): **{t['FN']}** of {t['TP'] + t['FN']} real failures.")
    st.subheader("Validation comparison")
    st.dataframe(pd.DataFrame(metrics["validation"]).T[["threshold", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]])

# ---------------- Explainability ----------------
with tab_exp:
    st.subheader("Permutation feature importance (selected model)")
    imp = importance.sort_values("importance")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.barh(imp["feature"], imp["importance"], color="#D1495B")
    ax.set_xlabel("Drop in PR-AUC when shuffled"); st.pyplot(fig)
    st.info("Combine this with the rule-based checks on the Predict tab: the model ranks risk, the rules explain what to inspect.")
