"""Credit risk scorecard - Streamlit dashboard.

Loads the artifacts saved from the training notebook and scores one applicant:
  raw inputs -> cleaning rules -> WoE -> logistic regression -> score -> risk band
XGBoost is shown alongside as a challenger model.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Scorecard", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    return joblib.load("scorecard_artifacts.joblib")


A = load_artifacts()
P = A["params"]


def clean_row(raw: dict) -> pd.DataFrame:
    """Same cleaning rules as training (all learned from the training set)."""
    d = pd.DataFrame([raw])
    d["special_code"] = (d["NumberOfTimes90DaysLate"] >= 96).astype(int)
    for c in A["codes"]:
        d.loc[d[c] >= 96, c] = 0
    d["income_missing"] = d["MonthlyIncome"].isna().astype(int)
    bad_dr = d["MonthlyIncome"].isna() | (d["MonthlyIncome"] == 0)
    d.loc[bad_dr, "DebtRatio"] = P["debt_median"]
    d["MonthlyIncome"] = d["MonthlyIncome"].fillna(P["income_median"])
    d["NumberOfDependents"] = d["NumberOfDependents"].fillna(P["dep_median"])
    d["RevolvingUtilizationOfUnsecuredLines"] = d["RevolvingUtilizationOfUnsecuredLines"].clip(upper=P["util_cap"])
    d["DebtRatio"] = d["DebtRatio"].clip(upper=P["debt_cap"])
    d["MonthlyIncome"] = d["MonthlyIncome"].clip(upper=P["income_cap"])
    return d


def to_woe(d: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(index=d.index)
    for col in A["features"]:
        f = d[col]
        if col in A["count_cols"]:
            g = f.clip(upper=3)
        elif col == "special_code":
            g = f
        else:
            g = pd.cut(f, bins=A["edge_maps"][col])
        w[col] = g.map(A["woe_maps"][col]).astype(float)
    return w[A["features"]]


def score_applicant(raw: dict):
    d = clean_row(raw)
    w = to_woe(d)
    p_lr = float(np.clip(A["lr"].predict_proba(w)[0, 1], 1e-6, 1 - 1e-6))
    score = A["offset"] + A["factor"] * np.log((1 - p_lr) / p_lr)
    xgb_cols = A["xgb"].get_booster().feature_names
    p_xgb = float(A["xgb"].predict_proba(d[xgb_cols])[0, 1])
    band = pd.cut([score], bins=A["band_edges"], labels=A["band_labels"])[0]
    return score, p_lr, p_xgb, str(band), w.iloc[0]


# ------------------------------------------------------------------ UI
st.title("💳 Credit Risk Scorecard")
st.caption(
    "Logistic-regression scorecard (WoE) with an XGBoost challenger, trained on the "
    "Kaggle *Give Me Some Credit* data. Educational project - not a real lending decision."
)

with st.form("applicant"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 100, 40)
    income_known = c2.checkbox("Monthly income known", value=True)
    income = c2.number_input("Monthly income", 0, 1_000_000, 6000, step=500, disabled=not income_known)
    util = c1.slider("Credit utilisation (balance / limit)", 0.0, 1.5, 0.30, 0.01)
    debt_ratio = c2.number_input("Debt ratio (monthly debt payments / income)", 0.0, 5.0, 0.30, 0.01)
    late30 = c1.number_input("Times 30-59 days late (last 2 yrs)", 0, 20, 0)
    late60 = c2.number_input("Times 60-89 days late", 0, 20, 0)
    late90 = c1.number_input("Times 90+ days late", 0, 20, 0)
    open_lines = c2.number_input("Open credit lines and loans", 0, 60, 6)
    real_estate = c1.number_input("Real estate loans / lines", 0, 30, 1)
    dependents = c2.number_input("Number of dependents", 0, 15, 0)
    submitted = st.form_submit_button("Score applicant", use_container_width=True)

st.sidebar.header("Lending policy")
cutoff = st.sidebar.slider("Approval cutoff (score)", 480, 630, 540, 5)
EAD = st.sidebar.number_input("Loan amount / EAD (Rs)", 10_000, 10_000_000, 100_000, step=10_000)
LGD = st.sidebar.slider("Loss given default (LGD)", 0.10, 1.00, 0.60, 0.05)
st.sidebar.caption("Cutoff 540 maximised profit in the notebook under the stated assumptions.")

if submitted:
    raw = {
        "RevolvingUtilizationOfUnsecuredLines": util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": float(income) if income_known else np.nan,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late60,
        "NumberOfDependents": dependents,
    }
    score, p_lr, p_xgb, band, woe_row = score_applicant(raw)
    approved = score >= cutoff

    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Credit score", f"{score:.0f}")
    m2.metric("Probability of default", f"{p_lr:.1%}")
    m3.metric("Risk band", band)

    if approved:
        st.success(f"APPROVE - score {score:.0f} is at or above the cutoff of {cutoff}.")
    else:
        st.error(f"DECLINE - score {score:.0f} is below the cutoff of {cutoff}.")

    st.markdown("**Expected loss** = PD x LGD x EAD")
    st.write(f"{p_lr:.1%} x {LGD:.0%} x Rs {EAD:,.0f} = **Rs {p_lr * LGD * EAD:,.0f}**")
    st.write(f"XGBoost challenger PD: **{p_xgb:.1%}** (scorecard PD: {p_lr:.1%})")

    st.markdown("**What drove this score?** (contribution of each feature, in log-odds of good vs bad)")
    contrib = pd.Series(A["lr"].coef_[0] * woe_row.values, index=A["features"])
    st.bar_chart(contrib.sort_values())
    st.caption("Positive bars push towards *bad*, negative bars towards *good*, because the model predicts default.")
