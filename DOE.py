import streamlit as st
import numpy as np
import pandas as pd
from itertools import product

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="D-Optimal DOE Generator", layout="wide")

# ------------------------------------------------------
# HEADER WITH TITLE + LOGO ON RIGHT
# ------------------------------------------------------
header = st.container()
with header:
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(
            "<h2 style='margin-bottom:0px;'>📘 D-Optimal Design Generator</h2>",
            unsafe_allow_html=True
        )

    with col2:
        try:
            st.image(r"C:\Users\risha\work\D Optimal DOE\download.png", width=120)
        except:
            st.write("⚠️ Logo not found")

# ------------------------------------------------------
# DOE NOTES BOX (Under header)
# ------------------------------------------------------
st.markdown("""
<div style='background-color:#F5F5F5;padding:12px;border-radius:8px;width:55%;margin-top:10px;'>
<b>📌 DOE Quality Notes (Based on Information Matrix XᵀX)</b><br>
- High <b>Determinant (|XᵀX|)</b> → stronger model stability<br>
- Large <b>Diagonal values</b> → lower variance in coefficients<br>
- Small <b>Off-diagonals</b> → low correlation between variables<br>
- Quadratic models require more runs than linear models<br>
- <b>D-Efficiency (%)</b> → higher = better design<br>
</div>
""", unsafe_allow_html=True)

# ======================================================
# FUNCTION — BUILD MODEL MATRIX
# ======================================================
def build_model_matrix(points, model_type, factor_names):
    n = len(factor_names)

    colnames = ["B0"]
    colnames += [f"B{i+1} ({factor_names[i]})" for i in range(n)]

    if model_type == "Quadratic":
        colnames += [f"B_sq_{factor_names[i]} ({factor_names[i]}²)" for i in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colnames.append(f"B_int_{factor_names[i]}×{factor_names[j]}")

    X = []
    for p in points:
        row = [1]
        row.extend(p)

        if model_type == "Quadratic":
            row.extend([x**2 for x in p])
            for i in range(n):
                for j in range(i+1, n):
                    row.append(p[i] * p[j])

        X.append(row)

    return np.array(X, float), colnames

# ======================================================
# FUNCTION — D-OPTIMAL GREEDY SELECTION
# ======================================================
def d_optimal_selection(candidate_points, n_points, model_type, factor_names):
    best = []
    remaining = candidate_points.copy()

    while len(best) < n_points:
        best_det = -1
        best_point = None

        for p in remaining:
            trial = best + [p]
            X, _ = build_model_matrix(trial, model_type, factor_names)
            M = X.T @ X
            try:
                det_val = np.linalg.det(M)
            except:
                det_val = -1

            if det_val > best_det:
                best_det = det_val
                best_point = p

        best.append(best_point)
        remaining.remove(best_point)

    return best

# ======================================================
# SECTION 1 — DEFINE FACTORS
# ======================================================
st.header("1️⃣ Define Factors")

n_factors = st.number_input(
    "Number of Factors:",
    min_value=1, max_value=10,
    value=3
)

factor_names, factor_levels = [], []

for i in range(n_factors):
    name = st.text_input(f"Factor {i+1} Name:", value=f"x{i+1}")
    level_str = st.text_input(
        f"Levels for {name} (comma separated):",
        value="-1, 0, 1"
    )

    levels = [float(x.strip()) for x in level_str.split(",")]

    factor_names.append(name)
    factor_levels.append(levels)

# Generate candidate points
candidate_points = [list(p) for p in product(*factor_levels)]
df_candidates = pd.DataFrame(candidate_points, columns=factor_names)

st.subheader("🔍 Full Candidate Set")
st.dataframe(df_candidates)

# ======================================================
# SECTION 2 — DOE GENERATION
# ======================================================
st.header("2️⃣ Select DOE Parameters")

model_type = st.selectbox("Model Type:", ["Linear", "Quadratic"])

n_points = st.number_input(
    "Number of D-Optimal Points:",
    min_value=1,
    max_value=len(candidate_points),
    value=min(10, len(candidate_points))
)

# ------------------------------------------------------
# GENERATE BUTTON
# ------------------------------------------------------
if st.button("Generate D-Optimal Design"):

    st.success("D-Optimal DOE Generated Successfully!")

    # Select DOE points
    selected = d_optimal_selection(
        candidate_points.copy(), n_points,
        model_type, factor_names
    )

    df_sel = pd.DataFrame(selected, columns=factor_names)

    st.subheader("⭐ Selected D-Optimal Points")
    st.dataframe(df_sel)

    # Build X matrix
    X, colnames = build_model_matrix(selected, model_type, factor_names)

    # ------------------------------------------------------
    # MODEL EQUATION DISPLAY
    # ------------------------------------------------------
    st.subheader("🧮 Model Equation")

    eq = "y = B0"

    for i, f in enumerate(factor_names):
        eq += f" + B{i+1}{f'·{f}'}"

    if model_type == "Quadratic":
        for f in factor_names:
            eq += f" + B_sq_{f}{f'·{f}²'}"
        for i in range(len(factor_names)):
            for j in range(i+1, len(factor_names)):
                eq += f" + B_int_{factor_names[i]}×{factor_names[j]}{f'·({factor_names[i]}×{factor_names[j]})'}"

    st.latex(eq)

    # ------------------------------------------------------
    # MODEL MATRIX TABLE
    # ------------------------------------------------------
    st.subheader("📐 Model Matrix (X)")
    st.dataframe(pd.DataFrame(X, columns=colnames))

    # ------------------------------------------------------
    # INFORMATION MATRIX XᵀX
    # ------------------------------------------------------
    M = X.T @ X
    st.subheader("📊 Information Matrix (XᵀX)")
    st.dataframe(pd.DataFrame(M, index=colnames, columns=colnames))

    # Determinant & efficiency (CORRECTED)
    # compute determinant safely
    try:
        det_val = float(np.linalg.det(M))
    except Exception:
        det_val = 0.0

    # number of parameters (p) and number of runs (N)
    p = X.shape[1]
    N = X.shape[0]

    # compute D-efficiency:
    # if determinant positive and valid, use (|M|)^(1/p) / N * 100
    if det_val > 0 and p > 0 and N > 0:
        d_eff = (abs(det_val) ** (1.0 / p)) / float(N) * 100.0
    else:
        d_eff = 0.0

    st.metric("Determinant |XᵀX|", f"{det_val:,.4f}")
    st.metric("D-Efficiency (%)", f"{d_eff:.4f}")

    # ------------------------------------------------------
    # DOWNLOAD BUTTON
    # ------------------------------------------------------
    csv = df_sel.to_csv(index=False)
    st.download_button(
        "Download Selected Design (CSV)",
        data=csv,
        file_name="d_optimal_design.csv",
        mime="text/csv"
    )

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Made for professional DOE and modelling workflows.</center>",
    unsafe_allow_html=True
)
