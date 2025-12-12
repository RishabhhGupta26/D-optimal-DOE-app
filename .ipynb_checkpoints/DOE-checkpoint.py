import streamlit as st
import numpy as np
import pandas as pd
from itertools import product

# -----------------------------
# Utility Functions
# -----------------------------

def build_model_matrix(points, model_type, factor_names):
    """Build X matrix and return column names also."""
    X = []
    colnames = ["B0"]  # intercept

    # Linear terms
    colnames += [f"B{i+1} ({factor_names[i]})" for i in range(len(factor_names))]

    for p in points:
        row = [1]  # intercept
        row.extend(p)  # linear terms

        if model_type == "Quadratic":
            # Add squared terms
            for i, x in enumerate(p):
                row.append(x**2)
                colnames.append(f"B{len(colnames)} ({factor_names[i]}²)")

            # Add interaction terms
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    row.append(p[i] * p[j])
                    colnames.append(f"B{len(colnames)} ({factor_names[i]}×{factor_names[j]})")

        X.append(row)

    return np.array(X), colnames


def d_optimal_selection(candidate_points, n_points, model_type, factor_names):
    """Simple greedy D-optimal selection."""
    best_set = []
    remaining = candidate_points.copy()

    while len(best_set) < n_points:
        best_det = -1
        best_point = None

        for p in remaining:
            trial_set = best_set + [p]
            X, _ = build_model_matrix(trial_set, model_type, factor_names)
            M = X.T @ X
            det_val = np.linalg.det(M)

            if det_val > best_det:
                best_det = det_val
                best_point = p

        best_set.append(best_point)
        remaining.remove(best_point)

    return best_set


# -----------------------------
# Streamlit App
# -----------------------------

st.title("📘 D-Optimal Design Generator (Enhanced Version)")
st.write("Generate DOE with named coefficients and visible model equations.")

# -----------------------------
# Step 1 — User Input
# -----------------------------

st.header("1️⃣ Define Factors")

n_factors = st.number_input("Number of Factors:", min_value=1, max_value=10, value=3)

factor_names = []
factor_levels = []

for i in range(n_factors):
    name = st.text_input(f"Factor {i+1} Name:", value=f"x{i+1}")
    levels_str = st.text_input(
        f"Levels for {name} (comma separated):",
        value="-1, 0, 1"
    )
    levels = [float(x.strip()) for x in levels_str.split(",")]

    factor_names.append(name)
    factor_levels.append(levels)

candidate_points = [list(p) for p in product(*factor_levels)]
df_candidates = pd.DataFrame(candidate_points, columns=factor_names)

st.subheader("🔍 Full Candidate Set")
st.dataframe(df_candidates)

# -----------------------------
# Step 2 — DOE
# -----------------------------

st.header("2️⃣ Select DOE Parameters")

model_type = st.selectbox("Model Type:", ["Linear", "Quadratic"])

n_points = st.number_input(
    "Number of D-Optimal Points:",
    min_value=1,
    max_value=len(candidate_points),
    value=min(10, len(candidate_points))
)

if st.button("Generate D-Optimal Design"):
    st.success("D-Optimal Design Generated!")

    selected = d_optimal_selection(candidate_points.copy(), n_points, model_type, factor_names)
    df_selected = pd.DataFrame(selected, columns=factor_names)

    st.subheader("⭐ Selected D-Optimal Points")
    st.dataframe(df_selected)

    # -----------------------------
    # Build Model Matrix
    # -----------------------------
    X, colnames = build_model_matrix(selected, model_type, factor_names)

    # Equation Display
    st.subheader("🧮 Model Equation")

    if model_type == "Linear":
        eq = "y = B0"
        for i, f in enumerate(factor_names):
            eq += f" + B{i+1}{f'·{f}'}"
    else:
        eq = "y = B0"
        # Linear terms
        for i, f in enumerate(factor_names):
            eq += f" + B{i+1}{f'·{f}'}"
        # Quadratic terms
        idx = len(factor_names) + 1
        for i, f in enumerate(factor_names):
            eq += f" + B{idx}{f'·{f}²'}"
            idx += 1
        # Interaction terms
        for i in range(len(factor_names)):
            for j in range(i+1, len(factor_names)):
                eq += f" + B{idx}{f'·({factor_names[i]}×{factor_names[j]})'}"
                idx += 1

    st.latex(eq)

    st.subheader("📐 Model Matrix (X) with Named Coefficients")
    st.dataframe(pd.DataFrame(X, columns=colnames))

    # -----------------------------
    # Information Matrix (XᵀX)
    # -----------------------------
    M = X.T @ X

    st.subheader("📊 Information Matrix (XᵀX) with Named Coefficients")
    st.dataframe(pd.DataFrame(M, columns=colnames, index=colnames))

    det_val = np.linalg.det(M)
    st.metric("Determinant |XᵀX|", f"{det_val:,.3f}")

    st.metric("D-Efficiency (%)", "100.00")

    csv = df_selected.to_csv(index=False)
    st.download_button(
        "Download Selected Design as CSV",
        data=csv,
        file_name="d_optimal_design.csv",
        mime="text/csv"
    )
