import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(layout="centered")
st.title("Adam Kadri's Histogram Graphing Generator")

st.header("1) Data input (index, value)")

input_mode = st.radio("Provide data by hand or upload a CSV:", ["Manually", "Upload CSV"])

df = None
values = None

if input_mode == "Manually":
    txt = st.text_area(
        "Enter numeric values (commas or newlines). Example:\n\n"
        "3.2, 4.1, 5.0\n\nor\n\n1\n2\n3\n4",
        height=120,
    )
    if txt:
        parts = [p.strip() for p in txt.replace(",", " ").split()]
        try:
            values = np.array([float(p) for p in parts])
            df = pd.DataFrame({"index": np.arange(len(values)), "value": values})
            st.subheader("Data (index, value)")
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.error("Could not parse numbers. Enter only numeric values separated by commas or newlines.")
else:
    uploaded = st.file_uploader("Upload CSV (two columns: index,value) or single column (value)", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)

            if raw.shape[1] >= 2:
                df = pd.DataFrame({"index": raw.iloc[:, 0].values, "value": raw.iloc[:, 1].values})
            else:
                col = raw.iloc[:, 0].dropna().values
                df = pd.DataFrame({"index": np.arange(len(col)), "value": col})

            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).reset_index(drop=True)
            values = df["value"].values
            st.subheader("Data (index, value)")
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.error("Failed to read CSV. Ensure it is a valid CSV file with numeric values.")

if df is None or values is None or len(values) == 0:
    st.info("Please enter data manually or upload a CSV to continue.")
    st.stop()

values = np.asarray(values)
values = values[np.isfinite(values)]
if values.size == 0:
    st.error("No valid numeric values found after cleaning.")
    st.stop()


st.header("2) Fit a distribution and visualize")

dist_list = [
    "norm", "gamma", "weibull_min", "expon", "beta",
    "lognorm", "chi2", "uniform", "t", "pareto",
    "f", "laplace"
]
dist_name = st.selectbox("Choose distribution to fit:", dist_list)
dist = getattr(stats, dist_name)

try:
    params = dist.fit(values)
except Exception as e:
    st.error(f"Could not fit {dist_name} to the data: {e}")
    st.stop()

data_min, data_max = float(values.min()), float(values.max())
pad = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
x_min, x_max = data_min - pad, data_max + pad

preferred_bins = 100
bins = min(preferred_bins, max(10, int(len(values) // 1)))

bin_edges = np.linspace(x_min, x_max, bins + 1)
hist_vals, hist_edges = np.histogram(values, bins=bin_edges, density=True)
bin_widths = np.diff(hist_edges)
hist_lefts = hist_edges[:-1] 

x_plot = np.linspace(x_min, x_max, 400)
pdf_plot = dist.pdf(x_plot, *params)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(hist_lefts, hist_vals, width=bin_widths, align='edge',
       color="orange", edgecolor="black", alpha=0.85, label="Data (density)")
ax.plot(x_plot, pdf_plot, color="blue", lw=2, label=f"{dist_name} fit")

hist_max = hist_vals.max() if hist_vals.size > 0 else 0.0
pdf_max = pdf_plot.max() if pdf_plot.size > 0 else 0.0
y_top = max(hist_max, pdf_max) * 1.10
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, max(y_top, 0.01))
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.header("3) Fitted parameters and fit quality")

st.subheader("Fitted parameters")
st.write(tuple(float(p) for p in params))

hist_y, hist_x = np.histogram(values, bins=bin_edges, density=True)
hist_mid = (hist_x[:-1] + hist_x[1:]) / 2
pdf_mid = dist.pdf(hist_mid, *params)
avg_err = float(np.mean(np.abs(pdf_mid - hist_y)))
max_err = float(np.max(np.abs(pdf_mid - hist_y)))

st.subheader("Goodness of fit")
st.write(f"**Average error (mean absolute difference):** {avg_err:.4f}")
st.write(f"**Max error (max absolute difference):** {max_err:.4f}")

st.header("4) Manual parameter fitting")
with st.expander("Adjust distribution parameters manually"):
    manual_params = []
    for i, p in enumerate(params):
        base = abs(p) if p != 0 else 1.0
        if p > 0:
            low = float(p * 0.1)
            high = float(p * 2.0)
        else:
            low = float(-base * 2.0)
            high = float(base * 2.0)
        if low == high:
            low, high = -base, base
        step = max((high - low) / 100.0, 1e-6)
        manual_params.append(st.slider(f"Parameter {i+1}", low, high, float(p), step=step))

    manual_pdf = dist.pdf(x_plot, *manual_params)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    hist_vals2, hist_edges2 = np.histogram(values, bins=bin_edges, density=True)
    bin_widths2 = np.diff(hist_edges2)
    hist_lefts2 = hist_edges2[:-1]

    ax2.bar(hist_lefts2, hist_vals2, width=bin_widths2, align='edge',
            color="orange", edgecolor="black", alpha=0.85, label="Data (density)")
    ax2.plot(x_plot, manual_pdf, color="green", lw=2, label="Manual fit")

    hist_max2 = hist_vals2.max() if hist_vals2.size > 0 else 0.0
    pdf_max2 = manual_pdf.max() if manual_pdf.size > 0 else 0.0
    y_top2 = max(hist_max2, pdf_max2) * 1.10
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, max(y_top2, 0.01))
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.write("Manual parameters:", tuple(float(p) for p in manual_params))
