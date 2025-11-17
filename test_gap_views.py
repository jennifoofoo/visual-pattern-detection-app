import streamlit as st
import plotly.express as px
import pandas as pd

from core.detection.gap_pattern import GapPattern
from tests.gap_view_tests.synthetic_gap_logs import (
    make_numeric_y_gap,
    make_categorical_y_gap,
    make_actual_time_gap,
    make_relative_time_gap,
    make_relative_ratio_gap,
    make_logical_time_gap,
    make_logical_relative_gap,
)


# ------------------------------------------------------------
# Helper: wrap df + configuration into a dict
# ------------------------------------------------------------

TEST_LOGS = {
    "Numeric Y (2D gap)": {
        "df": make_numeric_y_gap(),
        "x": "x",
        "y": "y",
        "categorical": False
    },
    "Categorical Y (horizontal gap)": {
        "df": make_categorical_y_gap(),
        "x": "x",
        "y": "y",
        "categorical": True
    },
    "Actual Time Gap": {
        "df": make_actual_time_gap(),
        "x": "actual_time",
        "y": "y",
        "categorical": True   # y is categorical
    },
    "Relative Time Gap": {
        "df": make_relative_time_gap(),
        "x": "relative_time",
        "y": "y",
        "categorical": False
    },
    "Relative Ratio Gap": {
        "df": make_relative_ratio_gap(),
        "x": "relative_ratio",
        "y": "y",
        "categorical": False
    },
    "Logical Time Gap": {
        "df": make_logical_time_gap(),
        "x": "logical_time",
        "y": "y",
        "categorical": False
    },
    "Logical Relative Gap": {
        "df": make_logical_relative_gap(),
        "x": "logical_relative",
        "y": "y",
        "categorical": False
    },
}


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.set_page_config(layout="wide", page_title="GapPattern Test Views")

st.title("🔍 GapPattern – View-by-View Test Page")

st.write("""
This mini test page renders all synthetic logs designed for testing 2D and 1D gaps.
You can inspect each dotted chart and see how GapPattern highlights the detected gaps.
""")



# Select test log
log_name = st.selectbox("Choose synthetic test log:", list(TEST_LOGS.keys()))
log = TEST_LOGS[log_name]

df = log["df"]
x = log["x"]
y = log["y"]
is_cat = log["categorical"]

st.subheader(f"Dataset: {log_name}")
st.write(df.head())

# ------------------------------------------------------------
# Base Plot
# ------------------------------------------------------------

fig = px.scatter(
    df,
    x=x,
    y=y,
    color="activity" if "activity" in df.columns else None,
    title=f"View: {y} vs {x}",
    opacity=0.8
)

# Fix categorical axis orientation like main app
if is_cat:
    fig.update_yaxes(categoryorder='array', categoryarray=sorted(df[y].unique()))

# ------------------------------------------------------------
# Gap Detection
# ------------------------------------------------------------

st.header("Gap Detection")

if st.button("Detect Gaps"):
    detector = GapPattern(
        view_config={"x": x, "y": y},
        y_is_categorical=is_cat,
        min_gap_x_width=0.1 if is_cat else None,
        min_gap_area=0.05 if not is_cat else None,
    )
    detector.detect(df)
    
    if detector.detected is None:
        st.warning("No gaps detected.")
    else:
        st.success(f"Detected {detector.detected['total_gaps']} gap(s).")
        st.json(detector.detected)
        fig = detector.visualize(df, fig)

st.plotly_chart(fig, use_container_width=True)

