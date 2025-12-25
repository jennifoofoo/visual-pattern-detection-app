import streamlit as st
import core.app_utils.app_handler as app_handler
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Visual Pattern Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# LOAD GLOBAL CSS (SINGLE SOURCE OF TRUTH)
# -------------------------------------------------
def load_css():
    css_file = Path(__file__).parent / "style.css"
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------------------------------
# KEEP STREAMLIT HEADER FOR SIDEBAR MOUNT (INVISIBLE)
# -------------------------------------------------
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        visibility: hidden !important;
        height: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
def main():

    # -----------------------------
    # HEADER (NORMAL FLOW)
    # -----------------------------
    st.markdown(
        """
        <div class="main-header">
            <h1>Visual Pattern Detection in Dotted Charts</h1>
            <p class="main-subtitle">
                Process Mining Praktikum WS 25/26 · LMU München ·
                Tan Tai Bui, Jennifer Nikolovic, Anna Tsaan
            </p>
            <p class="main-description">
                Discover hidden patterns in your process data through interactive visualizations.
                Automatically detect temporal clusters, outliers, and gaps in event logs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # -----------------------------
    # STATE INIT
    # -----------------------------
    app_handler.init_state()
    st.session_state.setdefault("ui_step", "load")

    # -----------------------------
    # SIDEBAR (FIXED, ALWAYS VISIBLE)
    # -----------------------------
    with st.sidebar:
        with st.expander("Load Data", expanded=st.session_state.ui_step == "load"):
            xes_path = st.text_input(
                "Enter XES log file path",
                value="data/Hospital_log.xes",
            )

            demo_mode = st.checkbox(
                "🎬 Demo Mode",
                value=True,
                help="Enable for fast gap detection (samples to 100 cases).",
            )

            if st.button("Load Data", type="primary"):
                app_handler.load_data_button(xes_path, demo_mode=demo_mode)
                st.session_state.ui_step = "config"
                st.rerun()

        with st.expander("Chart Configuration", expanded=st.session_state.ui_step == "config"):
            st.caption("Chart is auto-plotted on data load. Customize here if needed.")
            x_axis, y_axis, dots_config_label = app_handler.get_chart_config_with_selectboxes()

            if st.button("Re-plot Chart", type="primary"):
                if "df" in st.session_state:
                    app_handler.plot_chart_button(x_axis, y_axis, dots_config_label)
                    st.session_state.ui_step = "layers"
                    st.rerun()
                else:
                    st.warning("Load data first.")

        with st.expander("Pattern Layers", expanded=st.session_state.ui_step == "layers"):
            if st.session_state.get("chart_plotted", False):
                app_handler.sidebar_pattern_layer_controls()
            else:
                st.caption("Plot a chart to enable pattern layers.")

        with st.expander("AI Description"):
            if st.button(
                "Describe Chart",
                disabled=not st.session_state.get("data_loaded", False),
            ):
                app_handler.ollama_description_button()

    # -----------------------------
    # MAIN CONTENT
    # -----------------------------
    if not st.session_state.get("data_loaded", False):
        return

    app_handler.display_chart()

    if st.session_state.get("chart_plotted", False):
        st.divider()
        app_handler.handle_pattern_detection()


# -------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------
if __name__ == "__main__":
    main()
