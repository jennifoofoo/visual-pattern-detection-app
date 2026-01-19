import streamlit as st
from core.app_utils import pattern_ui
import core.app_utils.app_handler as app_handler
from core.utils.demo_sampling import SamplingMode
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

# Keyboard shortcuts: R=Reset, F=Focus
st.markdown("""
<script>
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const btns = {r: 'reset_focus_btn', f: 'focus_btn'};
    const key = btns[e.key.toLowerCase()];
    if (key) {
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.toLowerCase().includes(key.split('_')[0]));
        if (btn && !btn.disabled) btn.click();
    }
});
</script>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MINIMAL HEADER (KEEP SIDEBAR TOGGLE VISIBLE)
# -------------------------------------------------
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
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
        </div>
        """,
        unsafe_allow_html=True,
    )




        
        # Sampling Strategy Selection (only shown when demo mode is enabled)
        
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
            demo_mode = st.checkbox(
            "Demo Mode", 
            value=True,
            help="Enable sampling for faster analysis. Choose a sampling strategy below."
        )
            sampling_mode = SamplingMode.FULL  # Default
            if demo_mode:
                sampling_options = {
                    "Minimal (fastest)": SamplingMode.MINIMAL,
                    "Balanced (√n)": SamplingMode.SQRT,
                    "Optimized (~70%)": SamplingMode.OPTIMIZED,
                    "Legacy (first-N)": SamplingMode.LEGACY,
                }
                
                selected_strategy = st.selectbox(
                    "Sampling Strategy:",
                    options=list(sampling_options.keys()),
                    index=1,  # Default to Balanced
                    help="""
                    **Minimal**: 1-2 traces per variant - ultra fast demos
                    **Balanced (√n)**: Keeps √n traces for frequent variants, all rare variants - recommended
                    **Optimized**: ~70% of data, preserves variant distribution - gentle reduction  
                    **Legacy**: First 100 cases - original sampling method
                    """,
                    key='sampling_strategy'
                )
                sampling_mode = sampling_options[selected_strategy]

            xes_path = st.text_input(
                "XES file path",
                value="data/Hospital_log.xes",
                disabled=demo_mode,
            )

            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
            if st.button("Load Data", type="primary"):
                app_handler.load_data_button(xes_path, demo_mode=demo_mode, sampling_mode=sampling_mode)
                st.session_state.ui_step = "config"
                st.rerun()

        st.divider()

        with st.expander("Chart Configuration", expanded=st.session_state.ui_step == "config"):
            x_axis, y_axis, dots_config_label = app_handler.get_chart_config_with_selectboxes()
            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
            if st.button("Re-plot Chart", type="primary"):
                if "df" in st.session_state:
                    app_handler.plot_chart_button(x_axis, y_axis, dots_config_label)
                    st.session_state.ui_step = "layers"
                    st.rerun()
                else:
                    st.warning("Load data first.")

        st.divider()

        with st.expander("Time Filter", expanded=st.session_state.get("chart_plotted", False)):
            app_handler.sidebar_time_filter()

        st.divider()

        with st.expander("Selection", expanded=st.session_state.get("chart_plotted", False)):
            app_handler.sidebar_focus_controls()

        st.divider()

        with st.expander("Pattern Layers", expanded=st.session_state.ui_step == "layers"):
            if st.session_state.get("chart_plotted", False):
                app_handler.sidebar_focus_mode_toggle()
                st.markdown("---")
                app_handler.sidebar_pattern_layer_controls()
            else:
                st.caption("Plot a chart first")

    # -----------------------------
    # MAIN CONTENT
    # -----------------------------
    if not st.session_state.get("data_loaded", False):
        return

    app_handler.display_chart()

    if st.session_state.get("chart_plotted", False):
        st.divider()
        pattern_ui.handle_pattern_detection()


# -------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------
if __name__ == "__main__":
    main()
