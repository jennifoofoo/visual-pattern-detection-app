import streamlit as st
import core.app_utils.app_handler as app_handler
from core.utils.demo_sampling import SamplingMode

# Configure page for better performance
st.set_page_config(
    page_title="Event Log Dotted Chart",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar opens after plotting chart
)


def main():
    st.title('Event Log Dotted Chart')

    app_handler.init_state()

    # File input
    xes_path = st.text_input(
        'Enter XES log file path:',
        value='data/Hospital_log.xes',
        # value='data\Sepsis Cases - Event Log.xes\Sepsis Cases - Event Log.xes',
        key='xes_path_input'
    )

    # region Load XES File
    # Step 1: Load Data (Cached)
    col1, col2 = st.columns([1, 3])

    # Load Data Button
    with col1:
        # Demo Mode Checkbox
        demo_mode = st.checkbox(
            "🎬 Demo Mode", 
            value=True,
            help="Enable sampling for faster analysis. Choose a sampling strategy below."
        )
        
        # Sampling Strategy Selection (only shown when demo mode is enabled)
        sampling_mode = SamplingMode.FULL  # Default
        if demo_mode:
            sampling_options = {
                "⚡ Minimal (fastest)": SamplingMode.MINIMAL,
                "📊 Balanced (√n)": SamplingMode.SQRT,
                "🎯 Optimized (~70%)": SamplingMode.OPTIMIZED,
                "📁 Legacy (first-N)": SamplingMode.LEGACY,
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
        
        if st.button('Load Data', type="primary"):
            app_handler.load_data_button(xes_path, demo_mode=demo_mode, sampling_mode=sampling_mode)

    # Show data status
    with col2:
        if st.session_state.data_loaded:
            app_handler.show_xes_summary()
        else:
            st.info("Please load your XES file first")
            return
    # endregion

    # region Chart Configuration and Plotting
    # Step 2: Chart Configuration
    st.divider()
    st.subheader("Chart Configuration")

    x_axis, y_axis, dots_config_label = app_handler.get_chart_config_with_selectboxes()

    if st.button('Plot Chart', type="primary"):
        if 'df' in st.session_state:
            app_handler.plot_chart_button(x_axis, y_axis, dots_config_label)
    
    # Display chart persistently (survives reruns from pattern detection)
    app_handler.display_chart()
            # endregion

    # region Pattern Detection
    # Pattern Detection Section (only show if chart is plotted)
    if st.session_state.chart_plotted:
        st.divider()
        app_handler.handle_pattern_detection()
    # endregion
    
    # region Sidebar
    with st.sidebar:
        # Pattern Layer Controls
        if st.session_state.chart_plotted:
            app_handler.sidebar_pattern_layer_controls()
        
        st.divider()
        
        # Ollama Description
        st.subheader(" AI Description")
        if st.button("Describe Chart", disabled=not st.session_state.data_loaded):
            app_handler.ollama_description_button()
    # endregion


if __name__ == '__main__':
    main()
