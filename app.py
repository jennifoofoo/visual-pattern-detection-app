import streamlit as st
import core.app_utils.app_handler as app_handler

# Configure page for better performance
st.set_page_config(
    page_title="Event Log Dotted Chart",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    st.title('Event Log Dotted Chart')

    app_handler.init_state()

    # File input
    # ToDo: add to app_handler.py and maybe add xes_path to session state
    xes_path = st.text_input(
        'Enter XES log file path:',
        value='data/Hospital_log.xes',
        key='xes_path_input'
    )

    # region Load XES File
    # Step 1: Load Data (Cached)
    col1, col2 = st.columns([1, 3])

    # Load Data Button
    with col1:
        if st.button('Load Data', type="primary"):
            app_handler.load_data_button(xes_path)
    # Show data status
    with col2:
        # Only show configuration if data is loaded
        if st.session_state.get('data_loaded', False):
            app_handler.show_xes_summary()
        else:
            st.info("Please load your XES file first")
            return
    # endregion


    # region Chart Configuration and Plotting
    # Step 2: Chart Configuration
    st.divider()
    st.subheader("Chart Configuration")

    # ToDo: is it ok to put those into session state?
    x_axis, y_axis, dots_config_label = app_handler.get_chart_config_with_selectboxes()

    if st.button('Plot Chart', type="primary"):
        if 'df' in st.session_state:
            app_handler.plot_chart_button(x_axis, y_axis, dots_config_label)
    # endregion


    # region Pattern Detection
    # Pattern Detection Section (only show if chart is plotted)
    if st.session_state.chart_plotted:
        st.divider()
        st.subheader("Pattern Detection")
        
        app_handler.handle_pattern_detection()
    # endregion
    
    # region Ollama
    # Ollama Description (moved to sidebar for better performance)
    with st.sidebar:
        st.subheader(" AI Description")
        if st.button("Describe Chart", disabled=not st.session_state.data_loaded):
            app_handler.ollama_description_button()
    # endregion

if __name__ == '__main__':
    main()
