import streamlit as st
from core.data_processing import load_xes_log
from core.evaluation.summary_generator import summarize_event_log
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP
from core.visualization.visualizer import plot_dotted_chart as plot_chart

from core.evaluation.ollama import OllamaEvaluator
from core.utils.demo_sampling import (
    sample_small_eventlog,
    sample_eventlog_variant_aware,
    get_sampling_mode_options,
    SamplingMode,
    SAMPLING_CONFIGS
)
from config.extended_pattern_matrix import is_pattern_meaningful, get_pattern_info

from core.app_utils.app_handler_pattern_detection import _detect_temporal_clusters, _detect_outliers, _detect_gaps, _detect_sequences
from core.app_utils.app_handler_pattern_detection import _is_any_pattern_detected, _get_detected_pattern_tabs, _display_pattern_tab



# Streamlit caching for performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_load_xes_log(xes_path):
    """Cached version of load_xes_log for better performance."""
    return load_xes_log(xes_path)

@st.cache_data(ttl=3600)
def generate_summary(df):
    """Cached summary generation."""
    return summarize_event_log(df)


# region Main Utils
def init_state():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'chart_plotted' not in st.session_state:
        st.session_state.chart_plotted = False
    # Initialize layer visibility flags (all layers visible by default)
    if 'visible_gap' not in st.session_state:
        st.session_state.visible_gap = True
    if 'visible_outlier' not in st.session_state:
        st.session_state.visible_outlier = True
    if 'visible_temporal_cluster' not in st.session_state:
        st.session_state.visible_temporal_cluster = True

def load_data_button(xes_path, demo_mode=False, sampling_mode: SamplingMode = SamplingMode.SQRT):
    try:
        with st.spinner(f"Loading {xes_path}..."):
            # Use cached loading
            df = cached_load_xes_log(xes_path)

        if df.empty:
            st.warning(
                "The log file was loaded but contains no events.")
            return

        # Demo Mode: Sample event log using variant-aware sampling
        sampling_stats = None
        if demo_mode and 'case_id' in df.columns and sampling_mode != SamplingMode.FULL:
            df_original = df
            df, sampling_stats = sample_eventlog_variant_aware(
                df,
                mode=sampling_mode,
                case_col='case_id',
                activity_col='activity',
                time_col='actual_time',
                random_state=42
            )
            
            # Build info message based on sampling mode
            config = SAMPLING_CONFIGS[sampling_mode]
            reduction_pct = (1 - sampling_stats['reduction_ratio']) * 100
            
            variant_info = ""
            if sampling_stats.get('variants_total'):
                variant_info = f" | {sampling_stats['variants_total']} variants"
            
            st.info(
                f"🎬 **{config.name}:** {sampling_stats['sampled_events']:,} events "
                f"({sampling_stats['sampled_traces']} traces{variant_info}) "
                f"from {sampling_stats['original_events']:,} original "
                f"({reduction_pct:.0f}% reduction). "
                f"{config.description}"
            )
            
            # Store sampling stats for later reference
            st.session_state.sampling_stats = sampling_stats

        # Store in session state
        st.session_state.df = df
        st.session_state.loaded_file = xes_path
        st.session_state.data_loaded = True
        st.session_state.chart_plotted = False  # Reset chart state

        # Generate summary (cached)
        st.session_state.summary = generate_summary(df)

        st.success(f"Log loaded: {len(df):,} events")
        st.rerun()  # Refresh to show data info

    except Exception as e:
        st.error(f"Error loading XES log: {e}")
        st.session_state.data_loaded = False

def show_xes_summary():
    df_info = st.session_state.df
    summary = st.session_state.summary

    # Show key metrics
    col2a, col2b, col2c, col2d = st.columns(4)
    with col2a:
        st.metric("Events", f"{len(df_info):,}")

    with col2b:
        st.metric("File", st.session_state.get(
            'loaded_file', '').split('/')[-1])

    with st.expander("Event Log Summary", expanded=False):
        for k, v in summary.items():
            st.write(f"**{k}:** {v}")

def get_chart_config_with_selectboxes():
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox('Select x-axis:', list(X_AXIS_COLUMN_MAP.keys()))
    with col2:
        y_axis = st.selectbox('Select y-axis:', list(Y_AXIS_COLUMN_MAP.keys()))
    with col3:
        dots_config_label = st.selectbox(
            'Select Dot Color:', list(DOTS_COLOR_MAP.keys()))
    return x_axis, y_axis, dots_config_label

def plot_chart_button(x_axis, y_axis, dots_config_label):
    df_base = st.session_state['df']

    # Determine the columns to plot
    x_col = X_AXIS_COLUMN_MAP[x_axis]
    y_col = Y_AXIS_COLUMN_MAP[y_axis]
    dots_config_col = DOTS_COLOR_MAP[dots_config_label]

    # Performance optimization: work with view instead of copy when possible
    df_selected = df_base

    # Check for missing values in the selected columns
    if df_selected[x_col].isnull().any() or df_selected[y_col].isnull().any():
        # Filter them out (make a copy if we haven't already)
        if df_selected is df_base:
            df_selected = df_base.copy()
        df_selected.dropna(subset=[x_col, y_col], inplace=True)
        if df_selected.empty:
            st.warning(
                "No valid data to plot after removing missing values.")
            return

    # Use all data (no sampling)
    total_points = len(df_selected)
    df_plot = df_selected

    # Configure hover data and colors
    hover_cols = ['activity', 'logical_relative', 'actual_time']
    color_col = dots_config_col

    # Generate the Plotly Scatter (Dotted Chart)
    with st.spinner("Rendering chart..."):
        fig = plot_chart(
            df=df_plot,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} points)",
            labels={x_col: x_axis, y_col: y_axis,
                    color_col: dots_config_label},
            hover_data=hover_cols
        )

        # Improve visual appearance
        fig.update_traces(marker=dict(size=5, opacity=0.8))

        # Layout settings
        fig.update_layout(
            showlegend=(
                color_col is not None and color_col != 'case_id'),
            hovermode='closest',
            template='plotly_white',
            yaxis=dict(autorange='reversed')
        )

        # Note: Visualization overlays will be added by display_chart()
        # Do not display chart here - it will be displayed persistently

    # Store the current plot configuration and figure
    st.session_state['current_plot_config'] = {
        'x_col': x_col,
        'y_col': y_col,
        'dots_config_col': dots_config_col,
        'x_axis_label': x_axis,
        'y_axis_label': y_axis,
        'dots_config_label': dots_config_label,
        # Store the plotted data (potentially sampled)
        'df_selected': df_plot,
        'total_points': total_points
    }

    # Store the figure and view config for pattern detection (with 3D matrix support)
    st.session_state['fig'] = fig
    st.session_state['view_config'] = {
        'x': x_col,
        'y': y_col,
        'color': dots_config_col  # Include color for 3D matrix consistency
    }
    st.session_state['chart_plotted'] = True

    st.success("Chart created successfully!")

    # Auto-detect all meaningful patterns after plotting
    auto_detect_patterns(x_col, y_col, dots_config_col, x_axis, y_axis, df_plot)

def display_chart():
    """Display the chart from session state (persistent across reruns)."""
    if not st.session_state.get('chart_plotted', False):
        return
        
    plot_config = st.session_state.get('current_plot_config', {})
    if not plot_config:
        return
    
    df_selected = plot_config['df_selected']
    x_col = plot_config['x_col']
    y_col = plot_config['y_col']
    dots_config_col = plot_config['dots_config_col']
    x_axis = plot_config['x_axis_label']
    y_axis = plot_config['y_axis_label']
    dots_config_label = plot_config['dots_config_label']
    total_points = plot_config['total_points']
    color_col = dots_config_col
    hover_cols = ['activity', 'logical_relative', 'actual_time']
    
    # Recreate the chart
    fig = plot_chart(
        df=df_selected,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} points)",
        labels={x_col: x_axis, y_col: y_axis, color_col: dots_config_label},
        hover_data=hover_cols
    )
    
    # Improve visual appearance
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    
    # Layout settings
    fig.update_layout(
        showlegend=(color_col is not None and color_col != 'case_id'),
        hovermode='closest',
        template='plotly_white',
        yaxis=dict(autorange='reversed')
    )
    
    # Add gap visualization if gaps were detected AND layer is visible
    if st.session_state.get('visible_gap', True):
        if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            fig = st.session_state['gap_detector'].visualize(df_selected, fig)
    
    # Add outlier visualization if detected AND layer is visible
    if st.session_state.get('visible_outlier', True):
        if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
            fig = st.session_state.outlier_pattern.visualize(df_selected, fig)
    
    # Add temporal cluster visualization if detected AND layer is visible
    if st.session_state.get('visible_temporal_cluster', True):
        if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
            fig = st.session_state.temporal_clusters.visualize(df_selected, fig)
    
    # Add sequence visualization if detected AND layer is visible
    if st.session_state.get('visible_sequence', True):
        if st.session_state.get('sequence_detected', False) and 'sequence_detector' in st.session_state:
            selected = st.session_state.get('selected_seq_patterns', [])
            fig = st.session_state.sequence_detector.visualize(df_selected, fig, selected_patterns=selected)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Update stored figure
    st.session_state['fig'] = fig

def handle_pattern_detection():
    """Display pattern summary with tabs for each detected pattern."""
    st.subheader("Pattern Summary")
    st.caption("Patterns are automatically detected after plotting. Toggle visibility in sidebar.")
    
    if not _is_any_pattern_detected():
        st.info("Patterns will appear here after chart is plotted.")
        return
    
    tabs_names, tabs_types = _get_detected_pattern_tabs()
    tabs = st.tabs(tabs_names)
    
    for tab, pattern_type in zip(tabs, tabs_types):
        with tab:
            _display_pattern_tab(pattern_type)

def sidebar_pattern_layer_controls():
    """Display pattern layer visibility controls in sidebar."""
    if not _is_any_pattern_detected():
        return
    
    st.subheader("Pattern Layers")
    st.caption("Toggle pattern visualizations on the chart")
    
    # Temporal Clusters
    if st.session_state.get('temporal_detected', False):
        _render_pattern_checkbox(
            "Temporal Clusters", 
            'visible_temporal_cluster', 
            'temporal_cluster_version',
            'checkbox_temporal_cluster_'
        )

    # Outlier Detection
    if st.session_state.get('outlier_detected', False):
        _render_pattern_checkbox(
            "Outlier Detection",
            'visible_outlier',
            'outlier_type_version',
            'checkbox_outlier_type_'
        )
    
    # Gap Detection
    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        _render_pattern_checkbox(
            "Gap Detection",
            'visible_gap',
            'gap_transition_version',
            'checkbox_gap_transition_'
        )

    # Sequence Detection
    if st.session_state.get('sequence_detected', False):
        _render_pattern_checkbox(
            "Sequence Detection",
            'visible_sequence',
            'sequence_pattern_version',
            'checkbox_sequence_pattern_'
        )
# endregion

# region Helpers

def _render_pattern_checkbox(label: str, visibility_key: str, version_key: str, checkbox_key_pattern: str):
    """
    Render a pattern visibility checkbox with change detection and sub-pattern sync.
    
    Args:
        label: Display label for the checkbox
        visibility_key: Session state key for visibility (e.g., 'visible_temporal_cluster')
        version_key: Session state key for widget version (e.g., 'temporal_cluster_version')
        checkbox_key_pattern: Pattern to match sub-checkbox keys (e.g., 'checkbox_temporal_cluster_')
    """
    if visibility_key not in st.session_state:
        st.session_state[visibility_key] = True
    
    prev_state = st.session_state[visibility_key]
    
    st.checkbox(
        label,
        key=visibility_key,
        help=f"Show/hide {label.lower()} visualization. Also toggles all sub-patterns."
    )
    
    # Detect change and sync sub-patterns
    if st.session_state[visibility_key] != prev_state:
        st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
        keys_to_delete = [k for k in list(st.session_state.keys()) if checkbox_key_pattern in k]
        for key in keys_to_delete:
            del st.session_state[key]
        st.rerun()

def auto_detect_patterns(x_col, y_col, color_col, x_axis_label, y_axis_label, df_selected):
    """Automatically detect all meaningful patterns after chart is plotted."""
    temporal_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'temporal_cluster_x')
    outlier_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'outlier')
    gap_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'gap')
    sequence_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'horizontal_sequence')

    with st.spinner("Auto-detecting patterns..."):
        if temporal_meaningful:
            _detect_temporal_clusters(x_col, y_col, df_selected)
        if outlier_meaningful:
            _detect_outliers()
        if gap_meaningful:
            _detect_gaps(x_col, y_col, df_selected)
        if sequence_meaningful or True:
            _detect_sequences()

def ollama_description_button():
    with st.spinner("Generating description..."):
                try:
                    evaluator = OllamaEvaluator(
                        model="qwen2.5:3b-instruct-q4_0")
                    df = st.session_state.df
                    summary = st.session_state.summary

                    summary_text = "\n".join(
                        [f"{k}: {v}" for k, v in summary.items()])
                    description = evaluator.describe_chart(summary_text, df)

                    st.write(description)
                except Exception as e:
                    st.error(f"Error generating description: {e}")
# endregion