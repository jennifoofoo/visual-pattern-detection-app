import streamlit as st
from core.data_processing import load_xes_log
from core.evaluation.summary_generator import summarize_event_log
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP
from core.visualization.visualizer import plot_dotted_chart as plot_chart
from core.detection import OutlierDetectionPattern, TemporalClusterPattern, CaseArrivalTrendPattern
from core.detection.gap_pattern import GapPattern
from core.evaluation.ollama import OllamaEvaluator
from core.utils.demo_sampling import sample_small_eventlog
from config.extended_pattern_matrix import is_pattern_meaningful, get_pattern_info


# Streamlit caching for performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_load_xes_log(xes_path):
    """Cached version of load_xes_log for better performance."""
    return load_xes_log(xes_path)

@st.cache_data(ttl=3600)
def generate_summary(df):
    """Cached summary generation."""
    return summarize_event_log(df)


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
    # Selection-based Focus: focus_df=None means full view, not None means focused subset
    if 'focus_df' not in st.session_state:
        st.session_state.focus_df = None

def load_data_button(xes_path, demo_mode=False):
    try:
        with st.spinner(f"Loading {xes_path}..."):
            # Use cached loading
            df = cached_load_xes_log(xes_path)

        if df.empty:
            st.warning(
                "The log file was loaded but contains no events.")
            return

        # Demo Mode: Sample event log for fast gap detection
        if demo_mode and 'case_id' in df.columns:
            df_original = df
            df = sample_small_eventlog(
                df,
                max_cases=100,
                max_events_per_case=30,
                time_col='actual_time',
                random_state=42
            )
            # Show info in UI
            st.info(
                f"🎬 **DEMO MODE Active:** Sampled to {len(df):,} events from {len(df_original):,} events "
                f"({df['case_id'].nunique()} cases) for fast gap detection. "
                f"Uncheck 'Demo Mode' to analyze full dataset."
            )

        # Raw loaded log - never use for pattern detection (use get_active_view_df instead)
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
    hover_cols = ['activity', 'event_index', 'actual_time']
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

    # Reset focus when a new chart is plotted
    st.session_state.focus_df = None

    st.success("Chart created successfully!")

    # Auto-detect all meaningful patterns after plotting
    auto_detect_patterns(
        x_col, y_col, dots_config_col, x_axis, y_axis,
        get_active_view_df(st.session_state['current_plot_config'])
    )

def _detect_temporal_clusters(x_col, y_col, df_selected):
    """Detect temporal clusters and store in session state."""
    try:
        detector = TemporalClusterPattern(
            df=df_selected,
            x_axis=x_col,
            y_axis=y_col,
            min_cluster_size=10
        )
        if detector.detect():
            st.session_state.temporal_clusters = detector
            st.session_state.temporal_detected = True
            st.session_state.visible_temporal_cluster = True
    except Exception as e:
        st.warning(f"Temporal cluster detection skipped: {str(e)}")


def _detect_outliers(df):
    """Detect outliers and store in session state."""
    try:
        outlier_pattern = OutlierDetectionPattern(
            df=df,
            view_config=st.session_state.view_config
        )
        if outlier_pattern.detect():
            st.session_state.outlier_pattern = outlier_pattern
            st.session_state.outlier_detected = True
            st.session_state.visible_outlier = True
    except Exception as e:
        st.warning(f"Outlier detection skipped: {str(e)}")


def _detect_gaps(x_col, y_col, df_selected, min_samples=None):
    """Detect gaps and store in session state.
    
    Args:
        x_col: X-axis column name
        y_col: Y-axis column name  
        df_selected: DataFrame with selected data
        min_samples: Minimum samples per transition (default: from session_state or 5)
    """
    try:
        # Get min_samples from session state or use default
        if min_samples is None:
            min_samples = st.session_state.get('gap_min_samples', 5)
        
        y_is_categorical = df_selected[y_col].nunique() <= 60
        view_config = {'x': x_col, 'y': y_col}
        gap_detector = GapPattern(
            view_config=view_config,
            y_is_categorical=y_is_categorical
        )
        gap_detector.MIN_SAMPLES_FOR_NORMALITY = min_samples
        gap_detector.detect(df_selected)
        
        if gap_detector.detected is not None and len(gap_detector.detected) > 0:
            st.session_state['gap_detector'] = gap_detector
            st.session_state.visible_gap = True
    except Exception as e:
        st.warning(f"Gap detection skipped: {str(e)}")


def _detect_case_arrival_trend(x_col, df_selected):
    """Detect case arrival trends and store in session state."""
    try:
        detector = CaseArrivalTrendPattern(
            view_config={'x': x_col},
            aggregation_period='W',
            min_periods=3
        )
        if detector.detect(df_selected):
            st.session_state['case_arrival_trend_detector'] = detector
            st.session_state.case_arrival_trend_detected = True
        elif detector.trend_result is not None:
            # Show in UI even if no significant trend
            st.session_state['case_arrival_trend_detector'] = detector
            st.session_state.case_arrival_trend_detected = True
    except Exception as e:
        st.warning(f"Case arrival trend detection skipped: {str(e)}")


def _get_detection_cache_key(x_col, y_col, color_col, df_len):
    """Generate a cache key for pattern detection."""
    return f"{x_col}_{y_col}_{color_col}_{df_len}"


def auto_detect_patterns(x_col, y_col, color_col, x_axis_label, y_axis_label, df_selected):
    """Automatically detect all meaningful patterns after chart is plotted."""
    is_focus_view = st.session_state.get('focus_df') is not None

    # Skip cache when focus is active (selections of same size must recompute)
    if not is_focus_view:
        cache_key = _get_detection_cache_key(x_col, y_col, color_col, len(df_selected))
        last_cache_key = st.session_state.get('_pattern_cache_key', '')
        if cache_key == last_cache_key:
            return
        st.session_state['_pattern_cache_key'] = cache_key
    
    # Clear old pattern results
    st.session_state.temporal_detected = False
    st.session_state.outlier_detected = False
    st.session_state.case_arrival_trend_detected = False
    if 'gap_detector' in st.session_state:
        del st.session_state['gap_detector']

    temporal_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'temporal_cluster_x')
    outlier_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'outlier')
    gap_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'gap')

    with st.spinner("Auto-detecting patterns..."):
        if temporal_meaningful:
            _detect_temporal_clusters(x_col, y_col, df_selected)
        if outlier_meaningful:
            _detect_outliers(df_selected)
        if gap_meaningful:
            _detect_gaps(x_col, y_col, df_selected)
        if x_col == 'actual_time':
            _detect_case_arrival_trend(x_col, df_selected)


def get_active_view_df(plot_config: dict):
    """
    The active view dataframe represents the explicitly selected
    sub-event-log. All pattern detection is recomputed on this view.
    This is not a visual zoom.
    """
    focus_df = st.session_state.get('focus_df')
    if focus_df is not None:
        return focus_df
    return plot_config['df_selected']


def display_chart():
    """Display the chart from session state (persistent across reruns)."""
    if not st.session_state.get('chart_plotted', False):
        return

    plot_config = st.session_state.get('current_plot_config', {})
    if not plot_config:
        return

    df_display = get_active_view_df(plot_config)
    is_focus_view = st.session_state.get('focus_df') is not None

    # Create _point_id for strict 1:1 mapping between plot points and dataframe rows
    df_display = df_display.reset_index(drop=True)
    df_display['_point_id'] = df_display.index

    x_col = plot_config['x_col']
    y_col = plot_config['y_col']
    dots_config_col = plot_config['dots_config_col']
    x_axis = plot_config['x_axis_label']
    y_axis = plot_config['y_axis_label']
    dots_config_label = plot_config['dots_config_label']
    total_points = len(df_display)
    color_col = dots_config_col
    hover_cols = ['activity', 'event_index', 'actual_time']

    # Build title with focus indicator
    if is_focus_view:
        full_count = len(plot_config['df_selected'])
        title = f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} of {full_count:,} points) [FOCUS VIEW]"
    else:
        title = f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} points)"

    # Recreate the chart with _point_id in customdata for selection
    fig = plot_chart(
        df=df_display,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        labels={x_col: x_axis, y_col: y_axis, color_col: dots_config_label},
        hover_data=hover_cols,
        custom_data=['_point_id']
    )

    # Improve visual appearance
    fig.update_traces(marker=dict(size=5, opacity=0.8))

    # Layout settings - enable lasso and box select
    fig.update_layout(
        showlegend=(color_col is not None and color_col != 'case_id'),
        hovermode='closest',
        template='plotly_white',
        yaxis=dict(autorange='reversed'),
        dragmode='lasso'  # Default to lasso selection
    )

    # Add gap visualization if gaps were detected AND layer is visible
    if st.session_state.get('visible_gap', True):
        if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            fig = st.session_state['gap_detector'].visualize(df_display, fig)

    # Add outlier visualization if detected AND layer is visible
    if st.session_state.get('visible_outlier', True):
        if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
            fig = st.session_state.outlier_pattern.visualize(df_display, fig)

    # Add temporal cluster visualization if detected AND layer is visible
    if st.session_state.get('visible_temporal_cluster', True):
        if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
            fig = st.session_state.temporal_clusters.visualize(df_display, fig)

    # Display chart with selection callback
    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key="main_chart"
    )

    # Update stored figure
    st.session_state['fig'] = fig

    # Selection-based Focus controls
    _display_focus_controls(selection, plot_config, df_display, is_focus_view)


def _reset_pattern_detection_state():
    """Clear all pattern detection results to force re-detection."""
    st.session_state.temporal_detected = False
    st.session_state.outlier_detected = False
    st.session_state.case_arrival_trend_detected = False
    if 'gap_detector' in st.session_state:
        del st.session_state['gap_detector']
    st.session_state['_pattern_cache_key'] = ''


def _display_focus_controls(selection, plot_config, df_display, is_focus_view):
    """Display Selection-based Focus controls below the chart."""
    selected_indices = []
    if not is_focus_view and selection and selection.selection:
        for pt in selection.selection.get("points", []):
            customdata = pt.get("customdata")
            if customdata and len(customdata) > 0:
                selected_indices.append(customdata[0])

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        if is_focus_view:
            full_count = len(plot_config['df_selected'])
            st.info(f"**Focus View** — Analyzing {len(df_display):,} of {full_count:,} points")
        elif selected_indices:
            st.caption(f"{len(selected_indices):,} points selected")
        else:
            st.caption("Lasso or box select points to focus")

    with col2:
        if st.button("Focus", disabled=(is_focus_view or not selected_indices), type="primary", key="focus_btn"):
            _apply_focus_selection(selected_indices, df_display, plot_config)

    with col3:
        if st.button("Reset", disabled=not is_focus_view, key="reset_focus_btn"):
            _reset_focus_view(plot_config)


def _apply_focus_selection(selected_point_ids, df_display, plot_config):
    """Filter to selected points and re-run pattern detection."""
    st.session_state.focus_df = df_display[df_display['_point_id'].isin(selected_point_ids)].copy()
    _reset_pattern_detection_state()
    auto_detect_patterns(
        plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'],
        plot_config['x_axis_label'], plot_config['y_axis_label'],
        get_active_view_df(plot_config)
    )
    st.rerun()


def _reset_focus_view(plot_config):
    """Reset to full view and re-run pattern detection."""
    st.session_state.focus_df = None
    _reset_pattern_detection_state()
    auto_detect_patterns(
        plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'],
        plot_config['x_axis_label'], plot_config['y_axis_label'],
        get_active_view_df(plot_config)
    )
    st.rerun()


def _is_any_pattern_detected() -> bool:
    """Check if any pattern has been detected."""
    return (
        st.session_state.get('temporal_detected', False) or
        st.session_state.get('outlier_detected', False) or
        st.session_state.get('case_arrival_trend_detected', False) or
        ('gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None)
    )


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


def sidebar_pattern_layer_controls():
    """Display pattern layer visibility controls in sidebar."""
    if not _is_any_pattern_detected():
        return

    st.markdown("##### Pattern Layers")

    if st.session_state.get('temporal_detected', False):
        _render_pattern_checkbox(
            "Temporal Clusters",
            'visible_temporal_cluster',
            'temporal_cluster_version',
            'checkbox_temporal_cluster_'
        )

    if st.session_state.get('outlier_detected', False):
        _render_pattern_checkbox(
            "Outlier Detection",
            'visible_outlier',
            'outlier_type_version',
            'checkbox_outlier_type_'
        )

    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        _render_pattern_checkbox(
            "Gap Detection",
            'visible_gap',
            'gap_transition_version',
            'checkbox_gap_transition_'
        )


# region Pattern Detection

def _get_detected_pattern_tabs() -> tuple:
    """
    Get lists of detected pattern names and types for tab creation.

    Returns:
        Tuple of (tab_names, tab_types) lists
    """
    tabs_names = []
    tabs_types = []

    if st.session_state.get('temporal_detected', False):
        tabs_names.append("Temporal Clusters")
        tabs_types.append('temporal')
    if st.session_state.get('outlier_detected', False):
        tabs_names.append("Outlier Detection")
        tabs_types.append('outlier')
    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        tabs_names.append("Gap Detection")
        tabs_types.append('gap')
    if st.session_state.get('case_arrival_trend_detected', False):
        tabs_names.append("Case Arrival Trend")
        tabs_types.append('case_arrival_trend')

    return tabs_names, tabs_types


def _display_pattern_tab(pattern_type: str):
    """Render the appropriate pattern tab content."""
    if pattern_type == 'temporal':
        display_temporal_cluster_tab()
    elif pattern_type == 'outlier':
        display_outlier_tab()
    elif pattern_type == 'gap':
        display_gap_tab()
    elif pattern_type == 'case_arrival_trend':
        display_case_arrival_trend_tab()


def handle_pattern_detection():
    """Display pattern summary with tabs for each detected pattern."""
    st.subheader("Pattern Summary")

    if not _is_any_pattern_detected():
        st.caption("Patterns will appear here after plotting")
        return

    tabs_names, tabs_types = _get_detected_pattern_tabs()
    tabs = st.tabs(tabs_names)

    for tab, pattern_type in zip(tabs, tabs_types):
        with tab:
            _display_pattern_tab(pattern_type)


def display_temporal_cluster_tab():
    """Display Temporal Cluster pattern details in tab."""
    if not (st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state):
        return

    detector = st.session_state.temporal_clusters
    summary = detector.get_summary()
    layer_visible = st.session_state.get('visible_temporal_cluster', True)

    if not layer_visible:
        st.caption("Hidden — enable in sidebar")

    st.metric("Clusters", summary['count'])

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        st.text(summary['details']['summary_text'])

    with subtab2:
        if hasattr(detector, 'clusters') and 'temporal_bursts' in detector.clusters:
            cluster_list = detector.clusters['temporal_bursts']
            selected_clusters = list_to_multicheckbox(
                cluster_list,
                title="Select Clusters",
                key_prefix="temporal_cluster"
            )
            st.session_state['selected_temporal_clusters'] = selected_clusters


def display_outlier_tab():
    """Display Outlier Detection pattern details in tab."""
    if not (st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state):
        return

    outlier_pattern = st.session_state.outlier_pattern
    summary = outlier_pattern.get_summary()
    layer_visible = st.session_state.get('visible_outlier', True)

    if not layer_visible:
        st.caption("Hidden — enable in sidebar")

    stats = summary['details'].get('statistics', {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Outliers", summary['count'])
    with col2:
        st.metric("Percentage", f"{stats.get('outlier_percentage', 0):.1f}%")

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        if summary['details'].get('outlier_details'):
            for outlier_type, details in summary['details']['outlier_details'].items():
                st.write(f"- {outlier_type.replace('_', ' ').title()}: {details['count']} ({details['percentage']:.1f}%)")

    with subtab2:
        outlier_details = summary['details'].get('outlier_details', {})
        if outlier_details:
            outlier_types_dict = {
                f"{otype.replace('_', ' ').title()} ({details['count']})": otype
                for otype, details in outlier_details.items()
                if details['count'] > 0
            }
            selected_types = dict_to_multicheckbox(
                outlier_types_dict,
                title="Select Types",
                key_prefix="outlier_type"
            )
            st.session_state['selected_outlier_types'] = selected_types


def display_gap_tab():
    """Display Gap Detection pattern details in tab."""
    if not ('gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None):
        return

    gap_detector = st.session_state['gap_detector']
    summary = gap_detector.get_summary()
    details = summary['details']
    layer_visible = st.session_state.get('visible_gap', True)

    # Header with metrics and settings
    col1, col2, col3 = st.columns([1, 1, 0.3])
    with col1:
        st.metric("Gaps", summary['count'])
    with col2:
        st.metric("Transitions", details['transitions_with_anomalies'])
    with col3:
        with st.popover("⚙️"):
            current_min_samples = st.session_state.get('gap_min_samples', 5)
            min_samples = st.number_input(
                "Min samples",
                min_value=3, max_value=20, value=current_min_samples, step=1,
                key="gap_min_samples_tab_input"
            )
            if min_samples != current_min_samples:
                if st.button("Apply", use_container_width=True, type="primary", key="gap_redetect_tab"):
                    st.session_state['gap_min_samples'] = min_samples
                    plot_config = st.session_state.get('current_plot_config', {})
                    if plot_config:
                        _detect_gaps(plot_config['x_col'], plot_config['y_col'], get_active_view_df(plot_config), min_samples)
                        st.rerun()

    if not layer_visible:
        st.caption("Hidden — enable in sidebar")

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        trans_stats = details.get('transition_stats', {})
        for trans, stats in list(trans_stats.items())[:5]:
            st.write(f"- {trans}: {stats['count']} gaps, threshold {stats['threshold']/86400:.1f}d")

    with subtab2:
        trans_stats = details.get('transition_stats', {})
        if trans_stats:
            transition_dict = {
                f"{trans} ({stats['count']})": trans
                for trans, stats in trans_stats.items()
            }
            selected_transitions = dict_to_multicheckbox(
                transition_dict,
                title="Select Transitions",
                key_prefix="gap_transition"
            )
            st.session_state['selected_gap_transitions'] = selected_transitions


def display_case_arrival_trend_tab():
    """Display Case Arrival Trend pattern details in tab."""
    if not st.session_state.get('case_arrival_trend_detected', False):
        return
    if 'case_arrival_trend_detector' not in st.session_state:
        return

    detector = st.session_state['case_arrival_trend_detector']
    summary = detector.get_summary()

    direction = summary.get('direction', 'no_trend')
    slope_pct = summary.get('slope_percent', 0)
    p_value = summary.get('p_value', 1.0)
    total_cases = summary.get('total_cases', 0)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cases", total_cases)
    with col2:
        slope_str = f"{slope_pct:+.1f}%" if abs(slope_pct) >= 0.1 else "~0%"
        st.metric("Change/week", slope_str)
    with col3:
        st.metric("p-value", f"{p_value:.4f}")

    # Direction indicator
    if direction == 'increasing':
        st.success(f"↗ Increasing trend")
    elif direction == 'decreasing':
        st.error(f"↘ Decreasing trend")
    elif direction == 'stable':
        st.info("→ Stable")
    else:
        st.caption("No significant trend")

    st.caption("Measures whether new cases are arriving faster or slower over time.")


# ========== HELPER FUNCTIONS FOR SUB-PATTERN SELECTION ==========

def get_parent_visibility(key_prefix: str) -> bool:
    """
    Get the visibility state of the parent pattern from sidebar.
    
    Args:
        key_prefix: Pattern key prefix (e.g. 'temporal_cluster', 'outlier_type', 'gap_transition')
    
    Returns:
        True if parent pattern is visible, False otherwise
    """
    prefix_to_sidebar = {
        'temporal_cluster': 'visible_temporal_cluster',
        'outlier_type': 'visible_outlier',
        'gap_transition': 'visible_gap'
    }
    
    sidebar_key = prefix_to_sidebar.get(key_prefix)
    if sidebar_key:
        return st.session_state.get(sidebar_key, True)
    return True


def sync_sidebar_checkbox(key_prefix: str, value: bool):
    """
    Synchronize sidebar checkbox with tab selection.
    
    Args:
        key_prefix: Pattern key prefix (e.g. 'temporal_cluster', 'outlier_type', 'gap_transition')
        value: True to enable, False to disable
    """
    print(f"🟡 SYNC: sync_sidebar_checkbox('{key_prefix}', {value})")
    
    # Map key_prefix to sidebar session state key
    prefix_to_sidebar = {
        'temporal_cluster': 'visible_temporal_cluster',
        'outlier_type': 'visible_outlier',
        'gap_transition': 'visible_gap'
    }
    
    sidebar_key = prefix_to_sidebar.get(key_prefix)
    if sidebar_key:
        old_value = st.session_state.get(sidebar_key)
        st.session_state[sidebar_key] = value
        print(f"🟡 SYNC: Changed {sidebar_key} from {old_value} to {value}")


def deselect_all_subpatterns(pattern_type: str):
    """
    Deselect all sub-patterns when sidebar checkbox is unchecked.
    
    Args:
        pattern_type: 'temporal', 'outlier', or 'gap'
    """
    if pattern_type == 'temporal':
        for key in list(st.session_state.keys()):
            if key.startswith('list_checkbox_temporal_cluster_'):
                st.session_state[key] = False
    elif pattern_type == 'outlier':
        for key in list(st.session_state.keys()):
            if key.startswith('dict_checkbox_outlier_type_'):
                st.session_state[key] = False
    elif pattern_type == 'gap':
        for key in list(st.session_state.keys()):
            if key.startswith('dict_checkbox_gap_transition_'):
                st.session_state[key] = False


def select_all_subpatterns(pattern_type: str):
    """
    Select all sub-patterns when sidebar checkbox is checked.
    
    Args:
        pattern_type: 'temporal', 'outlier', or 'gap'
    """
    if pattern_type == 'temporal':
        for key in list(st.session_state.keys()):
            if key.startswith('list_checkbox_temporal_cluster_'):
                st.session_state[key] = True
    elif pattern_type == 'outlier':
        for key in list(st.session_state.keys()):
            if key.startswith('dict_checkbox_outlier_type_'):
                st.session_state[key] = True
    elif pattern_type == 'gap':
        for key in list(st.session_state.keys()):
            if key.startswith('dict_checkbox_gap_transition_'):
                st.session_state[key] = True


def list_to_multicheckbox(item_list: list, title: str = "Select Items", key_prefix: str = "item") -> list:
    """
    Renders a Streamlit multi-checkbox interface based on a Python list.
    
    Args:
        item_list: The input list of items to be displayed as checkboxes.
        title: The title to display above the group of checkboxes.
        key_prefix: Prefix for unique checkbox keys.
    
    Returns:
        List containing only the items selected by the user.
    """
    if not item_list:
        return []

    selected_items = []
    parent_visible = get_parent_visibility(key_prefix)

    if not parent_visible:
        st.caption("Enable in sidebar to configure")
        return []

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("All", key=f"{key_prefix}_select_all", use_container_width=True):
                for index in range(len(item_list)):
                    st.session_state[f"list_checkbox_{key_prefix}_{index}"] = True
                sync_sidebar_checkbox(key_prefix, True)
                st.rerun()
        with col_b:
            if st.button("None", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for index in range(len(item_list)):
                    st.session_state[f"list_checkbox_{key_prefix}_{index}"] = False
                sync_sidebar_checkbox(key_prefix, False)
                st.rerun()

        for index, item in enumerate(item_list):
            state_key = f"list_checkbox_{key_prefix}_{index}"
            if state_key not in st.session_state:
                st.session_state[state_key] = True
            checked = st.checkbox(str(item), key=state_key)
            if checked:
                selected_items.append(item)

    return selected_items


def dict_to_multicheckbox(data_dict: dict, title: str = "Select Items", key_prefix: str = "dict_item") -> list:
    """
    Renders a Streamlit multi-checkbox interface based on a Python dictionary.
    
    Args:
        data_dict: The input dictionary where keys are display labels and values are actual identifiers.
        title: The title to display above the group of checkboxes.
        key_prefix: Prefix for unique checkbox keys.
    
    Returns:
        List containing only the selected dictionary values.
    """
    if not data_dict:
        return []

    selected_items = []
    parent_visible = get_parent_visibility(key_prefix)

    if not parent_visible:
        st.caption("Enable in sidebar to configure")
        return []

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("All", key=f"{key_prefix}_select_all", use_container_width=True):
                for key in data_dict.keys():
                    st.session_state[f"dict_checkbox_{key_prefix}_{key}"] = True
                sync_sidebar_checkbox(key_prefix, True)
                st.rerun()
        with col_b:
            if st.button("None", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for key in data_dict.keys():
                    st.session_state[f"dict_checkbox_{key_prefix}_{key}"] = False
                sync_sidebar_checkbox(key_prefix, False)
                st.rerun()

        for key, value in data_dict.items():
            state_key = f"dict_checkbox_{key_prefix}_{key}"
            if state_key not in st.session_state:
                st.session_state[state_key] = True
            checked = st.checkbox(key, key=state_key)
            if checked:
                selected_items.append(value)

    return selected_items

                        
def ollama_description_button():
    with st.spinner("Generating description..."):
        try:
            plot_config = st.session_state.get('current_plot_config', {})
            if not plot_config:
                st.warning("Please plot a chart first")
                return
            df = get_active_view_df(plot_config)
            summary = summarize_event_log(df)
            summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
            evaluator = OllamaEvaluator(model="qwen2.5:3b-instruct-q4_0")
            description = evaluator.describe_chart(summary_text, df)
            st.write(description)
        except Exception as e:
            st.error(f"Error generating description: {e}")