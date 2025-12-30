import streamlit as st
from core.data_processing import load_xes_log
from core.evaluation.summary_generator import summarize_event_log
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP
from core.visualization.visualizer import plot_dotted_chart as plot_chart
from core.detection import OutlierDetectionPattern, TemporalClusterPattern, TrendPattern, CaseArrivalTrendPattern
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

    st.success("Chart created successfully!")

    # Auto-detect all meaningful patterns after plotting
    auto_detect_patterns(x_col, y_col, dots_config_col, x_axis, y_axis, df_plot)

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


def _detect_outliers():
    """Detect outliers and store in session state."""
    try:
        outlier_pattern = OutlierDetectionPattern(
            df=st.session_state.df,
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


def _detect_trend(x_col, y_col, df_selected):
    """Detect trends and store in session state."""
    try:
        view_config = {'x': x_col, 'y': y_col}
        trend_detector = TrendPattern(
            view_config=view_config,
            aggregation_period='D',  # Daily aggregation
            min_periods=3,  # Lowered for demo data
            analyze_per_category=True
        )
        if trend_detector.detect(df_selected):
            st.session_state['trend_detector'] = trend_detector
            st.session_state.trend_detected = True
            st.session_state.visible_trend = True
        else:
            # Still show in UI even if no significant trend
            if trend_detector.global_trend is not None:
                st.session_state['trend_detector'] = trend_detector
                st.session_state.trend_detected = True
                st.session_state.visible_trend = True
    except Exception as e:
        st.warning(f"Trend detection skipped: {str(e)}")
        import traceback
        print(f"Trend detection error: {traceback.format_exc()}")


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
    # Check if we already detected patterns for this exact configuration
    cache_key = _get_detection_cache_key(x_col, y_col, color_col, len(df_selected))
    last_cache_key = st.session_state.get('_pattern_cache_key', '')
    
    if cache_key == last_cache_key:
        # Patterns already detected for this config, skip re-detection
        return
    
    # Store new cache key
    st.session_state['_pattern_cache_key'] = cache_key
    
    # Clear old pattern results
    st.session_state.temporal_detected = False
    st.session_state.outlier_detected = False
    st.session_state.trend_detected = False
    st.session_state.case_arrival_trend_detected = False
    if 'gap_detector' in st.session_state:
        del st.session_state['gap_detector']

    temporal_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'temporal_cluster_x')
    outlier_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'outlier')
    gap_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'gap')
    trend_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'trend')

    with st.spinner("Auto-detecting patterns..."):
        if temporal_meaningful:
            _detect_temporal_clusters(x_col, y_col, df_selected)
        if outlier_meaningful:
            _detect_outliers()
        if gap_meaningful:
            _detect_gaps(x_col, y_col, df_selected)
        if trend_meaningful:
            _detect_trend(x_col, y_col, df_selected)
        # Case arrival trend: only requires actual_time on X-axis (case-level, Y-independent)
        if x_col == 'actual_time':
            _detect_case_arrival_trend(x_col, df_selected)


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
    hover_cols = ['activity', 'event_index', 'actual_time']
    
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
    
    # Add trend visualization if detected AND layer is visible
    if st.session_state.get('visible_trend', True):
        if st.session_state.get('trend_detected', False) and 'trend_detector' in st.session_state:
            fig = st.session_state['trend_detector'].visualize(df_selected, fig)

    st.plotly_chart(fig, use_container_width=True)
    
    # Update stored figure
    st.session_state['fig'] = fig


def _is_any_pattern_detected() -> bool:
    """Check if any pattern has been detected."""
    return (
        st.session_state.get('temporal_detected', False) or
        st.session_state.get('outlier_detected', False) or
        st.session_state.get('trend_detected', False) or
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
    
    # Trend Detection
    if st.session_state.get('trend_detected', False):
        _render_pattern_checkbox(
            "Trend Detection",
            'visible_trend',
            'trend_version',
            'checkbox_trend_'
        )



# region Pattern Detection
def handle_temporal_cluster_detection_logic(x_col, y_col, x_axis_label, y_axis_label, df_selected):
    """Execute temporal cluster detection logic."""
    if x_col and y_col and df_selected is not None:
        with st.spinner("Detecting temporal patterns..."):
            detector = TemporalClusterPattern(
                df=df_selected,
                x_axis=x_col,
                y_axis=y_col,
                min_cluster_size=10
            )

            if detector.detect():
                st.session_state.temporal_clusters = detector
                st.session_state.temporal_detected = True
            else:
                st.session_state.temporal_detected = False
                st.info(f"No meaningful temporal patterns for {y_axis_label} × {x_axis_label}")
    else:
        st.warning("Please plot a chart first")

def handle_outlier_detection_logic():
    """Execute outlier detection logic."""
    with st.spinner("Analyzing outliers..."):
        try:
            # Use original data for outlier detection (not sampled data)
            outlier_pattern = OutlierDetectionPattern(
                df=st.session_state.df,  # Use full dataset
                view_config=st.session_state.view_config
            )
            if outlier_pattern.detect():
                # Store outlier results in session state
                st.session_state.outlier_pattern = outlier_pattern
                st.session_state.outlier_detected = True
            else:
                st.session_state.outlier_detected = False
                st.info("No significant outliers detected!")
        except Exception as e:
            st.session_state.outlier_detected = False
            st.error(f"Error during outlier detection: {str(e)}")

# handle_gap_detection_logic REMOVED - use _detect_gaps instead (avoids code duplication)

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
    if st.session_state.get('trend_detected', False):
        tabs_names.append("Trend Detection")
        tabs_types.append('trend')
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
    elif pattern_type == 'trend':
        display_trend_tab()
    elif pattern_type == 'case_arrival_trend':
        display_case_arrival_trend_tab()


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


def display_temporal_cluster_tab():
    """Display Temporal Cluster pattern details in tab with individual cluster selection."""
    if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
        detector = st.session_state.temporal_clusters
        summary = detector.get_summary()
        
        # Layer visibility is now controlled by sidebar
        layer_visible = st.session_state.get('visible_temporal_cluster', True)
        
        if not layer_visible:
            st.info("Layer hidden - toggle in sidebar to show on chart")
        
        st.success(f"{summary['count']} clusters detected")
        
        # === NESTED TABS FOR OVERVIEW AND CLUSTER CONTROL ===
        subtab1, subtab2 = st.tabs(["Overview", "Cluster Control"])
        
        with subtab1:
            st.text(summary['details']['summary_text'])
        
        with subtab2:
            # Individual cluster selection
            # Get cluster data (assuming temporal_bursts exists in detector.clusters)
            if hasattr(detector, 'clusters') and 'temporal_bursts' in detector.clusters:
                cluster_list = detector.clusters['temporal_bursts']
                selected_clusters = list_to_multicheckbox(
                    cluster_list, 
                    title="Select Clusters to Display",
                    key_prefix="temporal_cluster"
                )
                
                # Store selected clusters in session state for visualization
                st.session_state['selected_temporal_clusters'] = selected_clusters
                
                if selected_clusters:
                    st.success(f"✅ {len(selected_clusters)} of {len(cluster_list)} clusters selected")


def display_outlier_tab():
    """Display Outlier Detection pattern details in tab with individual outlier type selection."""
    if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
        outlier_pattern = st.session_state.outlier_pattern
        summary = outlier_pattern.get_summary()
        
        # Layer visibility is now controlled by sidebar
        layer_visible = st.session_state.get('visible_outlier', True)
        
        if not layer_visible:
            st.info("Layer hidden - toggle in sidebar to show on chart")
        
        stats = summary['details'].get('statistics', {})
        st.success(f"{summary['count']} outliers detected ({stats.get('outlier_percentage', 0):.1f}%)")
        
        # === NESTED TABS FOR OVERVIEW AND OUTLIER TYPE CONTROL ===
        subtab1, subtab2 = st.tabs(["Overview", "Outlier Type Control"])
        
        with subtab1:
            if summary['details'].get('outlier_details'):
                for outlier_type, details in summary['details']['outlier_details'].items():
                    st.write(f"- {outlier_type.replace('_', ' ').title()}: {details['count']} ({details['percentage']:.1f}%)")
        
        with subtab2:
            # Individual outlier type selection
            # Get outlier types
            outlier_details = summary['details'].get('outlier_details', {})
            if outlier_details:
                # Create a dict with outlier types and their counts
                outlier_types_dict = {
                    f"{otype.replace('_', ' ').title()} ({details['count']})": otype 
                    for otype, details in outlier_details.items() 
                    if details['count'] > 0
                }
                
                selected_types = dict_to_multicheckbox(
                    outlier_types_dict,
                    title="Select Outlier Types to Display",
                    key_prefix="outlier_type"
                )
                
                # Store selected types in session state
                st.session_state['selected_outlier_types'] = selected_types
                
                if selected_types:
                    st.success(f"✅ {len(selected_types)} of {len(outlier_types_dict)} outlier types selected")
            else:
                st.info("No outlier type details available")


def display_gap_tab():
    """Display Gap Detection pattern details in tab with individual transition selection."""
    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        gap_detector = st.session_state['gap_detector']
        summary = gap_detector.get_summary()
        details = summary['details']
        
        # Settings and status in header
        col_info, col_settings = st.columns([0.85, 0.15])
        with col_info:
            layer_visible = st.session_state.get('visible_gap', True)
            if not layer_visible:
                st.info("Layer hidden - toggle in sidebar to show on chart")
            else:
                st.success(f"{summary['count']} gaps detected across {details['transitions_with_anomalies']} transitions")
        with col_settings:
            with st.popover("⚙️"):
                st.write("**Settings**")
                current_min_samples = st.session_state.get('gap_min_samples', 5)
                min_samples = st.number_input(
                    "Min samples per transition",
                    min_value=3,
                    max_value=20,
                    value=current_min_samples,
                    step=1,
                    key="gap_min_samples_tab_input",
                    help="Transitions with fewer samples are skipped"
                )
                if min_samples != current_min_samples:
                    if st.button("Apply & Re-detect", use_container_width=True, type="primary", key="gap_redetect_tab"):
                        st.session_state['gap_min_samples'] = min_samples
                        plot_config = st.session_state.get('current_plot_config', {})
                        if plot_config:
                            df_selected = plot_config['df_selected']
                            x_col = plot_config['x_col']
                            y_col = plot_config['y_col']
                            _detect_gaps(x_col, y_col, df_selected, min_samples)
                            st.rerun()
        
        # === NESTED TABS FOR OVERVIEW AND TRANSITION CONTROL ===
        subtab1, subtab2 = st.tabs(["Overview", "Transition Control"])
        
        with subtab1:
            st.write("**Top Transitions with Anomalies:**")
            trans_stats = details.get('transition_stats', {})
            for trans, stats in list(trans_stats.items())[:5]:
                st.write(f"- {trans}: {stats['count']} occurrences, threshold: {stats['threshold']/86400:.1f} days")
            
            st.write("**Top 10 Abnormal Gaps by Severity:**")
            abnormal_gaps = sorted(details['abnormal_gaps'], key=lambda x: x.get('severity', 0), reverse=True)[:10]
            for i, gap in enumerate(abnormal_gaps, 1):
                duration_days = gap['duration'] / 86400
                threshold_days = gap['threshold'] / 86400
                st.write(f"{i}. {gap['transition']} - Duration: {duration_days:.1f}d, Threshold: {threshold_days:.1f}d, Severity: {gap['severity']:.2f}x")
        
        with subtab2:
            # Individual transition selection
            # Get transitions with anomalies
            trans_stats = details.get('transition_stats', {})
            if trans_stats:
                # Create a dict with transitions and their anomaly counts
                transition_dict = {
                    f"{trans} ({stats['count']} anomalies)": trans 
                    for trans, stats in trans_stats.items()
                }
                
                selected_transitions = dict_to_multicheckbox(
                    transition_dict,
                    title="Select Transitions to Display",
                    key_prefix="gap_transition"
                )
                
                # Store selected transitions in session state
                st.session_state['selected_gap_transitions'] = selected_transitions
                
                if selected_transitions:
                    st.success(f"{len(selected_transitions)} of {len(transition_dict)} transitions selected")
            else:
                st.info("No transition stats available")


def display_trend_tab():
    """Display Trend Detection pattern details in tab."""
    if st.session_state.get('trend_detected', False) and 'trend_detector' in st.session_state:
        detector = st.session_state['trend_detector']
        summary = detector.get_summary()
        
        # Layer visibility status
        layer_visible = st.session_state.get('visible_trend', True)
        if not layer_visible:
            st.info("Layer hidden - toggle in sidebar to show on chart")
        
        # Global trend info
        global_trend = summary.get('global_trend', 'no_trend')
        global_change = summary.get('global_change', 0)
        p_value = summary.get('p_value', 1.0)
        
        # Trend icon
        if global_trend == 'increasing':
            trend_icon = '↗'
            trend_color = 'green'
        elif global_trend == 'decreasing':
            trend_icon = '↘'
            trend_color = 'red'
        elif global_trend == 'stable':
            trend_icon = '→'
            trend_color = 'blue'
        else:  # no_trend
            trend_icon = '−'
            trend_color = 'gray'
        
        # Format slope - use more precision for small values
        def format_slope(val):
            if abs(val) < 0.1:
                return f"{val:+.3f}%" if val != 0 else "~0%"
            return f"{val:+.1f}%"
        
        # Display summary
        if global_trend == 'increasing':
            st.success(f"{trend_icon} {global_trend.upper()}: {format_slope(global_change)} per period (p={p_value:.4f})")
        elif global_trend == 'decreasing':
            st.error(f"{trend_icon} {global_trend.upper()}: {format_slope(global_change)} per period (p={p_value:.4f})")
        elif global_trend == 'stable':
            st.info(f"{trend_icon} STABLE: No practical trend (change < 0.5% per period)")
        else:
            st.warning("No significant trend detected")
        
        # Nested tabs for Overview and Details
        subtab1, subtab2 = st.tabs(["Overview", "Category Trends"])
        
        with subtab1:
            st.write("**Global Event Frequency Trend:**")
            st.write(f"- Direction: {global_trend}")
            st.write(f"- Change rate: {format_slope(global_change)} per period")
            st.write(f"- Statistical significance: p = {p_value:.4f}")
            
            if p_value <= 0.05:
                st.write("- Trend is **statistically significant** (p ≤ 0.05)")
            else:
                st.write("- Trend is **not statistically significant** (p > 0.05)")
        
        with subtab2:
            # Show per-category trends (only for meaningful categories like activity/resource)
            details = summary.get('details', {})
            category_trends = details.get('category_trends', [])
            
            if category_trends:
                st.write("**Significant Category Trends:**")
                for cat, info in category_trends:
                    trend_dir = info.get('trend', 'no_trend')
                    slope_pct = info.get('slope_percent', 0)
                    cat_p_value = info.get('p_value', 1.0)
                    
                    if trend_dir == 'increasing':
                        st.write(f"- ↗ **{cat}**: {format_slope(slope_pct)} (p={cat_p_value:.4f})")
                    elif trend_dir == 'decreasing':
                        st.write(f"- ↘ **{cat}**: {format_slope(slope_pct)} (p={cat_p_value:.4f})")
            else:
                st.caption("Category trends only available for Activity or Resource axes")


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

    # Format slope
    slope_str = f"{slope_pct:+.1f}%" if abs(slope_pct) >= 0.1 else "~0%"

    # Result summary
    if direction == 'increasing':
        st.success(f"↗ INCREASING: {slope_str} per week (p={p_value:.4f})")
    elif direction == 'decreasing':
        st.error(f"↘ DECREASING: {slope_str} per week (p={p_value:.4f})")
    elif direction == 'stable':
        st.info("→ STABLE: No practical trend (< 0.5% change per week)")
    else:
        st.warning("No significant trend detected")

    # Explanation
    st.markdown("---")
    st.write("**What is Case Arrival Trend?**")
    st.write("Measures whether new cases are arriving faster or slower over time.")
    st.write(f"- **Definition:** Case arrival = min(actual_time) per case_id")
    st.write(f"- **Total cases:** {total_cases}")
    st.write(f"- **Statistical test:** Mann-Kendall (p = {p_value:.4f})")

    st.caption("Unlike event-frequency trends, this focuses on CASE STARTS only.")


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
        st.info("The input list is empty. No checkboxes to display.")
        return []
    
    selected_items = []
    
    # Use a container for visual grouping
    with st.container(border=True):
        st.write(f"**{title}**")
        st.caption("⚠️ Changes will update the chart automatically")
        
        # Add "Select All" / "Deselect All" functionality
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select All", key=f"{key_prefix}_select_all", use_container_width=True):
                for index in range(len(item_list)):
                    st.session_state[f"list_checkbox_{key_prefix}_{index}"] = True
                # Turn ON parent sidebar checkbox when selecting all
                sync_sidebar_checkbox(key_prefix, True)
                st.rerun()
        with col_b:
            if st.button("Deselect All", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for index in range(len(item_list)):
                    st.session_state[f"list_checkbox_{key_prefix}_{index}"] = False
                # Turn OFF parent sidebar checkbox when deselecting all
                sync_sidebar_checkbox(key_prefix, False)
                st.rerun()
        
        st.markdown("---")
        
        # Check if parent pattern is visible in sidebar
        parent_visible = get_parent_visibility(key_prefix)
        
        # If parent is NOT visible, show message and don't render checkboxes
        if not parent_visible:
            st.info("Enable this pattern in the sidebar to select individual items.")
            return []
        
        for index, item in enumerate(item_list):
            # Simple key without version - parent visible means all are active
            state_key = f"list_checkbox_{key_prefix}_{index}"
            
            # Initialize if needed
            if state_key not in st.session_state:
                st.session_state[state_key] = True  # Default to checked when parent is visible
            
            # Render checkbox
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
        st.info("The input dictionary is empty.")
        return []
    
    selected_items = []
    
    with st.container(border=True):
        st.write(f"**{title}**")
        st.caption("⚠️ Changes will update the chart automatically")
        
        # Add "Select All" / "Deselect All" functionality
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select All", key=f"{key_prefix}_select_all", use_container_width=True):
                for key in data_dict.keys():
                    st.session_state[f"dict_checkbox_{key_prefix}_{key}"] = True
                # Sync with sidebar checkbox
                sync_sidebar_checkbox(key_prefix, True)
                # Trigger chart redisplay
                st.session_state['chart_needs_update'] = True
                st.rerun()
        with col_b:
            if st.button("Deselect All", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for key in data_dict.keys():
                    st.session_state[f"dict_checkbox_{key_prefix}_{key}"] = False
                # Sync with sidebar checkbox
                sync_sidebar_checkbox(key_prefix, False)
                # Trigger chart redisplay
                st.session_state['chart_needs_update'] = True
                st.rerun()
        
        st.markdown("---")
        
        # Check if parent pattern is visible in sidebar
        parent_visible = get_parent_visibility(key_prefix)
        
        # If parent is NOT visible, show message and don't render checkboxes
        if not parent_visible:
            st.info("Enable this pattern in the sidebar to select individual items.")
            return []
        
        for key, value in data_dict.items():
            # Simple key without version - parent visible means all are active
            state_key = f"dict_checkbox_{key_prefix}_{key}"
            
            # Initialize if needed
            if state_key not in st.session_state:
                st.session_state[state_key] = True  # Default to checked when parent is visible
            
            # Render checkbox
            checked = st.checkbox(key, key=state_key)
            
            if checked:
                selected_items.append(value)

    return selected_items

                        
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