"""
App Handler - Core application logic for Visual Pattern Detection.

This module handles:
- Data loading and caching
- Chart configuration and plotting
- Focus selection and time filtering
- Sidebar controls
"""

import streamlit as st
from core.app_utils.pattern_detection import _reset_pattern_detection_state, auto_detect_patterns
from core.app_utils.pattern_ui import _is_any_pattern_detected
from core.data_processing import load_xes_log
from core.evaluation.summary_generator import summarize_event_log
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP
from core.visualization.visualizer import plot_dotted_chart as plot_chart
from core.utils.demo_sampling import (
    sample_eventlog_variant_aware,
    SamplingMode,
    SAMPLING_CONFIGS
)


# =============================================================================
# === Caching ===
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_xes_log(xes_path):
    """Cached version of load_xes_log for better performance."""
    return load_xes_log(xes_path)


@st.cache_data(ttl=3600, show_spinner=False)
def generate_summary(df):
    """Cached summary generation."""
    return summarize_event_log(df)


# =============================================================================
# === State Initialization ===
# =============================================================================

def init_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'chart_plotted' not in st.session_state:
        st.session_state.chart_plotted = False

    # Reset visibility flags if new patterns were detected (before widgets render)
    if st.session_state.get('_reset_pattern_visibility', False):
        st.session_state.visible_gap = True
        st.session_state.visible_outlier = True
        st.session_state.visible_temporal_cluster = True
        st.session_state.visible_cluster = True
        st.session_state.visible_sequence = True
        st.session_state.visible_case_arrival_trend = True
        st.session_state['_reset_pattern_visibility'] = False

    # Layer visibility flags (initial defaults)
    if 'visible_gap' not in st.session_state:
        st.session_state.visible_gap = True
    if 'visible_outlier' not in st.session_state:
        st.session_state.visible_outlier = True
    if 'visible_temporal_cluster' not in st.session_state:
        st.session_state.visible_temporal_cluster = True
    if 'visible_cluster' not in st.session_state:
        st.session_state.visible_cluster = True
    if 'visible_sequence' not in st.session_state:
        st.session_state.visible_sequence = True
    if 'visible_case_arrival_trend' not in st.session_state:
        st.session_state.visible_case_arrival_trend = True

    # Selection-based Focus
    if 'focus_df' not in st.session_state:
        st.session_state.focus_df = None

    # Time filter
    if 'time_filter_range' not in st.session_state:
        st.session_state.time_filter_range = None

    # Pattern Focus Mode
    if 'pattern_focus_mode' not in st.session_state:
        st.session_state.pattern_focus_mode = False


# =============================================================================
# === Data Loading ===
# =============================================================================

def load_data_button(xes_path, demo_mode=False, sampling_mode: SamplingMode = SamplingMode.SQRT):
    try:
        with st.spinner(f"Loading {xes_path}..."):
            df = cached_load_xes_log(xes_path)

        if df.empty:
            st.warning("The log file was loaded but contains no events.")
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

        st.session_state.df = df
        st.session_state.loaded_file = xes_path
        st.session_state.data_loaded = True
        st.session_state.chart_plotted = False
        st.session_state.summary = generate_summary(df)

        st.success(f"Log loaded: {len(df):,} events")

        # Auto-plot with default config (Actual time, Case ID, Activity)
        plot_chart_button("Actual time", "Case ID", "Activity")

    except Exception as e:
        st.error(f"Error loading XES log: {e}")
        st.session_state.data_loaded = False


# =============================================================================
# === Chart Configuration ===
# =============================================================================

def get_chart_config_with_selectboxes(default_x=None, default_y=None, default_color=None):
    """Render chart configuration selectboxes with optional defaults."""
    x_options = list(X_AXIS_COLUMN_MAP.keys())
    y_options = list(Y_AXIS_COLUMN_MAP.keys())
    color_options = list(DOTS_COLOR_MAP.keys())

    x_index = x_options.index(default_x) if default_x in x_options else 0
    y_index = y_options.index(default_y) if default_y in y_options else 0
    color_index = color_options.index(default_color) if default_color in color_options else 0

    x_axis = st.selectbox('X-Axis', x_options, index=x_index,
        help="Time-based axes reveal temporal patterns (gaps, trends). Use 'Actual time' for most analyses.")
    y_axis = st.selectbox('Y-Axis', y_options, index=y_index,
        help="Choose what to compare: Cases show individual progressions, Resources show workload, Activities show process flow.")
    dots_config_label = st.selectbox('Dot Color', color_options, index=color_index,
        help="Color coding adds a third dimension. Color by Activity to see what happens, by Resource to see who does it.")

    return x_axis, y_axis, dots_config_label


def plot_chart_button(x_axis, y_axis, dots_config_label):
    """Create and store chart based on configuration."""
    df_base = st.session_state['df']
    x_col = X_AXIS_COLUMN_MAP[x_axis]
    y_col = Y_AXIS_COLUMN_MAP[y_axis]
    dots_config_col = DOTS_COLOR_MAP[dots_config_label]

    df_selected = df_base.copy()
    if df_selected[x_col].isnull().any() or df_selected[y_col].isnull().any():
        df_selected.dropna(subset=[x_col, y_col], inplace=True)
        if df_selected.empty:
            st.warning("No valid data to plot after removing missing values.")
            return

    # Reset index and add _point_id for selection
    df_selected = df_selected.reset_index(drop=True)
    df_selected['_point_id'] = df_selected.index
    total_points = len(df_selected)

    with st.spinner("Rendering chart..."):
        # Build hover columns (exclude axes and color to avoid redundancy)
        hover_cols = []
        for col in ['case_id', 'activity', 'resource', 'actual_time', 'outlier_reason']:
            if col in df_selected.columns and col not in [x_col, y_col, dots_config_col]:
                hover_cols.append(col)

        fig = plot_chart(
            df=df_selected,
            x=x_col,
            y=y_col,
            color=dots_config_col,
            title=f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} points)",
            labels={x_col: x_axis, y_col: y_axis, dots_config_col: dots_config_label},
            hover_data=hover_cols,
            custom_data=['_point_id']
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_layout(
            showlegend=(dots_config_col is not None and dots_config_col != 'case_id'),
            hovermode='closest',
            template='plotly_white',
            yaxis=dict(autorange='reversed')
        )

    st.session_state['current_plot_config'] = {
        'x_col': x_col,
        'y_col': y_col,
        'dots_config_col': dots_config_col,
        'x_axis_label': x_axis,
        'y_axis_label': y_axis,
        'dots_config_label': dots_config_label,
        'df_selected': df_selected,
        'total_points': total_points
    }
    st.session_state['fig'] = fig
    st.session_state['view_config'] = {'x': x_col, 'y': y_col, 'color': dots_config_col}
    st.session_state['chart_plotted'] = True
    st.session_state.focus_df = None

    st.success("Chart created successfully!")
    auto_detect_patterns(x_col, y_col, dots_config_col, x_axis, y_axis, get_active_view_df(st.session_state['current_plot_config']))


# =============================================================================
# === Active View DataFrame ===
# =============================================================================

def get_active_view_df(plot_config: dict):
    """Get current active dataframe (full or focused, with time filter applied)."""
    import pandas as pd

    focus_df = st.session_state.get('focus_df')
    df = focus_df if focus_df is not None else plot_config['df_selected']

    time_range = st.session_state.get('time_filter_range')
    if time_range is not None and plot_config.get('x_col') == 'actual_time':
        start, end = time_range
        if df['actual_time'].dt.tz is not None:
            start = pd.Timestamp(start).tz_localize(df['actual_time'].dt.tz)
            end = pd.Timestamp(end).tz_localize(df['actual_time'].dt.tz)
        df = df[(df['actual_time'] >= start) & (df['actual_time'] <= end)]

    return df


# =============================================================================
# === Chart Display ===
# =============================================================================

def display_chart():
    """Display chart with pattern overlays and controls."""
    if not st.session_state.get('chart_plotted', False):
        return

    plot_config = st.session_state.get('current_plot_config', {})
    if not plot_config:
        return

    df_display = get_active_view_df(plot_config)
    is_focus_view = st.session_state.get('focus_df') is not None

    # Keep original df for pattern visualization (preserves original indices)
    df_for_patterns = df_display.copy()

    # Use _point_id for mapping instead of df_display.index
    # Pattern indices from detectors match _point_id values from df_selected
    point_ids = df_display['_point_id'].values  # Save before reset

    df_display = df_display.reset_index(drop=True)
    df_display['_point_id'] = df_display.index  # Renumber for display

    # Create mapping from pattern indices (_point_id values) to display positions
    point_id_to_display_idx = {pid: new_idx for new_idx, pid in enumerate(point_ids)}

    x_col = plot_config['x_col']
    y_col = plot_config['y_col']
    dots_config_col = plot_config['dots_config_col']
    x_axis = plot_config['x_axis_label']
    y_axis = plot_config['y_axis_label']
    dots_config_label = plot_config['dots_config_label']
    total_points = len(df_display)

    # Title
    if is_focus_view:
        full_count = len(plot_config['df_selected'])
        title = f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} of {full_count:,} points) [FOCUS VIEW]"
    else:
        title = f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} points)"

    # Determine focus mode indices (pattern focus mode)
    focus_indices = None
    if st.session_state.get('pattern_focus_mode'):
        from core.utils.pattern_indices import get_all_pattern_indices
        original_focus_indices = get_all_pattern_indices(st.session_state)
        if original_focus_indices:
            # Map pattern indices (which equal _point_id values) to display indices
            focus_indices = {point_id_to_display_idx[idx] for idx in original_focus_indices if idx in point_id_to_display_idx}
            if not focus_indices:  # Empty set means no patterns in view
                focus_indices = None  # Disable focus mode
            if focus_indices:
                title += " [PATTERN FOCUS]"

    # Create chart
    fig = plot_chart(
        df=df_display,
        x=x_col,
        y=y_col,
        color=dots_config_col,
        title=title,
        labels={x_col: x_axis, y_col: y_axis, dots_config_col: dots_config_label},
        hover_data=['activity', 'actual_time'],
        custom_data=['_point_id'],
        focus_mode_indices=focus_indices
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        height=600,
        showlegend=(dots_config_col is not None and dots_config_col != 'case_id'),
        hovermode='closest',
        template='plotly_white',
        yaxis=dict(autorange='reversed')
    )

    # Add pattern overlays (use df_for_patterns to preserve original indices)
    if st.session_state.get('visible_gap', True):
        if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            fig = st.session_state['gap_detector'].visualize(df_for_patterns, fig)

    if st.session_state.get('visible_outlier', True):
        if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
            fig = st.session_state.outlier_pattern.visualize(df_for_patterns, fig)

    if st.session_state.get('visible_temporal_cluster', True):
        if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
            fig = st.session_state.temporal_clusters.visualize(
                df_for_patterns, 
                fig,
                selected_clusters=st.session_state.get('selected_temporal_clusters')
            )

    if st.session_state.get('visible_cluster', True):
        if st.session_state.get('cluster_detected', False) and 'cluster_detector' in st.session_state:
            fig = st.session_state.cluster_detector.visualize(
                df_for_patterns, 
                fig,
                selected_clusters=st.session_state.get('selected_OPTICS_clusters'),
                show_noise=st.session_state.get('show_cluster_noise', False)
            )
   
    if st.session_state.get('visible_sequence', True):
        if st.session_state.get('sequence_detected', False) and 'sequence_detector' in st.session_state:
            fig = st.session_state.sequence_detector.visualize(
                df_for_patterns,
                fig,
                selected_seq_patterns=st.session_state.get('selected_seq_patterns')
            )
    
    # Display chart
    selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="main_chart")
    st.session_state['fig'] = fig

    # Store selection data for sidebar controls
    st.session_state['_current_df_display'] = df_display
    selected_indices = []
    if not is_focus_view and selection and selection.selection:
        for pt in selection.selection.get("points", []):
            customdata = pt.get("customdata")
            if customdata is not None:
                try:
                    if isinstance(customdata, (list, tuple)) and len(customdata) > 0:
                        selected_indices.append(customdata[0])
                    elif hasattr(customdata, '__iter__') and not isinstance(customdata, str):
                        val = list(customdata)
                        if val:
                            selected_indices.append(val[0])
                    else:
                        selected_indices.append(int(customdata))
                except (TypeError, IndexError, ValueError):
                    pass

    # Only update selection state if we have new selection data
    # (avoid overwriting with empty list on button clicks/reruns)
    if selected_indices:
        old_selection = st.session_state.get('_selected_point_indices', [])
        if set(selected_indices) != set(old_selection):
            st.session_state['_selected_point_indices'] = selected_indices
            st.rerun()


# =============================================================================
# === Focus Selection Helpers ===
# =============================================================================

def _apply_focus_selection(selected_point_ids, df_display, plot_config):
    """Apply focus to selected points."""
    st.session_state.focus_df = df_display[df_display['_point_id'].isin(selected_point_ids)].copy()
    _reset_pattern_detection_state()
    auto_detect_patterns(
        plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'],
        plot_config['x_axis_label'], plot_config['y_axis_label'],
        get_active_view_df(plot_config)
    )
    st.rerun()


def _reset_focus_view(plot_config):
    """Reset to full view."""
    st.session_state.focus_df = None
    _reset_pattern_detection_state()
    auto_detect_patterns(
        plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'],
        plot_config['x_axis_label'], plot_config['y_axis_label'],
        get_active_view_df(plot_config)
    )
    st.rerun()


# =============================================================================
# === Sidebar Controls ===
# =============================================================================

def _render_pattern_checkbox(label: str, visibility_key: str, version_key: str, checkbox_key_pattern: str, help_text: str = None):
    """Render pattern visibility checkbox with sub-pattern sync."""
    if visibility_key not in st.session_state:
        st.session_state[visibility_key] = True

    prev_state = st.session_state[visibility_key]
    st.checkbox(label, key=visibility_key, help=help_text or f"Show/hide {label.lower()}")

    if st.session_state[visibility_key] != prev_state:
        st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
        for key in [k for k in list(st.session_state.keys()) if checkbox_key_pattern in k]:
            del st.session_state[key]
        st.rerun()


def sidebar_time_filter():
    """Render time filter controls in sidebar."""
    import pandas as pd

    plot_config = st.session_state.get('current_plot_config')
    if not plot_config:
        st.caption("Plot a chart first")
        return

    x_col = plot_config.get('x_col')
    if x_col != 'actual_time':
        st.caption("Time filter only available for actual_time x-axis")
        return

    base_df = plot_config['df_selected']
    min_time = base_df['actual_time'].min()
    max_time = base_df['actual_time'].max()

    st.caption("Filter events by date range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=min_time.date() if pd.notna(min_time) else None,
            min_value=min_time.date() if pd.notna(min_time) else None,
            max_value=max_time.date() if pd.notna(max_time) else None,
            key="time_filter_start"
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=max_time.date() if pd.notna(max_time) else None,
            min_value=min_time.date() if pd.notna(min_time) else None,
            max_value=max_time.date() if pd.notna(max_time) else None,
            key="time_filter_end"
        )

    col_apply, col_clear = st.columns(2)
    with col_apply:
        if st.button("Apply", key="apply_time_filter", type="primary", use_container_width=True):
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date).replace(hour=23, minute=59, second=59, microsecond=999999)
            st.session_state.time_filter_range = (start_dt, end_dt)
            _reset_pattern_detection_state()
            auto_detect_patterns(
                plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'],
                plot_config['x_axis_label'], plot_config['y_axis_label'],
                get_active_view_df(plot_config)
            )
            st.rerun()
    with col_clear:
        if st.button("Clear", key="clear_time_filter", use_container_width=True,
                    disabled=st.session_state.get('time_filter_range') is None):
            st.session_state.time_filter_range = None
            _reset_pattern_detection_state()
            auto_detect_patterns(
                plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'],
                plot_config['x_axis_label'], plot_config['y_axis_label'],
                get_active_view_df(plot_config)
            )
            st.rerun()

    if st.session_state.get('time_filter_range'):
        start, end = st.session_state.time_filter_range
        st.caption(f"**Active:** {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")


def sidebar_focus_controls():
    """Render selection/focus controls in sidebar."""
    plot_config = st.session_state.get('current_plot_config')
    if not plot_config:
        st.caption("Plot a chart first")
        return

    st.caption("Select points with lasso/box on chart, then Focus to zoom in")

    is_focus_view = st.session_state.get('focus_df') is not None
    selected_indices = st.session_state.get('_selected_point_indices', [])
    df_display = st.session_state.get('_current_df_display')

    if is_focus_view:
        full_count = len(plot_config['df_selected'])
        focus_count = len(st.session_state.focus_df) if st.session_state.focus_df is not None else 0
        if selected_indices:
            st.caption(f"**Focus View:** {focus_count:,} pts | **{len(selected_indices):,} selected**")
        else:
            st.caption(f"**Focus View:** {focus_count:,} of {full_count:,} points")
    else:
        if selected_indices:
            st.caption(f"**{len(selected_indices):,} points selected**")
        else:
            st.caption("Use lasso/box select on chart")

    col1, col2 = st.columns(2)
    with col1:
        # Allow re-focusing within focus view (nested selection)
        if st.button("Focus", disabled=not selected_indices, key="focus_btn", type="primary", use_container_width=True):
            if df_display is not None:
                _apply_focus_selection(selected_indices, df_display, plot_config)
    with col2:
        if st.button("Reset", disabled=not is_focus_view, key="reset_focus_btn", use_container_width=True):
            _reset_focus_view(plot_config)


def _get_pattern_not_detected_reason(pattern_key: str, x_col: str, y_col: str, color_col: str) -> str:
    """Get explanation why a pattern was not detected."""
    from config.extended_pattern_matrix import is_pattern_meaningful, get_pattern_info

    # Map pattern keys to matrix pattern names
    pattern_map = {
        'temporal': 'temporal_cluster_x',
        'cluster': 'cluster',
        'outlier': 'outlier',
        'gap': 'gap',
        'sequence': 'sequence',
        'case_arrival_trend': 'trend'
    }
    matrix_name = pattern_map.get(pattern_key, pattern_key)

    info = get_pattern_info(x_col, y_col, color_col, matrix_name)
    if info is None:
        return "Not applicable for this view configuration"

    if not info.get("can_be_found", False):
        return "Cannot be detected with current axis configuration"
    if not info.get("makes_sense", False):
        return "Not meaningful for this view (would not provide useful insights)"

    # Pattern is meaningful but wasn't found
    return "No patterns found in data"


def sidebar_focus_mode_toggle():
    st.checkbox("Pattern Focus Mode", key="pattern_focus_mode",
                disabled=not _is_any_pattern_detected(),
                help="Highlight only pattern points. Non-pattern points are grayed out to reduce visual clutter.")


def sidebar_pattern_layer_controls():
    """Display pattern layer visibility controls in sidebar with explanations."""
    plot_config = st.session_state.get('current_plot_config')
    if not plot_config:
        return

    x_col = plot_config.get('x_col', '')
    y_col = plot_config.get('y_col', '')
    color_col = plot_config.get('dots_config_col', '')

    st.markdown("##### Pattern Layers")

    def get_detector_info(detector_key):
        """Get count and explanation from detector."""
        d = st.session_state.get(detector_key)
        if not d or not hasattr(d, 'get_summary'):
            return 0, None
        s = d.get_summary()
        count = s.get('count', 0)
        details = s.get('details', {})
        # Build explanation from summary
        if detector_key == 'gap_detector':
            return count, f"Detected via per-transition thresholds (P95/IQR). {details.get('transitions_with_anomalies', 0)} transitions affected."
        elif detector_key == 'outlier_pattern':
            return count, f"Detected via Isolation Forest + statistical methods across {len(details.get('outlier_types', []))} categories."
        elif detector_key == 'cluster_detector':
            return count, f"Detected via OPTICS clustering algorithm."
        elif detector_key == 'sequence_detector':
            return count, f"Detected via PrefixSpan with {details.get('min_support', 80)}% min support."
        elif detector_key == 'temporal_clusters':
            return count, "Detected via DBSCAN on time axis."
        return count, None

    def show_pattern(label, count, visibility_key, version_key, checkbox_key, matrix_key, explanation=None):
        if count:
            _render_pattern_checkbox(f"{label} ({count})", visibility_key, version_key, checkbox_key, explanation)
        else:
            reason = _get_pattern_not_detected_reason(matrix_key, x_col, y_col, color_col)
            st.checkbox(label, value=False, disabled=True, help=f"Not detected: {reason}")

    gap_count, gap_exp = get_detector_info('gap_detector')
    outlier_count, outlier_exp = get_detector_info('outlier_pattern')
    cluster_count, cluster_exp = get_detector_info('cluster_detector')
    seq_count, seq_exp = get_detector_info('sequence_detector')
    temporal_count, temporal_exp = get_detector_info('temporal_clusters')

    show_pattern("Temporal Clusters", temporal_count, 'visible_temporal_cluster', 'temporal_cluster_version', 'checkbox_temporal_cluster_', 'temporal', temporal_exp)
    show_pattern("Clusters", cluster_count, 'visible_cluster', 'cluster_version', 'checkbox_cluster_', 'cluster', cluster_exp)
    show_pattern("Outliers", outlier_count, 'visible_outlier', 'outlier_type_version', 'checkbox_outlier_type_', 'outlier', outlier_exp)
    show_pattern("Gaps", gap_count, 'visible_gap', 'gap_transition_version', 'checkbox_gap_transition_', 'gap', gap_exp)
    show_pattern("Sequences", seq_count, 'visible_sequence', 'sequence_version', 'checkbox_sequence_', 'sequence', seq_exp)
    show_pattern("Case Arrival Trend", 1 if st.session_state.get('case_arrival_trend_detected') else 0, 'visible_case_arrival_trend', 'case_arrival_trend_version', 'checkbox_case_arrival_trend_', 'case_arrival_trend', "Detected via Mann-Kendall trend test.")

