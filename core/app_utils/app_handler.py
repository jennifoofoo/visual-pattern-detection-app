"""
App Handler - Core application logic for Visual Pattern Detection.

This module handles:
- Data loading and caching
- Chart configuration and plotting
- Pattern detection orchestration
- Focus selection and time filtering
- Sidebar controls
"""

import streamlit as st
from core.data_processing import load_xes_log
from core.evaluation.summary_generator import summarize_event_log
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP
from core.visualization.visualizer import plot_dotted_chart as plot_chart
from core.detection import OutlierDetectionPattern, TemporalClusterPattern, CaseArrivalTrendPattern
from core.detection.gap_pattern import GapPattern
from core.evaluation.ollama import OllamaEvaluator
from core.utils.demo_sampling import sample_small_eventlog
from config.extended_pattern_matrix import is_pattern_meaningful

# Import pattern UI from separate module
from core.app_utils.app_handler_pattern_detection import handle_pattern_detection


# =============================================================================
# === Caching ===
# =============================================================================

@st.cache_data(ttl=3600)
def cached_load_xes_log(xes_path):
    """Cached version of load_xes_log for better performance."""
    return load_xes_log(xes_path)


@st.cache_data(ttl=3600)
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
        st.session_state['_reset_pattern_visibility'] = False

    # Layer visibility flags (initial defaults)
    if 'visible_gap' not in st.session_state:
        st.session_state.visible_gap = True
    if 'visible_outlier' not in st.session_state:
        st.session_state.visible_outlier = True
    if 'visible_temporal_cluster' not in st.session_state:
        st.session_state.visible_temporal_cluster = True

    # Selection-based Focus
    if 'focus_df' not in st.session_state:
        st.session_state.focus_df = None

    # Time filter
    if 'time_filter_range' not in st.session_state:
        st.session_state.time_filter_range = None


# =============================================================================
# === Data Loading ===
# =============================================================================

def load_data_button(xes_path, demo_mode=False):
    """Load XES file and initialize data."""
    try:
        with st.spinner(f"Loading {xes_path}..."):
            df = cached_load_xes_log(xes_path)

        if df.empty:
            st.warning("The log file was loaded but contains no events.")
            return

        # Demo Mode: Sample for fast detection
        if demo_mode and 'case_id' in df.columns:
            df_original = df
            df = sample_small_eventlog(
                df,
                max_cases=100,
                max_events_per_case=30,
                time_col='actual_time',
                random_state=42
            )
            st.info(
                f"**DEMO MODE:** Sampled to {len(df):,} events from {len(df_original):,} "
                f"({df['case_id'].nunique()} cases). Uncheck 'Demo Mode' for full dataset."
            )

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

def get_chart_config_with_selectboxes():
    """Render chart configuration selectboxes."""
    x_axis = st.selectbox('X-Axis', list(X_AXIS_COLUMN_MAP.keys()))
    y_axis = st.selectbox('Y-Axis', list(Y_AXIS_COLUMN_MAP.keys()))
    dots_config_label = st.selectbox('Dot Color', list(DOTS_COLOR_MAP.keys()))
    return x_axis, y_axis, dots_config_label


def plot_chart_button(x_axis, y_axis, dots_config_label):
    """Create and store chart based on configuration."""
    df_base = st.session_state['df']
    x_col = X_AXIS_COLUMN_MAP[x_axis]
    y_col = Y_AXIS_COLUMN_MAP[y_axis]
    dots_config_col = DOTS_COLOR_MAP[dots_config_label]

    df_selected = df_base
    if df_selected[x_col].isnull().any() or df_selected[y_col].isnull().any():
        df_selected = df_base.copy()
        df_selected.dropna(subset=[x_col, y_col], inplace=True)
        if df_selected.empty:
            st.warning("No valid data to plot after removing missing values.")
            return

    total_points = len(df_selected)
    hover_cols = ['activity', 'actual_time']

    with st.spinner("Rendering chart..."):
        fig = plot_chart(
            df=df_selected,
            x=x_col,
            y=y_col,
            color=dots_config_col,
            title=f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} points)",
            labels={x_col: x_axis, y_col: y_axis, dots_config_col: dots_config_label},
            hover_data=hover_cols
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
# === Pattern Detection ===
# =============================================================================

def _get_detection_cache_key(x_col, y_col, color_col, df_len):
    """Generate cache key for pattern detection."""
    time_filter = st.session_state.get('time_filter_range')
    time_key = f"{time_filter}" if time_filter else "none"
    return f"{x_col}_{y_col}_{color_col}_{df_len}_{time_key}"


def _reset_pattern_detection_state():
    """Clear all pattern detection results."""
    st.session_state.temporal_detected = False
    st.session_state.outlier_detected = False
    st.session_state.case_arrival_trend_detected = False
    if 'gap_detector' in st.session_state:
        del st.session_state['gap_detector']
    st.session_state['_pattern_cache_key'] = ''


def auto_detect_patterns(x_col, y_col, color_col, x_axis_label, y_axis_label, df_selected):
    """Automatically detect all meaningful patterns."""
    is_focus_view = st.session_state.get('focus_df') is not None

    if not is_focus_view:
        cache_key = _get_detection_cache_key(x_col, y_col, color_col, len(df_selected))
        if cache_key == st.session_state.get('_pattern_cache_key', ''):
            return
        st.session_state['_pattern_cache_key'] = cache_key

    # Clear old results
    st.session_state.temporal_detected = False
    st.session_state.outlier_detected = False
    st.session_state.case_arrival_trend_detected = False
    if 'gap_detector' in st.session_state:
        del st.session_state['gap_detector']

    # Clear sub-pattern selections (they reference old data)
    if 'selected_gap_transitions' in st.session_state:
        del st.session_state['selected_gap_transitions']
    if 'selected_outlier_types' in st.session_state:
        del st.session_state['selected_outlier_types']
    if 'selected_temporal_clusters' in st.session_state:
        del st.session_state['selected_temporal_clusters']

    # Flag to reset visibility on next rerun (before widgets render)
    st.session_state['_reset_pattern_visibility'] = True

    with st.spinner("Auto-detecting patterns..."):
        if is_pattern_meaningful(x_col, y_col, color_col, 'temporal_cluster_x'):
            _detect_temporal_clusters(x_col, y_col, df_selected)
        if is_pattern_meaningful(x_col, y_col, color_col, 'outlier'):
            _detect_outliers(df_selected)
        if is_pattern_meaningful(x_col, y_col, color_col, 'gap'):
            _detect_gaps(x_col, y_col, df_selected)
        if x_col == 'actual_time':
            _detect_case_arrival_trend(x_col, df_selected)


def _detect_temporal_clusters(x_col, y_col, df_selected):
    """Detect temporal clusters."""
    try:
        detector = TemporalClusterPattern(df=df_selected, x_axis=x_col, y_axis=y_col, min_cluster_size=10)
        if detector.detect():
            st.session_state.temporal_clusters = detector
            st.session_state.temporal_detected = True
    except Exception as e:
        st.warning(f"Temporal cluster detection skipped: {str(e)}")


def _detect_outliers(df):
    """Detect outliers."""
    try:
        outlier_pattern = OutlierDetectionPattern(df=df, view_config=st.session_state.view_config)
        if outlier_pattern.detect():
            st.session_state.outlier_pattern = outlier_pattern
            st.session_state.outlier_detected = True
    except Exception as e:
        st.warning(f"Outlier detection skipped: {str(e)}")


def _detect_gaps(x_col, y_col, df_selected, min_samples=None):
    """Detect gaps in transitions."""
    try:
        if min_samples is None:
            min_samples = st.session_state.get('gap_min_samples', 5)

        y_is_categorical = df_selected[y_col].nunique() <= 60
        gap_detector = GapPattern(view_config={'x': x_col, 'y': y_col}, y_is_categorical=y_is_categorical)
        gap_detector.MIN_SAMPLES_FOR_NORMALITY = min_samples
        gap_detector.detect(df_selected)

        if gap_detector.detected is not None and len(gap_detector.detected) > 0:
            st.session_state['gap_detector'] = gap_detector
    except Exception as e:
        st.warning(f"Gap detection skipped: {str(e)}")


def _detect_case_arrival_trend(x_col, df_selected):
    """Detect case arrival trends."""
    try:
        detector = CaseArrivalTrendPattern(view_config={'x': x_col}, aggregation_period='W', min_periods=3)
        if detector.detect(df_selected) or detector.trend_result is not None:
            st.session_state['case_arrival_trend_detector'] = detector
            st.session_state.case_arrival_trend_detected = True
    except Exception as e:
        st.warning(f"Case arrival trend detection skipped: {str(e)}")


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

    df_display = df_display.reset_index(drop=True)
    df_display['_point_id'] = df_display.index

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

    # Create chart
    fig = plot_chart(
        df=df_display,
        x=x_col,
        y=y_col,
        color=dots_config_col,
        title=title,
        labels={x_col: x_axis, y_col: y_axis, dots_config_col: dots_config_label},
        hover_data=['activity', 'actual_time'],
        custom_data=['_point_id']
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        height=600,
        showlegend=(dots_config_col is not None and dots_config_col != 'case_id'),
        hovermode='closest',
        template='plotly_white',
        yaxis=dict(autorange='reversed'),
        dragmode='lasso'
    )

    # Add pattern overlays
    if st.session_state.get('visible_gap', True):
        if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            fig = st.session_state['gap_detector'].visualize(df_display, fig)

    if st.session_state.get('visible_outlier', True):
        if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
            fig = st.session_state.outlier_pattern.visualize(df_display, fig)

    if st.session_state.get('visible_temporal_cluster', True):
        if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
            fig = st.session_state.temporal_clusters.visualize(df_display, fig)

    # Display chart
    selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="main_chart")
    st.session_state['fig'] = fig

    # Time filter and focus controls
    st.markdown("<div style='margin-top: 1.5rem'></div>", unsafe_allow_html=True)
    if x_col == 'actual_time':
        _display_time_filter(plot_config)
        st.markdown("<div style='margin-top: 1rem'></div>", unsafe_allow_html=True)
    _display_focus_controls(selection, plot_config, df_display, is_focus_view)


# =============================================================================
# === Time Filter ===
# =============================================================================

def _display_time_filter(plot_config):
    """Render time filter controls."""
    import pandas as pd

    base_df = plot_config['df_selected']
    min_time = base_df['actual_time'].min()
    max_time = base_df['actual_time'].max()

    with st.container(border=True):
        st.markdown("**Time Filter**")
        st.caption("Reduce event log by date range (optional)")
        st.markdown("")
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
        st.markdown("")
        col_apply, col_clear = st.columns(2)
        with col_apply:
            if st.button("Apply Filter", key="apply_time_filter", type="primary", use_container_width=True):
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                st.session_state.time_filter_range = (start_dt, end_dt)
                _reset_pattern_detection_state()
                auto_detect_patterns(
                    plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'],
                    plot_config['x_axis_label'], plot_config['y_axis_label'],
                    get_active_view_df(plot_config)
                )
                st.rerun()
        with col_clear:
            if st.button("Clear Filter", key="clear_time_filter", use_container_width=True,
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
            st.caption(f"Active: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")


# =============================================================================
# === Focus Selection ===
# =============================================================================

def _display_focus_controls(selection, plot_config, df_display, is_focus_view):
    """Display selection-based focus controls."""
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

    with st.container(border=True):
        if is_focus_view:
            full_count = len(plot_config['df_selected'])
            st.markdown("**Selection** · Focus View")
            st.caption(f"{len(df_display):,} of {full_count:,} points · patterns re-analyzed")
        else:
            st.markdown("**Selection**")
            if selected_indices:
                st.caption(f"{len(selected_indices):,} points selected · click Focus to analyze")
            else:
                st.caption("Use lasso or box select on the chart")
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Focus", disabled=(is_focus_view or not selected_indices), key="focus_btn", type="primary", use_container_width=True):
                _apply_focus_selection(selected_indices, df_display, plot_config)
        with col2:
            if st.button("Reset", disabled=not is_focus_view, key="reset_focus_btn", use_container_width=True):
                _reset_focus_view(plot_config)


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

def _is_any_pattern_detected() -> bool:
    """Check if any pattern has been detected."""
    return (
        st.session_state.get('temporal_detected', False) or
        st.session_state.get('outlier_detected', False) or
        st.session_state.get('case_arrival_trend_detected', False) or
        ('gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None)
    )


def _render_pattern_checkbox(label: str, visibility_key: str, version_key: str, checkbox_key_pattern: str):
    """Render pattern visibility checkbox with sub-pattern sync."""
    if visibility_key not in st.session_state:
        st.session_state[visibility_key] = True

    prev_state = st.session_state[visibility_key]
    st.checkbox(label, key=visibility_key, help=f"Show/hide {label.lower()} visualization")

    if st.session_state[visibility_key] != prev_state:
        st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
        for key in [k for k in list(st.session_state.keys()) if checkbox_key_pattern in k]:
            del st.session_state[key]
        st.rerun()


def sidebar_pattern_layer_controls():
    """Display pattern layer visibility controls in sidebar."""
    if not _is_any_pattern_detected():
        return

    st.markdown("##### Pattern Layers")

    if st.session_state.get('temporal_detected', False):
        _render_pattern_checkbox("Temporal Clusters", 'visible_temporal_cluster', 'temporal_cluster_version', 'checkbox_temporal_cluster_')

    if st.session_state.get('outlier_detected', False):
        _render_pattern_checkbox("Outlier Detection", 'visible_outlier', 'outlier_type_version', 'checkbox_outlier_type_')

    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        _render_pattern_checkbox("Gap Detection", 'visible_gap', 'gap_transition_version', 'checkbox_gap_transition_')


# =============================================================================
# === AI Description ===
# =============================================================================

def ollama_description_button():
    """Generate AI description of chart."""
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
