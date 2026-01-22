"""
Pattern Detection - Pattern detection logic for Visual Pattern Detection.

This module handles all pattern detection algorithms:
- Temporal cluster detection
- OPTICS cluster detection
- Outlier detection
- Gap detection
- Case arrival trend detection
- Sequence detection (PrefixSpan)
"""

import streamlit as st
from core.detection import OutlierDetectionPattern, TemporalClusterPattern, CaseArrivalTrendPattern, ClusterPattern, HorizontalSequencePatternDetector
from core.detection.gap_pattern import GapPattern
from config.extended_pattern_matrix import is_pattern_meaningful


# =============================================================================
# === Detection Cache ===
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
    st.session_state.cluster_detected = False
    st.session_state.sequence_detected = False
    st.session_state.pattern_focus_mode = False
    if 'gap_detector' in st.session_state:
        del st.session_state['gap_detector']
    if 'cluster_detector' in st.session_state:
        del st.session_state['cluster_detector']
    if 'sequence_detector' in st.session_state:
        del st.session_state['sequence_detector']
    st.session_state['_pattern_cache_key'] = ''


# =============================================================================
# === Pattern Detection Orchestration ===
# =============================================================================

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
    st.session_state.cluster_detected = False
    st.session_state.sequence_detected = False
    if 'gap_detector' in st.session_state:
        del st.session_state['gap_detector']
    if 'cluster_detector' in st.session_state:
        del st.session_state['cluster_detector']
    if 'sequence_detector' in st.session_state:
        del st.session_state['sequence_detector']

    # Clear sub-pattern selections (they reference old data)
    if 'selected_gap_transitions' in st.session_state:
        del st.session_state['selected_gap_transitions']
    if 'selected_gap_resources' in st.session_state:
        del st.session_state['selected_gap_resources']
    if 'selected_outlier_types' in st.session_state:
        del st.session_state['selected_outlier_types']
    if 'selected_temporal_clusters' in st.session_state:
        del st.session_state['selected_temporal_clusters']

    # Flag to reset visibility on next rerun (before widgets render)
    st.session_state['_reset_pattern_visibility'] = True

    with st.spinner("Auto-detecting patterns..."):
        if is_pattern_meaningful(x_col, y_col, color_col, 'temporal_cluster_x'):
            _detect_temporal_clusters(x_col, y_col, df_selected)
        if is_pattern_meaningful(x_col, y_col, color_col, 'cluster'):
            _detect_clusters(x_col, y_col, color_col, df_selected)
        if is_pattern_meaningful(x_col, y_col, color_col, 'outlier'):
            _detect_outliers(df_selected)
        if is_pattern_meaningful(x_col, y_col, color_col, 'gap'):
            _detect_gaps(x_col, y_col, df_selected)
        if x_col == 'actual_time':
            _detect_case_arrival_trend(x_col, df_selected)
        if is_pattern_meaningful(x_col, y_col, color_col, 'sequence'):
            _detect_sequences(x_col, y_col, color_col, df_selected)


# =============================================================================
# === Individual Pattern Detectors ===
# =============================================================================

def _detect_temporal_clusters(x_col, y_col, df_selected):
    """Detect temporal clusters."""
    try:
        detector = TemporalClusterPattern(df=df_selected, x_axis=x_col, y_axis=y_col)
        if detector.detect():
            st.session_state.temporal_clusters = detector
            st.session_state.temporal_detected = True
    except Exception as e:
        st.warning(f"Temporal cluster detection skipped: {str(e)}")


def _detect_clusters(x_col, y_col, color_col, df_selected):
    """Detect clusters using OPTICS algorithm."""
    try:
        view_config = {'x': x_col, 'y': y_col}
        if color_col:
            view_config['color'] = color_col

        detector = ClusterPattern(view_config=view_config, algorithm='optics')
        detector.detect(df_selected)
        if detector.detected is not None:
            st.session_state.cluster_detector = detector
            st.session_state.cluster_detected = True
    except Exception as e:
        st.warning(f"Cluster detection skipped: {str(e)}")


def _detect_outliers(df):
    """Detect outliers."""
    try:
        outlier_pattern = OutlierDetectionPattern(df=df, view_config=st.session_state.view_config)
        if outlier_pattern.detect():
            st.session_state.outlier_pattern = outlier_pattern
            st.session_state.outlier_detected = True
    except Exception as e:
        st.warning(f"Outlier detection skipped: {str(e)}")


def _detect_gaps(x_col, y_col, df_selected, min_samples=None, gap_mode=None):
    """
    Detect gaps based on mode.

    Parameters
    ----------
    x_col : str
        X-axis column (must be time-like)
    y_col : str
        Y-axis column
    df_selected : pd.DataFrame
        Event log data
    min_samples : int, optional
        Minimum samples for threshold computation
    gap_mode : str, optional
        Detection mode: 'transition' (default) or 'resource_inactivity'
        Resource inactivity mode requires y_col='resource' or 'resource' in df
    """
    try:
        if min_samples is None:
            min_samples = st.session_state.get('gap_min_samples', 5)

        if gap_mode is None:
            gap_mode = st.session_state.get('gap_mode', 'transition')

        # Validate resource_inactivity mode requirements
        if gap_mode == 'resource_inactivity':
            if 'resource' not in df_selected.columns:
                st.warning("Resource inactivity mode requires 'resource' column")
                return

        y_is_categorical = df_selected[y_col].nunique() <= 60
        gap_detector = GapPattern(
            view_config={'x': x_col, 'y': y_col},
            y_is_categorical=y_is_categorical,
            gap_mode=gap_mode
        )
        gap_detector.MIN_SAMPLES_FOR_NORMALITY = min_samples
        gap_detector.detect(df_selected)

        if gap_detector.detected is not None and len(gap_detector.detected) > 0:
            st.session_state['gap_detector'] = gap_detector
            # Store current mode for cache validation
            st.session_state['_gap_mode_cache'] = gap_mode
        else:
            # Clear detector if no gaps found
            if 'gap_detector' in st.session_state:
                del st.session_state['gap_detector']

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

def _detect_sequences(x_col: str, y_col: str, color_col: str, df_selected, top_k: int = 5):
    """Detect horizontal sequences using PrefixSpan and filter to top k.

    Args:
        x_col: Column key for the x-axis (sequence detection axis).
        y_col: Column key for the y-axis (grouping key).
        color_col: Column key for the dot color (event key).
        df_selected: DataFrame containing the event log data.
        top_k: Number of top patterns by support_count to keep (default: 5).
    """
    try:
        sequence_detector = HorizontalSequencePatternDetector(
            x_axis=x_col,
            y_axis=y_col,
            dot_color=color_col,
            df=df_selected,
            min_support=50
        )
        if sequence_detector.detect():
            # Apply top k filtering immediately
            sequence_detector.get_top_k_sequences(top_k)
            
            summary = sequence_detector.get_summary()
            # Only mark as detected if actual patterns were found
            if summary.get('count', 0) > 0:
                st.session_state.sequence_detector = sequence_detector
                st.session_state.sequence_detected = True
                st.session_state.visible_sequence = True
    except Exception as e:
        st.warning(f"Sequence detection skipped: {str(e)}")
