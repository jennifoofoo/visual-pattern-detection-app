import streamlit as st

from core.detection import OutlierDetectionPattern, TemporalClusterPattern
from core.detection.gap_pattern import GapPattern
from core.detection.sequence_detection import HorizontalSequencePatternDetector

from core.app_utils.app_handler_pattern_filtering import list_to_multicheckbox, dict_to_multicheckbox

# region Utils App Handler
def _is_any_pattern_detected() -> bool:
    """Check if any pattern has been detected."""
    return (
        st.session_state.get('temporal_detected', False) or
        st.session_state.get('outlier_detected', False) or
        st.session_state.get('case_arrival_trend_detected', False) or
        ('gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None)
    )

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
    if st.session_state.get('sequence_detected', False):
        tabs_names.append("Sequence Detection")
        tabs_types.append('sequence')
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
    elif pattern_type == 'sequence':
        display_sequence_tab()
    elif pattern_type == 'case_arrival_trend':
        display_case_arrival_trend_tab()


def handle_pattern_detection():
    """Display pattern summary with tabs for each detected pattern."""
    st.markdown("#### Pattern Summary")

    if not _is_any_pattern_detected():
        st.caption("No patterns detected")
        return

    tabs_names, tabs_types = _get_detected_pattern_tabs()
    tabs = st.tabs(tabs_names)

    for tab, pattern_type in zip(tabs, tabs_types):
        with tab:
            _display_pattern_tab(pattern_type)


# region Temp Cluster
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
# endregion

# region Outlier
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
# endregion

# region Gap
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
        else:
            # Clear gap detector if no gaps found
            if 'gap_detector' in st.session_state:
                del st.session_state['gap_detector']
    except Exception as e:
        st.warning(f"Gap detection skipped: {str(e)}")

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
                    st.success(f"✅ {len(selected_transitions)} of {len(transition_dict)} transitions selected")
            else:
                st.info("No transition stats available")
# endregion

# region Sequence
# TODO: remove outlier dummy code
def _detect_sequences():
    """Detect horizontal sequences and store in session state."""
    try:
        sequence_detector = HorizontalSequencePatternDetector(
            min_support=50
        )
        # TODO: 1. detect sequence to bool
        if sequence_detector.detect():
            st.session_state.sequence_detector = sequence_detector
            st.session_state.sequence_detected = True
            st.session_state.visible_sequence = True
    except Exception as e:
        st.warning(f"Sequence detection skipped: {str(e)}")

def display_sequence_tab():
    """Display Sequence Detection pattern details in tab with individual outlier type selection."""
    if st.session_state.get('sequence_detected', False) and 'sequence_detector' in st.session_state:
        sequence_detector = st.session_state.sequence_detector
        summary = sequence_detector.get_summary()
        details = summary['details']
        
        # Layer visibility is now controlled by sidebar
        layer_visible = st.session_state.get('visible_sequence', True)
        
        if not layer_visible:
            st.info("Layer hidden - toggle in sidebar to show on chart")
        
        stats = summary['details'].get('statistics', {})
        st.success(f"{summary['count']} sequences detected.")
        
        # === NESTED TABS FOR OVERVIEW AND OUTLIER TYPE CONTROL ===
        subtab1, subtab2 = st.tabs(["Overview", "Sequences Control"])
        
        with subtab1:
            pass
            # if summary['details'].get('outlier_details'):
            #     for outlier_type, details in summary['details']['outlier_details'].items():
            #         st.write(f"- {outlier_type.replace('_', ' ').title()}: {details['count']} ({details['percentage']:.1f}%)")
        
        with subtab2:
            # Individual outlier type selection
            # Get outlier types
            pattern_stats = details.get('pattern_stats', {})

            if pattern_stats:
                # Create a dict: "Label (Count)" -> "Unique Pattern String"
                # This matches the logic you used for the gap_detector
                sequence_dict = {
                    f"{p_str} ({stats['count']} cases)": p_str 
                    for p_str, stats in pattern_stats.items()
                }
                
                # Call your helper function
                selected_patterns = dict_to_multicheckbox(
                    sequence_dict,
                    title="Select Sequences to Display",
                    key_prefix="seq_pattern"
                )
                
                print("display sequence tab selected patterns:")
                print(selected_patterns)

                # Store selected types in session state
                st.session_state['selected_seq_patterns'] = selected_patterns
                
                if selected_patterns:
                    st.success(f"✅ {len(selected_patterns)} of {len(sequence_dict)} sequences selected")
            else:
                st.info("No sequence details available")

# endregion

# region Case Arrival Trend
def display_case_arrival_trend_tab():
    """Display Case Arrival Trend pattern details."""
    if not (st.session_state.get('case_arrival_trend_detected', False) and 'case_arrival_trend_detector' in st.session_state):
        return

    detector = st.session_state['case_arrival_trend_detector']
    summary = detector.get_summary()
    layer_visible = st.session_state.get('visible_case_arrival_trend', True)

    if not layer_visible:
        st.caption("Hidden — enable in sidebar")

    direction = summary.get('direction', 'no_trend')
    slope_pct = summary.get('slope_percent', 0)
    p_value = summary.get('p_value', 1.0)
    total_cases = summary.get('total_cases', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cases", total_cases)
    with col2:
        slope_str = f"{slope_pct:+.1f}%" if abs(slope_pct) >= 0.1 else "~0%"
        st.metric("Change/week", slope_str)
    with col3:
        st.metric("p-value", f"{p_value:.4f}")

    # Trend direction indicator
    direction_labels = {
        'increasing': '↗ Increasing',
        'decreasing': '↘ Decreasing',
        'stable': '→ Stable',
        'no_trend': 'No significant trend'
    }
    st.caption(direction_labels.get(direction, 'No significant trend'))
# endregion
