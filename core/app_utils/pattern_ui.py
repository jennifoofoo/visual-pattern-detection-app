"""
Pattern UI - Pattern Summary Tabs and Sub-pattern Selection Helpers.

This module handles all pattern-related UI:
- Pattern summary tabs (Temporal, Outlier, Gap, Sequence, Case Arrival Trend)
- Multi-checkbox helpers for sub-pattern selection
- Visibility sync between sidebar and tabs
"""

import streamlit as st


# =============================================================================
# === UI Helpers ===
# =============================================================================

def _get_parent_visibility(key_prefix: str) -> bool:
    """Get visibility state of parent pattern from sidebar."""
    prefix_to_sidebar = {
        'temporal_cluster': 'visible_temporal_cluster',
        'outlier_type': 'visible_outlier',
        'gap_transition': 'visible_gap'
    }
    sidebar_key = prefix_to_sidebar.get(key_prefix)
    return st.session_state.get(sidebar_key, True) if sidebar_key else True


def _sync_sidebar_checkbox(key_prefix: str, value: bool):
    """Sync sidebar checkbox with tab selection."""
    prefix_to_sidebar = {
        'temporal_cluster': 'visible_temporal_cluster',
        'outlier_type': 'visible_outlier',
        'gap_transition': 'visible_gap'
    }
    sidebar_key = prefix_to_sidebar.get(key_prefix)
    if sidebar_key:
        st.session_state[sidebar_key] = value


def list_to_multicheckbox(item_list: list, title: str, key_prefix: str) -> list:
    """Render multi-checkbox UI for a list of items."""
    if not item_list:
        return []

    selected_items = []
    parent_visible = _get_parent_visibility(key_prefix)

    if not parent_visible:
        st.caption("Enable in sidebar to configure")
        return []

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("All", key=f"{key_prefix}_select_all", use_container_width=True):
                for index in range(len(item_list)):
                    st.session_state[f"list_checkbox_{key_prefix}_{index}"] = True
                _sync_sidebar_checkbox(key_prefix, True)
                st.rerun()
        with col_b:
            if st.button("None", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for index in range(len(item_list)):
                    st.session_state[f"list_checkbox_{key_prefix}_{index}"] = False
                _sync_sidebar_checkbox(key_prefix, False)
                st.rerun()

        for index, item in enumerate(item_list):
            state_key = f"list_checkbox_{key_prefix}_{index}"
            if state_key not in st.session_state:
                st.session_state[state_key] = True
            if st.checkbox(str(item), key=state_key):
                selected_items.append(item)

    return selected_items


def dict_to_multicheckbox(
    data_dict: dict,
    title: str,
    key_prefix: str,
    default_checked: bool = True
) -> list:
    """Render multi-checkbox UI for a dictionary."""
    if not data_dict:
        return []

    selected_items = []
    parent_visible = _get_parent_visibility(key_prefix)

    if not parent_visible:
        st.caption("Enable in sidebar to configure")
        return []

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("All", key=f"{key_prefix}_select_all", use_container_width=True):
                for key in data_dict.keys():
                    st.session_state[f"dict_checkbox_{key_prefix}_{key}"] = True
                _sync_sidebar_checkbox(key_prefix, True)
                st.rerun()
        with col_b:
            if st.button("None", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for key in data_dict.keys():
                    st.session_state[f"dict_checkbox_{key_prefix}_{key}"] = False
                _sync_sidebar_checkbox(key_prefix, False)
                st.rerun()

        for key, value in data_dict.items():
            state_key = f"dict_checkbox_{key_prefix}_{key}"
            if state_key not in st.session_state:
                st.session_state[state_key] = default_checked
            if st.checkbox(key, key=state_key):
                selected_items.append(value)

    return selected_items


# =============================================================================
# === Pattern Summary ===
# =============================================================================

def _is_any_pattern_detected() -> bool:
    """Check if any pattern has been detected."""
    return (
        st.session_state.get('temporal_detected', False) or
        st.session_state.get('outlier_detected', False) or
        st.session_state.get('case_arrival_trend_detected', False) or
        st.session_state.get('cluster_detected', False) or
        st.session_state.get('sequence_detected', False) or
        ('gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None)
    )


def _get_detected_pattern_tabs() -> tuple:
    """Get lists of detected pattern names and types for tab creation."""
    tabs_names = []
    tabs_types = []

    if st.session_state.get('temporal_detected', False):
        tabs_names.append("Temporal Clusters")
        tabs_types.append('temporal')
    if st.session_state.get('cluster_detected', False):
        tabs_names.append("Clusters (OPTICS)")
        tabs_types.append('cluster')
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


def _display_pattern_tab(pattern_type: str):
    """Route to appropriate pattern tab renderer."""
    if pattern_type == 'temporal':
        _display_temporal_cluster_tab()
    elif pattern_type == 'cluster':
        _display_cluster_tab()
    elif pattern_type == 'outlier':
        _display_outlier_tab()
    elif pattern_type == 'gap':
        _display_gap_tab()
    elif pattern_type == 'sequence':
        _display_sequence_tab()
    elif pattern_type == 'case_arrival_trend':
        _display_case_arrival_trend_tab()


# =============================================================================
# === Pattern Tab Displays ===
# =============================================================================

def _display_temporal_cluster_tab():
    """Display Temporal Cluster pattern details."""
    if not (st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state):
        return

    detector = st.session_state.temporal_clusters
    summary = detector.get_summary()
    layer_visible = st.session_state.get('visible_temporal_cluster', True)

    if not layer_visible:
        st.caption("Hidden - enable in sidebar")

    st.metric("Clusters", summary['count'])

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        st.caption(summary['details']['summary_text'])

    with subtab2:
        if hasattr(detector, 'clusters') and 'temporal_bursts' in detector.clusters:
            cluster_list = detector.clusters['temporal_bursts']
            selected = list_to_multicheckbox(
                cluster_list, "Select Clusters", "temporal_cluster")
            st.session_state['selected_temporal_clusters'] = selected


def _display_cluster_tab():
    """Display OPTICS Cluster pattern details."""
    if not (st.session_state.get('cluster_detected', False) and 'cluster_detector' in st.session_state):
        return

    detector = st.session_state.cluster_detector
    summary = detector.get_summary()
    layer_visible = st.session_state.get('visible_cluster', True)

    if not layer_visible:
        st.caption("Hidden - enable in sidebar")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Clusters", summary.get('count', 0))
    with col2:
        noise_count = summary.get('details', {}).get('noise_count', 0)
        st.metric("Noise Points", noise_count)

    st.caption(
        f"Algorithm: {summary.get('details', {}).get('algorithm', 'OPTICS')}")


def _display_outlier_tab():
    """Display Outlier Detection pattern details."""
    if not (st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state):
        return

    outlier_pattern = st.session_state.outlier_pattern
    summary = outlier_pattern.get_summary()
    layer_visible = st.session_state.get('visible_outlier', True)

    if not layer_visible:
        st.caption("Hidden - enable in sidebar")

    stats = summary['details'].get('statistics', {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Outliers", summary['count'])
    with col2:
        st.metric("Percentage", f"{stats.get('outlier_percentage', 0):.1f}%")

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        # Get detailed summary
        detailed_summary = outlier_pattern.get_outlier_summary()

        # Top reasons for outliers
        if 'top_reasons' in detailed_summary:
            st.subheader("🔍 Most Common Anomaly Types")
            for reason_info in detailed_summary['top_reasons']:
                st.caption(
                    f"**{reason_info['reason']}**: {reason_info['count']} events "
                    f"({reason_info['percentage']:.1f}% of outliers)"
                )

        # Most common specific explanations
        if 'most_common_explanations' in detailed_summary and detailed_summary['most_common_explanations']:
            st.subheader("💡 Most Frequent Specific Reasons")
            for exp_info in detailed_summary['most_common_explanations']:
                st.caption(
                    f"• *{exp_info['explanation']}* — {exp_info['count']} times "
                    f"({exp_info['percentage']:.1f}%)"
                )

        # Top affected cases
        if 'top_outlier_cases' in detailed_summary and detailed_summary['top_outlier_cases']:
            st.subheader("📋 Most Affected Cases")
            top_cases = detailed_summary['top_outlier_cases'][:5]
            for case_info in top_cases:
                st.caption(
                    f"• Case **{case_info['case_id']}**: {case_info['outlier_events']} outlier events")

        # Top affected activities
        if 'outlier_activities' in detailed_summary and detailed_summary['outlier_activities']:
            st.subheader("📊 Most Affected Activities")
            top_activities = detailed_summary['outlier_activities'][:5]
            for activity_info in top_activities:
                st.caption(
                    f"• **{activity_info['activity']}**: {activity_info['outlier_count']} outliers")

    with subtab2:
        # --- Select individual outliers ---
        outlier_pattern = st.session_state.outlier_pattern
        all_indices = outlier_pattern.outliers.get('combined', [])

        if all_indices:
            df = outlier_pattern.df
            case_col = outlier_pattern._find_column(
                'case_id', 'case:concept:name', 'caseid', 'trace_id')
            activity_col = outlier_pattern._find_column(
                'activity', 'concept:name', 'event_name', 'activity_name')
            score_dict = outlier_pattern.outlier_scores if hasattr(
                outlier_pattern, 'outlier_scores') else {}
            outlier_explanations = outlier_pattern.outlier_explanations if hasattr(
                outlier_pattern, 'outlier_explanations') else {}

            # Create dict for multi-checkbox selection
            outlier_dict = {}
            for idx in all_indices:
                case_id = df.loc[idx,
                                 case_col] if case_col and case_col in df.columns else 'N/A'
                activity = df.loc[idx,
                                  activity_col] if activity_col and activity_col in df.columns else 'N/A'
                score = round(score_dict.get(idx, 0), 3)

                # Get explanation (now a simple string)
                explanation = outlier_explanations.get(
                    idx, 'No reason available')

                # Create display label with case, activity, reason, and score
                label = f"Case {case_id} | {activity} | {explanation} (Score: {score})"
                outlier_dict[label] = idx

            selected_indices = dict_to_multicheckbox(
                outlier_dict, "Select Outliers", "outlier_individual")
            st.session_state['selected_outlier_indices'] = selected_indices
        else:
            st.caption("No outliers found.")


def _display_gap_tab():
    """Display Gap Detection pattern details with mode selection and severity distribution."""
    from core.app_utils.pattern_detection import _detect_gaps

    # Mode selection dropdown
    gap_mode_options = {
        "transition": "Transition gaps (within cases)",
        "resource_inactivity": "Resource inactivity (within resources)"
    }
    current_mode = st.session_state.get('gap_mode', 'transition')

    # Check if resource column exists for resource_inactivity mode
    plot_config = st.session_state.get('current_plot_config', {})
    df_selected = plot_config.get('df_selected')
    has_resource_col = df_selected is not None and 'resource' in df_selected.columns

    selected_mode = st.selectbox(
        "Gap Mode",
        options=list(gap_mode_options.keys()),
        format_func=lambda x: gap_mode_options[x],
        index=list(gap_mode_options.keys()).index(current_mode),
        key="gap_mode_selector",
        disabled=False
    )

    # Explanatory text for semantic distinction
    if selected_mode == "transition":
        st.caption("*Process-flow gaps: delays between activities within a case*")
    else:
        st.caption(
            "*Resource timeline gaps: periods without events (not process-flow)*")
        if not has_resource_col:
            st.warning("Resource inactivity requires 'resource' column in data")

    # Re-run gap detection if mode changed
    if selected_mode != current_mode:
        st.session_state['gap_mode'] = selected_mode
        # Clear mode-specific selections
        if 'selected_gap_transitions' in st.session_state:
            del st.session_state['selected_gap_transitions']
        if 'selected_gap_resources' in st.session_state:
            del st.session_state['selected_gap_resources']
        if plot_config and df_selected is not None:
            _detect_gaps(
                plot_config['x_col'], plot_config['y_col'], df_selected,
                gap_mode=selected_mode
            )
            st.rerun()

    # Check if we have detection results
    if not ('gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None):
        st.caption("No gaps detected for this mode")
        return

    gap_detector = st.session_state['gap_detector']
    summary = gap_detector.get_summary()
    details = summary['details']
    abnormal_gaps = details.get('abnormal_gaps', [])
    gap_mode = details.get('gap_mode', 'transition')
    layer_visible = st.session_state.get('visible_gap', True)

    # Compute severity distribution
    sev_counts = {'Mild (1-2x)': 0, 'Moderate (2-3x)': 0,
                  'Severe (3-5x)': 0, 'Critical (>5x)': 0}
    worst_severity = 0
    for gap in abnormal_gaps:
        sev = gap.get('severity', 1)
        worst_severity = max(worst_severity, sev)
        if sev >= 5:
            sev_counts['Critical (>5x)'] += 1
        elif sev >= 3:
            sev_counts['Severe (3-5x)'] += 1
        elif sev >= 2:
            sev_counts['Moderate (2-3x)'] += 1
        else:
            sev_counts['Mild (1-2x)'] += 1

    # Header metrics - mode-specific labels
    col1, col2, col3, col4 = st.columns([1, 1, 1, 0.3])
    with col1:
        st.metric("Gaps", summary['count'])
    with col2:
        if gap_mode == 'resource_inactivity':
            st.metric("Resources", details.get('resources_with_anomalies', 0))
        else:
            st.metric("Transitions", details.get(
                'transitions_with_anomalies', 0))
    with col3:
        st.metric(
            "Worst", f"{worst_severity:.1f}x" if worst_severity > 0 else "-")
    with col4:
        with st.popover("..."):
            current = st.session_state.get('gap_min_samples', 15)
            min_samples = st.number_input(
                "Min samples", 3, 30, current, key="gap_min_samples_input")
            if min_samples != current and st.button("Apply", key="gap_apply"):
                st.session_state['gap_min_samples'] = min_samples
                if plot_config and df_selected is not None:
                    _detect_gaps(
                        plot_config['x_col'], plot_config['y_col'], df_selected,
                        min_samples, gap_mode=gap_mode)
                    st.rerun()

    if not layer_visible:
        st.caption("Hidden - enable in sidebar")

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        # Severity distribution
        if abnormal_gaps:
            st.caption("**Severity Distribution**")
            for label, count in sev_counts.items():
                if count > 0:
                    bar_len = min(count, 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    st.caption(f"`{bar}` {count} {label}")

            # Worst gaps - mode-specific labels
            if gap_mode == 'resource_inactivity':
                st.caption("**Worst by Resource**")
                worst_per_group = {}
                for gap in abnormal_gaps:
                    resource = gap.get('resource', 'unknown')
                    if resource not in worst_per_group or gap['severity'] > worst_per_group[resource]['severity']:
                        worst_per_group[resource] = gap

                sorted_worst = sorted(worst_per_group.values(
                ), key=lambda g: g['severity'], reverse=True)[:3]
                for gap in sorted_worst:
                    dur_h = gap['duration'] / 3600
                    dur_str = f"{dur_h:.1f}h" if dur_h < 24 else f"{dur_h/24:.1f}d"
                    st.caption(
                        f"⚠️ {gap['resource']}: {dur_str} ({gap['severity']:.1f}x)")
            else:
                st.caption("**Worst by Transition**")
                worst_per_trans = {}
                for gap in abnormal_gaps:
                    trans = gap.get('transition', 'unknown')
                    if trans not in worst_per_trans or gap['severity'] > worst_per_trans[trans]['severity']:
                        worst_per_trans[trans] = gap

                sorted_worst = sorted(worst_per_trans.values(
                ), key=lambda g: g['severity'], reverse=True)[:3]
                for gap in sorted_worst:
                    dur_h = gap['duration'] / 3600
                    dur_str = f"{dur_h:.1f}h" if dur_h < 24 else f"{dur_h/24:.1f}d"
                    st.caption(
                        f"⚠️ {gap.get('transition', 'N/A')}: {dur_str} ({gap['severity']:.1f}x)")

    with subtab2:
        group_stats = details.get(
            'group_stats', details.get('transition_stats', {}))
        if group_stats:
            if gap_mode == 'resource_inactivity':
                # Resource selection
                resource_dict = {
                    f"{r} ({s['count']})": r for r, s in group_stats.items()}
                selected = dict_to_multicheckbox(
                    resource_dict, "Select Resources", "gap_resource")
                st.session_state['selected_gap_resources'] = selected
            else:
                # Transition selection
                trans_dict = {f"{t} ({s['count']})": t for t,
                              s in group_stats.items()}
                selected = dict_to_multicheckbox(
                    trans_dict, "Select Transitions", "gap_transition")
                st.session_state['selected_gap_transitions'] = selected


def _display_sequence_tab():
    """Display Sequence Detection pattern details."""
    if not (st.session_state.get('sequence_detected', False) and 'sequence_detector' in st.session_state):
        return

    detector = st.session_state.sequence_detector
    summary = detector.get_summary()
    details = summary['details']
    layer_visible = st.session_state.get('visible_sequence', True)

    if not layer_visible:
        st.caption("Hidden - enable in sidebar")

    # Strict Mode Toggle
    is_strict = st.session_state.get('sequence_strict_mode', False)
    if st.checkbox("Strict Matching (No Gaps)", value=is_strict, key="seq_strict_toggle"):
        if not is_strict:
            st.session_state['sequence_strict_mode'] = True
            # Trigger re-detection
            from core.app_utils.pattern_detection import _detect_sequences
            plot_config = st.session_state.get('current_plot_config', {})
            if plot_config:
                 _detect_sequences(
                     plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'], 
                     plot_config['df_selected']
                 )
            st.rerun()
    else:
        if is_strict:
            st.session_state['sequence_strict_mode'] = False
            # Trigger re-detection
            from core.app_utils.pattern_detection import _detect_sequences
            plot_config = st.session_state.get('current_plot_config', {})
            if plot_config:
                 _detect_sequences(
                     plot_config['x_col'], plot_config['y_col'], plot_config['dots_config_col'], 
                     plot_config['df_selected']
                 )
            st.rerun()

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Unique Patterns", summary.get('unique_patterns_count', '-'))
    with col2:
        st.metric("Total Instances", summary['count'])
    with col3:
        group_cov = summary.get('group_coverage', 0)
        st.metric("Group Coverage", f"{group_cov:.1%}")
    with col4:
        avg_sup = summary.get('avg_support', 0)
        st.metric("Avg Support", f"{avg_sup:.1f}")

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Pattern Length Distribution
            length_dist = summary.get('length_distribution', {})
            if length_dist:
                st.markdown("**Pattern Length Distribution**")
                max_len = max(length_dist.values()) if length_dist else 1
                for length, count in sorted(length_dist.items()):
                    # Simple ASCII bar chart
                    bar_len = int((count / max_len) * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    st.caption(f"Length {length}: `{bar}` ({count})")
            
            # Additional stats
            event_cov = summary.get('event_coverage', 0)
            st.markdown("**Coverage Stats**")
            st.caption(f"• Event Coverage: {event_cov:.1%}")
            
            min_sup = summary.get('min_support_found', 0)
            max_sup = summary.get('max_support_found', 0)
            min_pct = summary.get('min_support_percentage', 0)
            max_pct = summary.get('max_support_percentage', 0)
            
            support_range = f"{min_sup} ({min_pct:.1%}) - {max_sup} ({max_pct:.1%})"
            st.caption(f"• Support Range: {support_range}")

        with col_b:
            # Top Frequent Elements
            top_elements = summary.get('top_frequent_elements', {})
            if top_elements:
                st.markdown("**Most Frequent Elements in Patterns**")
                for element, count in top_elements.items():
                    st.caption(f"• **{element}**: {count}")

    with subtab2:
        pattern_stats = details.get('pattern_stats', {})
        if pattern_stats:
            # Sort by support count for better UX
            sorted_patterns = sorted(
                pattern_stats.items(), 
                key=lambda item: item[1]['count'], 
                reverse=True
            )
            
            seq_dict = {
                f"{p} (Support: {data['count']})": p 
                for p, data in sorted_patterns
            }
            
            selected = dict_to_multicheckbox(
                seq_dict, "Select Sequences", "seq_pattern", default_checked=False)
            st.session_state['selected_seq_patterns'] = selected


def _display_case_arrival_trend_tab():
    """Display Case Arrival Trend pattern details."""
    if not (st.session_state.get('case_arrival_trend_detected', False) and 'case_arrival_trend_detector' in st.session_state):
        return

    detector = st.session_state['case_arrival_trend_detector']
    summary = detector.get_summary()
    layer_visible = st.session_state.get('visible_case_arrival_trend', True)

    if not layer_visible:
        st.caption("Hidden - enable in sidebar")

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

    direction_labels = {
        'increasing': '↗ Increasing',
        'decreasing': '↘ Decreasing',
        'stable': '→ Stable',
        'no_trend': 'No significant trend'
    }
    st.caption(direction_labels.get(direction, 'No significant trend'))
