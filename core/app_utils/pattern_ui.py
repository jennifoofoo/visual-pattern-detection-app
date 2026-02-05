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
        'gap_transition': 'visible_gap',
        'OPTICS_cluster': 'visible_cluster',
        'sequence': 'visible_sequence'
    }
    sidebar_key = prefix_to_sidebar.get(key_prefix)
    return st.session_state.get(sidebar_key, True) if sidebar_key else True


def _sync_sidebar_checkbox(key_prefix: str, value: bool):
    """Sync sidebar checkbox with tab selection."""
    # causes crash when selecting all/none in tabs
    pass
    # prefix_to_sidebar = {
    #     'temporal_cluster': 'visible_temporal_cluster',
    #     'outlier_type': 'visible_outlier',
    #     'gap_transition': 'visible_gap'
    # }
    # sidebar_key = prefix_to_sidebar.get(key_prefix)
    # if sidebar_key:
    #     st.session_state[sidebar_key] = value


def list_to_multicheckbox(
    item_list: list, 
    title: str, 
    key_prefix: str, 
    label_func=None,
    sort_metadata: dict = None,
    default_sort: str = "Default"
) -> list:
    """Render multi-checkbox UI for a list of items with sorting."""
    if not item_list:
        return []

    selected_items = []
    parent_visible = _get_parent_visibility(key_prefix)

    if not parent_visible:
        st.caption("Enable in sidebar to configure")
        return []

    with st.container(border=True):
        col_sort, col_all, col_none = st.columns([2, 1, 1])
        
        # Prepare items with labels and metadata for sorting
        labeled_items = []
        for i, item in enumerate(item_list):
            label = label_func(item, i) if label_func else str(item)
            meta = {k: v[i] for k, v in sort_metadata.items()} if sort_metadata else {}
            labeled_items.append({'item': item, 'label': label, 'index': i, 'meta': meta})

        # Sorting UI
        sort_options = ["Default", "A-Z", "Z-A"]
        if sort_metadata:
            sort_options.extend(list(sort_metadata.keys()))
        
        # Determine default index
        def_idx = sort_options.index(default_sort) if default_sort in sort_options else 0
        
        sort_by = col_sort.selectbox(
            "Sort by", options=sort_options, index=def_idx, key=f"{key_prefix}_sort_by", label_visibility="collapsed")

        # Apply Sorting
        if sort_by == "A-Z":
            labeled_items.sort(key=lambda x: x['label'])
        elif sort_by == "Z-A":
            labeled_items.sort(key=lambda x: x['label'], reverse=True)
        elif sort_by in (sort_metadata or {}):
            labeled_items.sort(key=lambda x: x['meta'].get(sort_by, 0), reverse=True)

        with col_all:
            if st.button("All", key=f"{key_prefix}_select_all", use_container_width=True):
                for entry in labeled_items:
                    st.session_state[f"list_checkbox_{key_prefix}_{entry['index']}"] = True
                _sync_sidebar_checkbox(key_prefix, True)
                st.rerun()
        with col_none:
            if st.button("None", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for entry in labeled_items:
                    st.session_state[f"list_checkbox_{key_prefix}_{entry['index']}"] = False
                _sync_sidebar_checkbox(key_prefix, False)
                st.rerun()

        for i, entry in enumerate(labeled_items):
            state_key = f"list_checkbox_{key_prefix}_{entry['index']}"
            if state_key not in st.session_state:
                # Default to selected for the first 5 items in the sorted list
                st.session_state[state_key] = (i < 5)
            
            if st.checkbox(entry['label'], key=state_key):
                selected_items.append(entry['item'])

    return selected_items


def dict_to_multicheckbox(
    data_dict: dict,
    title: str,
    key_prefix: str,
    default_checked: bool = True,
    sort_metadata: dict = None,
    default_sort: str = "Default"
) -> list:
    """Render multi-checkbox UI for a dictionary with sorting."""
    if not data_dict:
        return []

    selected_items = []
    parent_visible = _get_parent_visibility(key_prefix)

    if not parent_visible:
        st.caption("Enable in sidebar to configure")
        return []

    with st.container(border=True):
        col_sort, col_all, col_none = st.columns([2, 1, 1])
        
        # Prepare items for sorting
        labeled_items = []
        for i, (key, value) in enumerate(data_dict.items()):
            meta = {k: v[i] if isinstance(v, list) else v.get(key) for k, v in (sort_metadata or {}).items()}
            labeled_items.append({'key': key, 'value': value, 'index': i, 'meta': meta})

        # Sorting UI
        sort_options = ["Default", "A-Z", "Z-A"]
        if sort_metadata:
            sort_options.extend(list(sort_metadata.keys()))
        
        # Determine default index
        def_idx = sort_options.index(default_sort) if default_sort in sort_options else 0
            
        sort_by = col_sort.selectbox(
            "Sort by", options=sort_options, index=def_idx, key=f"{key_prefix}_sort_by", label_visibility="collapsed")

        # Apply Sorting
        if sort_by == "A-Z":
            labeled_items.sort(key=lambda x: x['key'])
        elif sort_by == "Z-A":
            labeled_items.sort(key=lambda x: x['key'], reverse=True)
        elif sort_by in (sort_metadata or {}):
            labeled_items.sort(key=lambda x: x['meta'].get(sort_by, 0), reverse=True)

        with col_all:
            if st.button("All", key=f"{key_prefix}_select_all", use_container_width=True):
                for entry in labeled_items:
                    st.session_state[f"dict_checkbox_{key_prefix}_{entry['key']}"] = True
                _sync_sidebar_checkbox(key_prefix, True)
                st.rerun()
        with col_none:
            if st.button("None", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for entry in labeled_items:
                    st.session_state[f"dict_checkbox_{key_prefix}_{entry['key']}"] = False
                _sync_sidebar_checkbox(key_prefix, False)
                st.rerun()

        for i, entry in enumerate(labeled_items):
            state_key = f"dict_checkbox_{key_prefix}_{entry['key']}"
            if state_key not in st.session_state:
                # Default to selected for the first 5 items in the sorted list
                # only if default_checked is True (Respect False if provided)
                st.session_state[state_key] = (i < 5) if default_checked else False
            if st.checkbox(entry['key'], key=state_key):
                selected_items.append(entry['value'])

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
    details = summary['details']
    clusters = details.get('clusters', {})
    layer_visible = st.session_state.get('visible_temporal_cluster', True)

    if not layer_visible:
        st.caption("Hidden - enable in sidebar")

    st.caption("**Detects time periods with unusually high event density (bursts of activity) and activity-specific temporal patterns.**", 
               help="Temporal Burst Detection identifies periods when significantly more events happen than usual. "
                    "Activity-Time Clustering finds activities that occur in specific 'blocks' of time.")

    # Calculate refined metrics from bursts
    bursts = clusters.get('temporal_bursts', [])
    total_relevant_events = sum(b['event_count'] for b in bursts)
    
    earliest_b = min(bursts, key=lambda x: x['start_time']) if bursts else None
    latest_b = max(bursts, key=lambda x: x['start_time']) if bursts else None
    maximal_b = max(bursts, key=lambda x: x['event_count']) if bursts else None
    smallest_b = min(bursts, key=lambda x: x['event_count']) if bursts else None

    # Top-level metrics - Summary Row (Simplified)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clusters", summary['count'], help="Total number of distinct temporal patterns found (Bursts).")
    with col2:
        st.metric("Relevant Events", total_relevant_events, help="Total number of events contained within all detected burst periods.")
    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        # --- Timeline and Magnitude Metrics (Moved here) ---
        m_col1, m_col2 = st.columns(2)
        
        def format_time(t):
            if hasattr(t, 'strftime'):
                return t.strftime('%Y-%m-%d %H:%M')
            return f"{t:.2f}"

        with m_col1:
            st.markdown("**Burst Timeline**")
            e_time = format_time(earliest_b['start_time']) if earliest_b else "N/A"
            l_time = format_time(latest_b['start_time']) if latest_b else "N/A"
            e_id = f" (Burst {earliest_b['cluster_id']+1})" if earliest_b else ""
            l_id = f" (Burst {latest_b['cluster_id']+1})" if latest_b else ""
            st.caption(f"🕒 **Earliest{e_id}:** {e_time}", help="The start time/position of the first detected burst.")
            st.caption(f"🕕 **Latest{l_id}:** {l_time}", help="The start time/position of the last detected burst.")

        with m_col2:
            st.markdown("**Burst Magnitude**")
            max_val = maximal_b['event_count'] if maximal_b else 0
            min_val = smallest_b['event_count'] if smallest_b else 0
            max_id = f" (Burst {maximal_b['cluster_id']+1})" if maximal_b else ""
            min_id = f" (Burst {smallest_b['cluster_id']+1})" if smallest_b else ""
            st.caption(f"📈 **Maximal{max_id}:** {max_val} events", help="The highest number of events found in a single burst.")
            st.caption(f"📉 **Smallest{min_id}:** {min_val} events", help="The lowest number of events in a burst (meeting the minimum threshold).")

        st.divider()

        # --- Temporal Bursts ---
        if bursts:
            st.markdown("#### ⚡ Top Temporal Bursts")
            st.caption("Periods of intense activity across cases/activities.", 
                       help="A burst is a short period with a high concentration of events. "
                            "These often indicate batch processing, shift handovers, or system-wide events.")
            
            # Show top 5 by event count
            sorted_bursts = sorted(bursts, key=lambda x: x['event_count'], reverse=True)[:5]
            
            for burst in sorted_bursts:
                b_id = burst['cluster_id'] + 1
                with st.expander(f"Burst {b_id}: {burst['event_count']} events"):
                    dur = burst['duration_seconds']
                    dur_str = f"{dur:.1f}s" if dur < 60 else (f"{dur/60:.1f}m" if dur < 3600 else f"{dur/3600:.1f}h")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Start:** {burst['start_time']}")
                        st.write(f"**Duration:** {dur_str}")
                    with c2:
                        y_axis_name = details.get('y_axis', 'Y')
                        st.write(f"**Events:** {burst['event_count']}")
                        st.write(f"**Unique {y_axis_name}:** {burst.get(f'unique_{y_axis_name}', 'N/A')}")
                    
                    # Concrete Events Table
                    if hasattr(detector, 'cluster_point_indices'):
                        cluster_id = burst.get('cluster_id')
                        indices = detector.cluster_point_indices.get(cluster_id, [])
                        if indices:
                            # Show relevant columns from original df
                            df_burst = detector.df.loc[indices]
                            cols_to_show = ['case_id', 'activity', 'timestamp']
                            # Filter only existing columns
                            existing_cols = [c for c in cols_to_show if c in df_burst.columns]
                            if not existing_cols:
                                # Fallback to first few columns if standard ones not found
                                existing_cols = df_burst.columns[:3].tolist()
                            
                            with st.expander("🔍 View Concrete Events"):
                                st.dataframe(df_burst[existing_cols], hide_index=True, use_container_width=True)

    with subtab2:
        if bursts:
            def burst_label(b, i):
                return f"Burst {b['cluster_id']+1} ({b['event_count']} events)"
            
            sort_meta = {"Events ↓": [b['event_count'] for b in bursts]}
            selected = list_to_multicheckbox(
                bursts, "Select Clusters", "temporal_cluster", label_func=burst_label, 
                sort_metadata=sort_meta, default_sort="Events ↓")
            st.session_state['selected_temporal_clusters'] = selected


def _display_cluster_tab():
    """Display OPTICS Cluster pattern details."""
    if not (st.session_state.get('cluster_detected', False) and 'cluster_detector' in st.session_state):
        return

    detector = st.session_state.cluster_detector
    summary = detector.get_summary()
    details = summary.get('details', {})
    layer_visible = st.session_state.get('visible_cluster', True)

    if not layer_visible:
        st.caption("Hidden - enable in sidebar")

    st.caption("**Shows dense groups of points based on visual proximity (X/Y coordinates).**", 
               help="Point Clustering identifies regions where many events are concentrated. "
                    "This helps find common process paths or hotspot regions in the visualization.")

    # Top-level metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clusters", summary.get('count', 0), help="Number of dense regions accurately identified.")
    with col2:
        noise_count = details.get('noise_count', 0)
        st.metric("Noise Points", noise_count, help="Number of points that do not belong to any cluster (outliers/noise).")
    with col3:
        total = details.get('total_points', 1)
        clustered = details.get('clustered_points', 0)
        coverage = (clustered / total) if total > 0 else 0
        st.metric("Cluster Coverage", f"{coverage:.1%}", help="Percentage of total events that are part of a cluster.")

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
    
        # --- Cluster Breakdown ---
        clusters_data = details.get('clusters', {})
        if clusters_data:
            st.markdown("#### 📊 Top 5 Cluster Breakdown")
            
            # Prepare data: Sort by size descending and take top 5
            sorted_clusters = sorted(clusters_data.items(), key=lambda x: x[1]['size'], reverse=True)[:5]
            
            for cid, stats in sorted_clusters:
                with st.expander(f"Cluster {cid}: {stats['size']} events ({stats['percentage']:.1f}%)"):
                    # Concrete Events Table
                    mask = detector.detected['labels'] == cid
                    indices = detector.detected['original_indices'][mask]
                    
                    if len(indices) > 0 and hasattr(detector, 'df') and detector.df is not None:
                        df_cluster = detector.df.loc[indices]
                        cols_to_show = ['case_id', 'activity', 'timestamp', 'resource']
                        existing_cols = [c for c in cols_to_show if c in df_cluster.columns]
                        if not existing_cols:
                            existing_cols = df_cluster.columns[:3].tolist()
                        
                        st.dataframe(df_cluster[existing_cols], hide_index=True, use_container_width=True)
                    else:
                        st.caption("No event data available")

    with subtab2:
        clusters_data = details.get('clusters', {})
        if clusters_data:
            # Create dict for multi-checkbox
            cluster_dict = {f"Cluster {cid} ({stats['size']} events)": cid for cid, stats in clusters_data.items()}
            sort_meta = {"Events ↓": {f"Cluster {cid} ({stats['size']} events)": stats['size'] for cid, stats in clusters_data.items()}}
            
            # Add Noise Points toggle
            st.checkbox("Show Noise Points", key="show_cluster_noise", value=False, help="Show points that don't belong to any cluster with 'x' markers")
            
            selected = dict_to_multicheckbox(
                cluster_dict, "Select Clusters", "OPTICS_cluster", sort_metadata=sort_meta, default_sort="Events ↓")
            st.session_state['selected_OPTICS_clusters'] = selected
        else:
            st.caption("No clusters available for selection.")


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
        st.metric("Outliers", summary['count'], help="Total count of anomalous events detected.")
    with col2:
        st.metric("Percentage", f"{stats.get('outlier_percentage', 0):.1f}%", help="Proportion of the dataset flagged as outliers.")

    st.caption("**Identifies individual events that are anomalous (statistical outliers).**")

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        # Get detailed summary
        detailed_summary = outlier_pattern.get_outlier_summary()

        # Top reasons for outliers
        if 'top_reasons' in detailed_summary and detailed_summary['top_reasons']:
            st.subheader("🔍 Anomaly Type Distribution")
            
            anomaly_data = []
            for reason_info in detailed_summary['top_reasons']:
                anomaly_data.append({
                    "Type": reason_info['reason'],
                    "Count": reason_info['count'],
                    "Percentage": f"{reason_info['percentage']:.1f}%"
                })
            
            st.dataframe(
                anomaly_data,
                column_config={
                    "Type": st.column_config.TextColumn("Anomaly Type", width="medium"),
                    "Count": st.column_config.NumberColumn("Occurrences", format="%d"),
                    "Percentage": st.column_config.TextColumn("Prop.")
                },
                hide_index=True,
                use_container_width=True
            )

        # Most common specific explanations
        if 'most_common_explanations' in detailed_summary and detailed_summary['most_common_explanations']:
            st.subheader("💡 Top Specific Explanations")
            
            explanation_data = []
            for exp_info in detailed_summary['most_common_explanations']:
                explanation_data.append({
                    "Explanation": exp_info['explanation'],
                    "Count": exp_info['count'],
                    "Percentage": f"{exp_info['percentage']:.1f}%"
                })
                
            st.dataframe(
                explanation_data,
                column_config={
                    "Explanation": st.column_config.TextColumn("Reason", width="large"),
                    "Count": st.column_config.NumberColumn("Count", format="%d"),
                    "Percentage": st.column_config.TextColumn("Prop.")
                },
                hide_index=True,
                use_container_width=True
            )

        # Cases and Activities breakdown
        col_c, col_a = st.columns(2)
        
        with col_c:
            if 'top_outlier_cases' in detailed_summary and detailed_summary['top_outlier_cases']:
                st.subheader("📋 Affected Cases")
                case_data = []
                for case_info in detailed_summary['top_outlier_cases'][:10]:
                    case_data.append({
                        "Case ID": str(case_info['case_id']),
                        "Outliers": case_info['outlier_events']
                    })
                
                st.dataframe(
                    case_data,
                    column_config={
                        "Case ID": st.column_config.TextColumn("Case"),
                        "Outliers": st.column_config.NumberColumn("Count", format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )

        with col_a:
            if 'outlier_activities' in detailed_summary and detailed_summary['outlier_activities']:
                st.subheader("📊 Affected Activities")
                activity_data = []
                for activity_info in detailed_summary['outlier_activities'][:10]:
                    activity_data.append({
                        "Activity": activity_info['activity'],
                        "Outliers": activity_info['outlier_count']
                    })
                
                st.dataframe(
                    activity_data,
                    column_config={
                        "Activity": st.column_config.TextColumn("Activity"),
                        "Outliers": st.column_config.NumberColumn("Count", format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )

    with subtab2:
        # --- Select individual outliers (Multi-reason only) ---
        outlier_pattern = st.session_state.outlier_pattern
        
        # Show only multi-reason outliers
        filtered_outliers = outlier_pattern.get_multi_reason_outliers()
        all_indices = list(filtered_outliers.keys())
        
        multi_reason_count = len(filtered_outliers)
        st.info(f"Showing {multi_reason_count} outliers with multiple contributing factors")

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
            sort_meta = {"Case ID ↓": {}, "Score ↓": {}}
            
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
                
                # Sorting metadata - try to convert case_id to numeric if possible for better sorting
                try:
                    c_id_val = float(case_id)
                except (ValueError, TypeError):
                    c_id_val = str(case_id)
                
                sort_meta["Case ID ↓"][label] = c_id_val
                sort_meta["Score ↓"][label] = score

            selected_indices = dict_to_multicheckbox(
                outlier_dict, "Select Outliers", "outlier_individual", 
                sort_metadata=sort_meta, default_sort="Score ↓")
            st.session_state['selected_outlier_indices'] = selected_indices
        else:
            st.caption("No outliers found.")


def _display_gap_tab():
    """Display Gap Detection pattern details with mode selection and severity distribution."""
    from core.app_utils.pattern_detection import _detect_gaps
    
    # Description moved to top
    st.caption("**Identifies abnormally long time gaps using statistical normality learning.**", 
               help="Gap Detection learns what is 'normal' for each transition or resource from the data itself. "
                    "It then flags any duration that significantly exceeds this normal range (e.g. > 3x longer than usual).")

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
        st.metric("Gaps", summary['count'], help="Total detected delays.")
    with col2:
        if gap_mode == 'resource_inactivity':
            st.metric("Resources", details.get('resources_with_anomalies', 0), help="Number of resources with abnormal idle time.")
        else:
            st.metric("Transitions", details.get(
                'transitions_with_anomalies', 0), help="Number of process steps with delays.")
    with col3:
        st.metric(
            "Worst", f"{worst_severity:.1f}x" if worst_severity > 0 else "-", help="Severity multiplier (e.g. 5x normal duration).")
    with col4:
        with st.popover("..."):
            current = st.session_state.get('gap_min_samples', 15)
            min_samples = st.number_input(
                "Min samples", min_value=3, max_value=30, value=current, key="gap_min_samples_input",
                help="Minimum events per transition to analyze. Lower = more detail but potentially noisy. Default 15 works well.")
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
            st.caption("**Severity Distribution**", 
                       help="Severity indicates how much longer a gap is compared to the normal duration.\n\n"
                            "• **Mild (1-2x)**: Slightly delayed.\n\n"
                            "• **Moderate (2-3x)**: Noticeable delay.\n\n"
                            "• **Severe (3-5x)**: Significant interruption.\n\n"
                            "• **Critical (>5x)**: Extreme delay (requires attention).")
            for label, count in sev_counts.items():
                if count > 0:
                    bar_len = min(count, 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    st.caption(f"`{bar}` {count} {label}")

            # Worst gaps Table
            if abnormal_gaps:
                label = "Transition" if gap_mode == 'transition' else "Resource"
                st.subheader(f"⚠️ Worst by {label}")
                
                # Aggregate data for the table
                summary_map = {}
                for gap in abnormal_gaps:
                    key = gap.get('transition' if gap_mode == 'transition' else 'resource', 'unknown')
                    if key not in summary_map:
                        summary_map[key] = {
                            "Group": key,
                            "Occurrences": 0,
                            "Max Severity": 0,
                            "Max Duration Sec": 0,
                            "Example Case": gap.get('case_id' if gap_mode == 'transition' else 'case_from', 'N/A')
                        }
                    
                    stats = summary_map[key]
                    stats["Occurrences"] += 1
                    stats["Max Severity"] = max(stats["Max Severity"], gap['severity'])
                    if gap['duration'] > stats["Max Duration Sec"]:
                        stats["Max Duration Sec"] = gap['duration']
                        stats["Example Case"] = gap.get('case_id' if gap_mode == 'transition' else 'case_from', 'N/A')

                # Format for display
                table_data = []
                for entry in summary_map.values():
                    dur = entry["Max Duration Sec"]
                    dur_str = f"{dur:.1f}s" if dur < 60 else (f"{dur/60:.1f}m" if dur < 3600 else (f"{dur/3600:.1f}h" if dur < 86400 else f"{dur/86400:.1f}d"))
                    
                    table_data.append({
                        label: entry["Group"],
                        "Occurrences": entry["Occurrences"],
                        "Max Severity": entry["Max Severity"],
                        "Max Duration": dur_str,
                        "Example Case": entry["Example Case"]
                    })

                # Sort by severity
                table_data.sort(key=lambda x: x["Max Severity"], reverse=True)

                st.dataframe(
                    table_data[:10],
                    column_config={
                        label: st.column_config.TextColumn(label, width="medium"),
                        "Occurrences": st.column_config.NumberColumn("Count"),
                        "Max Severity": st.column_config.NumberColumn("Severity", format="%.1fx"),
                        "Max Duration": st.column_config.TextColumn("Longest"),
                        "Example Case": st.column_config.TextColumn("Case ID")
                    },
                    hide_index=True,
                    use_container_width=True
                )

    with subtab2:
        group_stats = details.get(
            'group_stats', details.get('transition_stats', {}))
        if group_stats:
            if gap_mode == 'resource_inactivity':
                # Resource selection
                resource_dict = {}
                sort_meta = {"Occurrences ↓": {}, "Max Severity ↓": {}}
                
                # Fetch max severity per group for sorting
                max_sevs = {}
                for g_idx in abnormal_gaps:
                    r = g_idx.get('resource')
                    sev = g_idx.get('severity', 0)
                    if r not in max_sevs or sev > max_sevs[r]:
                        max_sevs[r] = sev

                for r, s in group_stats.items():
                    m_sev = max_sevs.get(r, 0)
                    label = f"{r} ({s['count']}) - Max Sev: {m_sev:.1f}x"
                    resource_dict[label] = r
                    sort_meta["Occurrences ↓"][label] = s['count']
                    sort_meta["Max Severity ↓"][label] = m_sev

                selected = dict_to_multicheckbox(
                    resource_dict, "Select Resources", "gap_resource", 
                    sort_metadata=sort_meta, default_sort="Max Severity ↓")
                st.session_state['selected_gap_resources'] = selected
            else:
                # Transition selection
                trans_dict = {}
                sort_meta = {"Occurrences ↓": {}, "Max Severity ↓": {}}
                
                # Fetch max severity per transition for sorting
                max_sevs = {}
                for g_idx in abnormal_gaps:
                    t = g_idx.get('transition')
                    sev = g_idx.get('severity', 0)
                    if t not in max_sevs or sev > max_sevs[t]:
                        max_sevs[t] = sev

                for t, s in group_stats.items():
                    m_sev = max_sevs.get(t, 0)
                    label = f"{t} ({s['count']}) - Max Sev: {m_sev:.1f}x"
                    trans_dict[label] = t
                    sort_meta["Occurrences ↓"][label] = s['count']
                    sort_meta["Max Severity ↓"][label] = m_sev

                selected = dict_to_multicheckbox(
                    trans_dict, "Select Transitions", "gap_transition", 
                    sort_metadata=sort_meta, default_sort="Max Severity ↓")
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


    st.caption("**Identifies frequent event patterns (e.g. A->B).**")

    # Strict Mode Toggle
    is_strict = st.session_state.get('sequence_strict_mode', False)
    if st.checkbox("Strict Matching ", value=is_strict, key="seq_strict_toggle",
                   help="Strict Matching: patterns are found only if events occur consecutively (A immediately followed by B).  \n"
                        "Turn off to detect sequences that may have other events in between (A followed eventually by B)."):
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
        st.metric("Unique Patterns", summary.get('unique_patterns_count', '-'), help="Number of distinct sequence types found.")
    with col2:
        st.metric("Total Instances", summary['count'], help="Total occurrence count of all patterns.")
    with col3:
        group_cov = summary.get('group_coverage', 0)
        st.metric("Group Coverage", f"{group_cov:.1%}", help="Percentage of cases that contain at least one pattern.")
    with col4:
        avg_sup = summary.get('avg_support', 0)
        st.metric("Avg Support", f"{avg_sup:.1f}", help="Average frequency of patterns.")

    subtab1, subtab2 = st.tabs(["Overview", "Selection"])

    with subtab1:
        # --- Detailed Statistics ---
        st.markdown("#### 📊 Statistics")
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.markdown("**Pattern Lengths**")
            st.caption(
                f"• Range: **{summary.get('min_pattern_length', 0)} - {summary.get('max_pattern_length', 0)}** items\n"
                f"• Average: **{summary.get('avg_pattern_length', 0):.1f}** items"
            )
            
            # Pattern Length Distribution
            length_dist = summary.get('length_distribution', {})
            if length_dist:
                max_len = max(length_dist.values()) if length_dist else 1
                for length, count in sorted(length_dist.items()):
                    bar_len = int((count / max_len) * 15)
                    bar = "█" * bar_len + "░" * (15 - bar_len)
                    st.caption(f"Length {length}: `{bar}` ({count})")

        with stat_col2:
            st.markdown("**Support & Coverage**")
            min_sup = summary.get('min_support_found', 0)
            max_sup = summary.get('max_support_found', 0)
            min_pct = summary.get('min_support_percentage', 0)
            max_pct = summary.get('max_support_percentage', 0)
            
            st.caption(
                f"• Support Range: **{min_sup} ({min_pct:.1%}) - {max_sup} ({max_pct:.1%})**\n"
                f"• Event Coverage: **{summary.get('event_coverage', 0):.1%}**"
            )

            # Top Frequent Elements
            top_elements = summary.get('top_frequent_elements', {})
            if top_elements:
                st.markdown("**Common Elements**")
                for element, count in list(top_elements.items())[:5]:
                    st.caption(f"• **{element}**: {count}")

        # --- Top Patterns Table ---
        st.divider()
        st.markdown("#### 🏆 Top Detected Patterns")
        
        pattern_stats = details.get('pattern_stats', {})
        if pattern_stats:
            # Create a list of dictionaries for the table
            table_data = []
            for pat_str, stats in pattern_stats.items():
                table_data.append({
                    "Pattern Sequence": pat_str,
                    "Support": stats['count'],
                    "Groups": ", ".join(map(str, stats.get('group_ids', []))),
                    "Support %": f"{stats.get('support_percentile', 0):.1%}",
                    "Length": len(stats.get('sequence', []))
                })
            
            # Sort by support count descending
            table_data.sort(key=lambda x: x['Support'], reverse=True)
            
            # Show top 10
            top_10 = table_data[:10]
            st.dataframe(
                top_10,
                column_config={
                    "Pattern Sequence": st.column_config.TextColumn("Sequence", width="medium"),
                    "Support": st.column_config.NumberColumn("Count", format="%d"),
                    "Groups": st.column_config.TextColumn("Groups", width="large"),
                    "Support %": st.column_config.TextColumn("Support Frequency"),
                    "Length": st.column_config.NumberColumn("Length")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.caption("No pattern statistics available.")

    with subtab2:
        pattern_stats = details.get('pattern_stats', {})
        if pattern_stats:
            seq_dict = {}
            sort_meta = {"Support ↓": {}, "Length ↓": {}}
            
            for p, data in pattern_stats.items():
                p_len = len(data.get('sequence', []))
                label = f"{p} (Support: {data['count']}, Len: {p_len})"
                seq_dict[label] = p
                sort_meta["Support ↓"][label] = data['count']
                sort_meta["Length ↓"][label] = p_len
            
            selected = dict_to_multicheckbox(
                seq_dict, "Select Sequences", "seq_pattern", default_checked=False, 
                sort_metadata=sort_meta, default_sort="Support ↓")
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

    st.caption("**Analyzes the rate of new cases over time (increasing/decreasing).**")

    direction = summary.get('direction', 'no_trend')
    slope_pct = summary.get('slope_percent', 0)
    p_value = summary.get('p_value', 1.0)
    total_cases = summary.get('total_cases', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cases", total_cases, help="Total number of cases analyzed.")
    with col2:
        slope_str = f"{slope_pct:+.1f}%" if abs(slope_pct) >= 0.1 else "~0%"
        st.metric("Change/week", slope_str, help="Estimated percentage growth (+) or decline (-) in case volume per week.")
    with col3:
        st.metric("p-value", f"{p_value:.4f}", help="Statistical significance (values < 0.05 indicate a strong trend).")

    direction_labels = {
        'increasing': '↗ Increasing',
        'decreasing': '↘ Decreasing',
        'stable': '→ Stable',
        'no_trend': 'No significant trend'
    }
    st.caption(direction_labels.get(direction, 'No significant trend'))

    # Render trend sparkline
    _render_trend_sparkline(direction)


def _render_trend_sparkline(direction: str):
    """Render a sparkline chart showing case arrival trend."""
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    plot_config = st.session_state.get('current_plot_config')
    if not plot_config:
        return

    df = plot_config.get('df_selected')
    if df is None or 'case_id' not in df.columns or 'actual_time' not in df.columns:
        return

    # Apply time filter if active
    time_range = st.session_state.get('time_filter_range')
    if time_range is not None:
        start, end = time_range
        if df['actual_time'].dt.tz is not None:
            start = pd.Timestamp(start).tz_localize(df['actual_time'].dt.tz)
            end = pd.Timestamp(end).tz_localize(df['actual_time'].dt.tz)
        df = df[(df['actual_time'] >= start) & (df['actual_time'] <= end)]

    try:
        # Get case arrivals (min timestamp per case)
        case_arrivals = df.groupby('case_id')['actual_time'].min().reset_index()
        case_arrivals.columns = ['case_id', 'arrival_time']
        case_arrivals['arrival_time'] = pd.to_datetime(case_arrivals['arrival_time'])

        # Aggregate by week
        arrival_counts = case_arrivals.set_index('arrival_time').resample('W').size()

        if len(arrival_counts) < 3:
            return

        # Create sparkline figure
        fig = go.Figure()

        # Area chart for weekly counts
        fig.add_trace(go.Scatter(
            x=arrival_counts.index,
            y=arrival_counts.values,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.2)',
            line=dict(color='rgba(99, 102, 241, 0.6)', width=1.5),
            name='Cases/Week',
            hovertemplate='%{x|%Y-%m-%d}<br>Cases: %{y}<extra></extra>'
        ))

        # Add trend line (simple linear regression)
        x_numeric = np.arange(len(arrival_counts))
        y_values = arrival_counts.values

        # Linear regression
        slope, intercept = np.polyfit(x_numeric, y_values, 1)
        trend_line = slope * x_numeric + intercept

        # Determine trend color
        if direction == 'increasing':
            trend_color = 'rgba(34, 197, 94, 0.9)'  # Green
            trend_text = '↗ Trend'
        elif direction == 'decreasing':
            trend_color = 'rgba(239, 68, 68, 0.9)'  # Red
            trend_text = '↘ Trend'
        else:
            trend_color = 'rgba(156, 163, 175, 0.9)'  # Gray
            trend_text = '→ Trend'

        fig.add_trace(go.Scatter(
            x=arrival_counts.index,
            y=trend_line,
            mode='lines',
            line=dict(color=trend_color, width=2.5, dash='dash'),
            name=trend_text,
            hoverinfo='skip'
        ))

        # Compact layout for sparkline
        fig.update_layout(
            height=150,
            margin=dict(l=50, r=20, t=10, b=30),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=10)
            ),
            xaxis=dict(
                showgrid=False,
                showticklabels=True,
                tickfont=dict(size=9),
                tickformat='%b %Y'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)',
                showticklabels=True,
                tickfont=dict(size=9),
                title=dict(text='Cases/Week', font=dict(size=9))
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        st.plotly_chart(fig, width='stretch', key="trend_sparkline_tab")

    except Exception:
        # Silently fail - sparkline is optional
        pass
