import streamlit as st
import plotly.express as px
from core.evaluation.ollama import OllamaEvaluator
from core.visualization.visualizer import plot_dotted_chart
from core.evaluation.summary_generator import summarize_event_log
from core.data_processing import load_xes_log, DataPreprocessor
from core.detection.cluster_pattern import ClusterPattern
from core.detection.gap_pattern import GapPattern


# Maps user selection to the column name generated in load_xes_log
X_AXIS_COLUMN_MAP = {
    'Actual time': 'actual_time',               # 0.
    'Relative time': 'relative_time',           # 1. Relative Time (seconds)
    # 2. Relative Ratio (time-based [0, 1])
    'Relative ratio': 'relative_ratio',
    # 3. Global Logical Time (index)
    'Logical time': 'logical_time',
    # 4. Logical Relative (global index)
    'Logical relative': 'logical_relative'
}

# Maps user selection to the column name available in the DataFrame
Y_AXIS_COLUMN_MAP = {
    'Case ID': 'case_id',
    'Activity': 'activity',
    # Used 'event_index_in_trace' in latest load_xes_log
    'Event Index': 'event_index',
    # Assuming 'resource' is in the log/DataFrame
    'Resource': 'resource',
    'Variant': 'variant'                    # Will be calculated if selected
}

# Mapping for Dot Colors (Color/Dots Config)
DOTS_COLOR_MAP = {
    'Case ID (Default)': 'case_id',
    'Activity': 'activity',
    'Resource': 'resource',
    'Event Index (in trace)': 'event_index_in_trace',
    'Global Logical Time': 'timestamp_logical_global',
}


def main():
    st.title('Event Log Dotted Chart')

    xes_path = st.text_input('Enter XES log file path:',
                             'data/Hospital_log.xes', key='xes_path_input')

    x_axis_options = list(X_AXIS_COLUMN_MAP.keys())
    y_axis_options = list(Y_AXIS_COLUMN_MAP.keys())
    dots_config_options = list(DOTS_COLOR_MAP.keys())

    x_axis = st.selectbox('Select x-axis:', x_axis_options)
    y_axis = st.selectbox('Select y-axis:', y_axis_options)
    dots_config_label = st.selectbox(
        'Select Dot Color/Configuration:', dots_config_options)

    # 1. Load Data and Store in session_state
    if st.button('Load and Plot'):
        try:
            df = load_xes_log(xes_path)
            if df.empty:
                st.warning("The log file was loaded but contains no events.")
                return

            st.session_state['df'] = df
            st.session_state['summary'] = summarize_event_log(df)

            st.success(f"Log loaded successfully with {len(df)} events.")
            st.subheader('Event Log Summary')
            for k, v in st.session_state['summary'].items():
                st.write(f"**{k}:** {v}")

        except Exception as e:
            st.error(f"Error loading XES log: {e}")
            st.stop()

        # 2. Plotting Logic (always runs if data is in session_state)
        if 'df' in st.session_state:
            df_base = st.session_state['df']

            # Determine the columns to plot
            x_col = X_AXIS_COLUMN_MAP[x_axis]
            y_col = Y_AXIS_COLUMN_MAP[y_axis]
            # Default to case_id
            dots_config_col = DOTS_COLOR_MAP[dots_config_label]
            # Create a copy for modification if needed (e.g., variant calculation)
            df_selected = df_base.copy()

            # region Varian Calculation Logic
            # --- Handle Variant Calculation (Your existing logic) ---
            if y_axis == 'Variant':
                if 'variant' not in df_selected.columns:
                    n = 10  # Number of most common variants to show

                    # Combine activities into a string per case
                    case_variants = df_selected.groupby('case_id')['activity'].apply(
                        lambda x: '-'.join(x.astype(str)))
                    variant_counts = case_variants.value_counts()
                    top_variants = variant_counts.head(n).index.tolist()

                    # Map the variant string back to the event DataFrame
                    df_selected['variant'] = df_selected['case_id'].map(
                        case_variants)

                    # Filter to only top n variants for a cleaner chart
                    df_selected = df_selected[df_selected['variant'].isin(
                        top_variants)].copy()
            # --- End Variant Logic ---
            # endregion

            # Check for missing values in the selected columns
            if df_selected[x_col].isnull().any() or df_selected[y_col].isnull().any():
                # st.warning(f"Skipping plot: Selected column '{x_col}' or '{y_col}' contains missing values (NaN/None).")
                # Optionally, filter them out:
                df_selected.dropna(subset=[x_col, y_col], inplace=True)
                return

            # Generate the Plotly Scatter (Dotted Chart)
            fig = px.scatter(
                df_selected,
                x=x_col,                          # Use the dynamically selected X-column
                y=y_col,                          # Use the dynamically selected Y-column
                color=dots_config_col,                  # Color by trace
                title=f"Dotted Chart: {y_axis} vs {x_axis}",
                labels={x_col: x_axis, y_col: y_axis,
                        dots_config_col: dots_config_label},
                hover_data=['activity', 'event_index', 'actual_time']
            )

            # Optional: Improve visual appearance for better density/clarity
            fig.update_traces(marker=dict(size=5, opacity=0.8))
            # Usually too many case IDs to show legend
            fig.update_layout(showlegend=False)

            # Add gap visualization if gaps were detected
            if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
                fig = st.session_state['gap_detector'].visualize(df_selected, fig)

            st.plotly_chart(fig, width='stretch')

            # Store the current plot configuration for clustering
            st.session_state['current_plot_config'] = {
                'x_col': x_col,
                'y_col': y_col,
                'dots_config_col': dots_config_col,
                'x_axis_label': x_axis,
                'y_axis_label': y_axis,
                'dots_config_label': dots_config_label,
                'df_selected': df_selected
            }

    # Clustering Section
    if 'df' in st.session_state and 'current_plot_config' in st.session_state:
        st.divider()
        st.subheader("Cluster Analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Clustering algorithm selection
            algorithm = st.selectbox(
                "Select Clustering Algorithm:",
                ["OPTICS", "DBSCAN"],
                help="OPTICS: Good for varying density clusters. DBSCAN: Good for arbitrary shapes."
            )

        with col2:
            # Algorithm-specific parameters
            if algorithm == "OPTICS":
                min_samples = st.slider(
                    "Min Samples", 3, 20, 5, help="Minimum points to form a cluster")
                xi = st.slider("Xi (cluster extraction)", 0.01, 0.2,
                               0.05, help="Determines how clusters are extracted")
                max_eps = st.number_input(
                    "Max Eps", 0.1, 10.0, 2.0, help="Maximum distance between points")
                params = {'min_samples': min_samples,
                          'xi': xi, 'max_eps': max_eps}

            elif algorithm == "DBSCAN":
                eps = st.slider("Eps (neighborhood size)", 0.1, 5.0, 0.5,
                                help="Maximum distance between points in same cluster")
                min_samples = st.slider(
                    "Min Samples", 3, 20, 5, help="Minimum points to form a cluster")
                params = {'eps': eps, 'min_samples': min_samples}

        # Cluster detection button
        if st.button("Find Clusters", type="primary"):
            try:
                plot_config = st.session_state['current_plot_config']
                df_selected = plot_config['df_selected']

                # Create view configuration for clustering
                # Determine view type based on column types
                preprocessor = DataPreprocessor()
                view_type = preprocessor._determine_view_type(df_selected, plot_config['x_col'], plot_config['y_col'])
                
                view_config = {
                    'x': plot_config['x_col'],
                    'y': plot_config['y_col'],
                    'view': view_type  # View sollte Teil der Config sein
                }

                # Create cluster detector
                with st.spinner(f"Running {algorithm} clustering..."):
                    cluster_detector = ClusterPattern(
                        view_config=view_config,
                        algorithm=algorithm.lower(),
                        **params
                    )

                    # Detect clusters
                    cluster_detector.detect(df_selected)

                    if cluster_detector.detected is None:
                        st.warning(
                            "No clusters found with current parameters. Try adjusting the settings.")
                    else:
                        # Store clustering results
                        st.session_state['cluster_detector'] = cluster_detector

                        # Get cluster summary
                        summary = cluster_detector.get_cluster_summary()

                        # Display results
                        st.success(
                            f"Clustering completed! Found {summary.get('total_clusters', 0)} clusters")

                        # Show cluster statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Clusters", summary.get(
                                'total_clusters', 0))
                        with col2:
                            st.metric("Clustered Points", summary.get(
                                'clustered_points', 0))
                        with col3:
                            st.metric("Noise Points", summary.get(
                                'noise_points', 0))
                        with col4:
                            trend_points = summary.get('trend_points', 0)
                            st.metric("Trend Points", trend_points)

                        # Show quality metrics if available
                        if summary.get('quality_metrics') and summary['quality_metrics'].get('overall'):
                            silhouette_score = summary['quality_metrics']['overall']
                            st.info(f"Silhouette Score: {silhouette_score:.3f} "
                                    f"({'Good' if silhouette_score > 0.5 else 'Moderate' if silhouette_score > 0.3 else 'Poor'} quality)")

                        # Create enhanced visualization
                        fig_clustered = px.scatter(
                            df_selected,
                            x=plot_config['x_col'],
                            y=plot_config['y_col'],
                            color=plot_config['dots_config_col'],
                            title=f"Dotted Chart with {algorithm} Clusters: {plot_config['y_axis_label']} vs {plot_config['x_axis_label']}",
                            labels={
                                plot_config['x_col']: plot_config['x_axis_label'],
                                plot_config['y_col']: plot_config['y_axis_label'],
                                plot_config['dots_config_col']: plot_config['dots_config_label']
                            },
                            hover_data=['activity',
                                        'event_index', 'actual_time']
                        )

                        # Enhance with cluster visualization
                        fig_clustered = cluster_detector.visualize(
                            df_selected, fig_clustered)

                        # Update layout for better visibility
                        fig_clustered.update_traces(
                            marker=dict(size=5, opacity=0.7))
                        fig_clustered.update_layout(
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.01
                            ),
                            width=1000,
                            height=600
                        )

                        st.plotly_chart(fig_clustered, width='stretch')

                        # Detailed cluster information
                        with st.expander(" Detailed Cluster Information"):
                            st.json(summary)

                            # Show cluster details
                            if summary.get('clusters'):
                                st.write("**Cluster Details:**")
                                for cluster_id, cluster_info in summary['clusters'].items():
                                    cluster_type = "Trend" if cluster_info.get(
                                        'is_trend') else "Regular"
                                    st.write(
                                        f"- **Cluster {cluster_id}** ({cluster_type}): {cluster_info['size']} points")

            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")
                st.exception(e)

    # Gap Detection Section
    if 'df' in st.session_state and 'current_plot_config' in st.session_state:
        st.divider()
        st.subheader("Gap Detection")

        plot_config = st.session_state['current_plot_config']
        x_col = plot_config['x_col']
        
        # Determine unit based on X-axis column
        # Note: actual_time is shown in seconds for user convenience, but converted to nanoseconds internally
        unit_info = {
            'actual_time': ('Sekunden', '1 Minute = 60 Sekunden', 1.0, True),  # True = convert to nanoseconds
            'relative_time': ('Sekunden', '1 Minute = 60 Sekunden', 1.0, False),
            'relative_ratio': ('Ratio [0-1]', '0.1 = 10% der Trace-Dauer', 0.01, False),
            'logical_time': ('Event-Index', '10 = 10 Events', 1.0, False),
            'logical_relative': ('Event-Index in Trace', '5 = 5 Events im Trace', 1.0, False)
        }
        
        unit_name, unit_example, unit_scale, needs_conversion = unit_info.get(
            x_col, ('Einheiten', 'Basierend auf X-Achse', 1.0, False))

        col1, col2 = st.columns([1, 1])

        with col1:
            min_gap_duration = st.number_input(
                f"Min Gap Duration in {unit_name} (optional)",
                min_value=0.0,
                value=None,
                step=unit_scale,
                help=f"Minimum gap duration to detect in {unit_name}. {unit_example}. Leave empty for automatic detection based on data distribution."
            )
            if min_gap_duration is not None and min_gap_duration == 0.0:
                min_gap_duration = None

        with col2:
            group_by_y = st.checkbox(
                "Group by Y-Axis",
                value=False,
                help="If enabled, detects gaps separately for each Y-axis value (e.g., per activity). If disabled, detects global gaps."
            )

        # Gap detection button
        if st.button("Detect Gaps", type="primary"):
            try:
                plot_config = st.session_state['current_plot_config']
                df_selected = plot_config['df_selected']

                # Create view configuration for gap detection
                # Determine view type based on column types
                preprocessor = DataPreprocessor()
                view_type = preprocessor._determine_view_type(df_selected, plot_config['x_col'], plot_config['y_col'])
                
                view_config = {
                    'x': plot_config['x_col'],
                    'y': plot_config['y_col'],
                    'view': view_type
                }

                # Create gap detector
                with st.spinner("Detecting gaps..."):
                    gap_detector = GapPattern(
                        view_config=view_config,
                        min_gap_duration=min_gap_duration,
                        group_by_y=group_by_y
                    )

                    # Detect gaps
                    gap_detector.detect(df_selected)

                    if gap_detector.detected is None:
                        # Clear gap detector if no gaps found
                        if 'gap_detector' in st.session_state:
                            del st.session_state['gap_detector']
                        st.warning(
                            "No gaps found with current parameters. Try adjusting the minimum gap duration or grouping settings.")
                    else:
                        # Store gap detection results
                        st.session_state['gap_detector'] = gap_detector
                        # Store parameters for display
                        st.session_state['gap_group_by_y'] = group_by_y
                        # Trigger chart refresh to show gaps
                        st.rerun()

            except Exception as e:
                st.error(f"Error during gap detection: {str(e)}")
                st.exception(e)

        # Display gap detection results (persistent, outside button block)
        if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            gap_detector = st.session_state['gap_detector']
            group_by_y = st.session_state.get('gap_group_by_y', False)
            plot_config = st.session_state['current_plot_config']
            x_col = plot_config['x_col']
            
            # Get gap summary
            summary = gap_detector.get_gap_summary()
            
            # Helper function to format timestamp (for start/end)
            def format_timestamp(value, x_col):
                """Format timestamp value based on X-axis column type."""
                if x_col == 'actual_time':
                    # Convert nanoseconds to datetime
                    import pandas as pd
                    timestamp_ns = int(value)
                    dt = pd.Timestamp(timestamp_ns)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    return f"{value:.2f}"
            
            # Helper function to format duration based on X-axis type
            def format_duration(value, x_col):
                """Format duration value based on X-axis column type."""
                if x_col == 'actual_time':
                    # Convert nanoseconds to readable units
                    seconds = value / 1_000_000_000
                    if seconds >= 86400:  # >= 1 day
                        days = seconds / 86400
                        return f"{days:.1f} Tage"
                    elif seconds >= 3600:  # >= 1 hour
                        hours = seconds / 3600
                        return f"{hours:.1f} Stunden"
                    elif seconds >= 60:  # >= 1 minute
                        minutes = seconds / 60
                        return f"{minutes:.1f} Minuten"
                    else:
                        return f"{seconds:.1f} Sekunden"
                elif x_col in ['relative_time']:
                    # Also in seconds (but relative)
                    seconds = value
                    if seconds >= 86400:
                        days = seconds / 86400
                        return f"{days:.1f} Tage"
                    elif seconds >= 3600:
                        hours = seconds / 3600
                        return f"{hours:.1f} Stunden"
                    elif seconds >= 60:
                        minutes = seconds / 60
                        return f"{minutes:.1f} Minuten"
                    else:
                        return f"{seconds:.1f} Sekunden"
                elif x_col == 'relative_ratio':
                    return f"{value:.3f} ({value*100:.1f}%)"
                elif x_col in ['logical_time', 'logical_relative']:
                    return f"{value:.0f} Events"
                else:
                    return f"{value:.2f}"

            # Display results
            st.success(
                f"Gap detection completed! Found {summary.get('total_gaps', 0)} gaps")

            # Show gap statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Gaps", summary.get(
                    'total_gaps', 0))
            with col2:
                total_duration = summary.get('total_gap_duration', 0)
                formatted_total = format_duration(total_duration, x_col)
                st.metric("Total Gap Duration", formatted_total)
            with col3:
                avg_duration = summary.get('average_gap_duration', 0)
                formatted_avg = format_duration(avg_duration, x_col)
                st.metric("Avg Gap Duration", formatted_avg)
            with col4:
                min_threshold = summary.get('min_gap_threshold', 0)
                formatted_threshold = format_duration(min_threshold, x_col)
                st.metric("Min Threshold", formatted_threshold)

            # Show gap details
            with st.expander("📊 Detailed Gap Information"):
                st.json(summary)

                # Show individual gaps (limited to first 50 for readability)
                if summary.get('gaps'):
                    st.write("**Detected Gaps:**")
                    gaps_to_show = summary['gaps'][:50]  # Show first 50
                    for i, gap in enumerate(gaps_to_show, 1):
                        gap_duration = format_duration(gap['duration'], x_col)
                        gap_start = format_timestamp(gap['start'], x_col)
                        gap_end = format_timestamp(gap['end'], x_col)
                        
                        if group_by_y:
                            st.write(
                                f"- **Gap {i}** (Y='{gap['y_value']}'): "
                                f"von {gap_start} bis {gap_end} "
                                f"(Dauer: {gap_duration})")
                        else:
                            st.write(
                                f"- **Gap {i}**: "
                                f"von {gap_start} bis {gap_end} "
                                f"(Dauer: {gap_duration})")
                    
                    if len(summary['gaps']) > 50:
                        st.info(f"⚠️ Nur die ersten 50 von {len(summary['gaps'])} Gaps werden angezeigt. "
                               f"Verwenden Sie die JSON-Ansicht für alle Details.")

    if st.button("Describe Chart with Ollama"):
        OllamaEvaluator_instance = OllamaEvaluator()
        df = st.session_state.get('df', None)
        summary = st.session_state.get('summary', None)
        if df is not None and summary is not None:
            summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
            description = OllamaEvaluator_instance.describe_chart(
                summary_text, df)
        else:
            description = "No data to describe. Please load a log first."
        st.subheader("Ollama Chart Description")
        st.write(description)


if __name__ == '__main__':
    main()
