import streamlit as st
import plotly.express as px
import pandas as pd
from core.data_processing import load_xes_log, DataPreprocessor
from core.detection import OutlierDetectionPattern, TemporalClusterPattern
from core.detection.gap_pattern import GapPattern
from core.evaluation.ollama import OllamaEvaluator
from core.visualization.visualizer import plot_dotted_chart
from core.evaluation.summary_generator import summarize_event_log
from core.utils.demo_sampling import sample_small_eventlog, print_sampling_stats

# ⚠️ DEMO MODE - Set to True for fast gap detection during presentations
# This samples the event log to ~100 cases for instant results
# Set to False for full production analysis
DEMO_MODE = True

# Configure page for better performance
st.set_page_config(
    page_title="Event Log Dotted Chart",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_load_xes_log(xes_path):
    """Cached version of load_xes_log for better performance."""
    return load_xes_log(xes_path)


@st.cache_data(ttl=3600)
def generate_summary(df):
    """Cached summary generation."""
    return summarize_event_log(df)


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

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'chart_plotted' not in st.session_state:
        st.session_state.chart_plotted = False

    # File input
    xes_path = st.text_input(
        'Enter XES log file path:',
        value='data/Hospital_log.xes',
        key='xes_path_input'
    )

    # Step 1: Load Data (Cached)
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button('Load Data', type="primary"):
            try:
                with st.spinner(f"Loading {xes_path}..."):
                    # Use cached loading
                    df = cached_load_xes_log(xes_path)

                if df.empty:
                    st.warning(
                        "The log file was loaded but contains no events.")
                    return

                # ⚠️ DEMO MODE: Sample event log for fast gap detection
                if DEMO_MODE and 'case_id' in df.columns:
                    df_original = df
                    df = sample_small_eventlog(
                        df,
                        max_cases=100,
                        max_events_per_case=30,
                        time_col='actual_time',
                        random_state=42
                    )
                    # Print sampling stats to console (for debugging)
                    print_sampling_stats(df_original, df)
                    # Show info in UI
                    st.info(f"🎬 DEMO MODE: Sampled to {len(df):,} events from {len(df_original):,} events "
                           f"({df['case_id'].nunique()} cases) for fast gap detection.")

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

    # Show data status
    with col2:
        if st.session_state.data_loaded:
            df_info = st.session_state.df
            summary = st.session_state.get('summary', {})

            # Show key metrics
            col2a, col2b, col2c, col2d = st.columns(4)
            with col2a:
                st.metric("Events", f"{len(df_info):,}")

            with col2b:
                st.metric("File", st.session_state.get(
                    'loaded_file', '').split('/')[-1])

            with st.expander("Event Log Summary", expanded=False):
                if summary:
                    for k, v in summary.items():
                        st.write(f"**{k}:** {v}")
                else:
                    st.info("Summary not available")

    # Only show configuration if data is loaded
    if not st.session_state.data_loaded:
        st.info("Please load your XES file first")
        return

    # Step 2: Chart Configuration
    st.divider()
    st.subheader("Chart Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox('Select x-axis:', list(X_AXIS_COLUMN_MAP.keys()))
    with col2:
        y_axis = st.selectbox('Select y-axis:', list(Y_AXIS_COLUMN_MAP.keys()))
    with col3:
        dots_config_label = st.selectbox(
            'Select Dot Color:', list(DOTS_COLOR_MAP.keys()))

    if st.button('Plot Chart', type="primary"):
        if 'df' in st.session_state:
            df_base = st.session_state['df']

            # Determine the columns to plot
            x_col = X_AXIS_COLUMN_MAP[x_axis]
            y_col = Y_AXIS_COLUMN_MAP[y_axis]
            dots_config_col = DOTS_COLOR_MAP[dots_config_label]

            # Performance optimization: work with view instead of copy when possible
            df_selected = df_base

            # region Variant Calculation Logic
            if y_axis == 'Variant':
                # Need a copy for variant calculation
                df_selected = df_base.copy()
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
            # endregion

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
                fig = plot_dotted_chart(
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

                # Add gap visualization if gaps were detected
                if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
                    fig = st.session_state['gap_detector'].visualize(df_selected, fig)

                st.plotly_chart(fig, use_container_width=True)
                
                # Store the figure for later use (e.g., after gap detection)
                st.session_state['fig'] = fig

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

            # Store the figure and view config for pattern detection
            # (fig is already stored above if gap_detector exists)
            if 'fig' not in st.session_state:
                st.session_state['fig'] = fig
            st.session_state['view_config'] = {
                'x': x_col,
                'y': y_col
            }
            st.session_state['chart_plotted'] = True
            
            # Clear old gap detector when new chart is plotted
            if 'gap_detector' in st.session_state:
                del st.session_state['gap_detector']

        st.success("Chart created successfully!")

    # Pattern Detection Section (only show if chart is plotted)
    if st.session_state.chart_plotted:
        st.divider()
        st.subheader("Pattern Detection")

        # Get current plot configuration from session state
        plot_config = st.session_state.get('current_plot_config', {})
        x_col = plot_config.get('x_col')
        y_col = plot_config.get('y_col')
        x_axis_label = plot_config.get('x_axis_label')
        y_axis_label = plot_config.get('y_axis_label')
        df_selected = plot_config.get('df_selected')

        # Temporal Cluster Detection
        if st.button('Detect Temporal Clusters', type="secondary"):
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
                        st.info(
                            f"No meaningful temporal patterns for {y_axis_label} × {x_axis_label}")
            else:
                st.warning("Please plot a chart first")

        # Display temporal cluster results if they exist
        if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
            detector = st.session_state.temporal_clusters

            with st.expander("Temporal Cluster Analysis", expanded=True):
                st.text(detector.get_summary())

            # Visualize clusters on the chart
            if st.session_state.get('fig'):
                st.subheader("📊 Cluster Visualization")
                with st.spinner("Adding cluster overlays to chart..."):
                    # Create a copy of the figure and add cluster visualization
                    import copy
                    enhanced_fig = copy.deepcopy(st.session_state['fig'])
                    enhanced_fig = detector.visualize(
                        df=df_selected, fig=enhanced_fig)
                    st.plotly_chart(enhanced_fig, use_container_width=True)

            st.success("Temporal cluster detection completed!")

        # Outlier Detection
        if st.button("Detect Outliers", type="primary"):
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

        # Display outlier results if they exist in session state
        if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
            outlier_pattern = st.session_state.outlier_pattern

            # Display metrics
            stats = outlier_pattern.statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Outliers", f"{stats['total_outliers']:,}")
            with col2:
                st.metric("Outlier %", f"{stats['outlier_percentage']:.1f}%")
            with col3:
                st.metric("Max Score", stats['max_outlier_score'])
            with col4:
                st.metric(
                    "Cases Affected", f"{stats['cases_with_outliers']}/{stats['total_cases']}")

            # Enhanced visualization
            enhanced_fig = outlier_pattern.visualize(st.session_state.fig)
            st.plotly_chart(enhanced_fig, use_container_width=True)

            # Detailed analysis (collapsible)
            with st.expander("Detailed Outlier Analysis", expanded=False):
                summary = outlier_pattern.get_outlier_summary()

                col1, col2 = st.columns(2)
                with col1:
                    st.write(
                        f"**Detection Methods:** {stats['detection_methods_used']}/6")
                    st.write(
                        f"**Available Columns:** {len(stats['available_features'])}")

                with col2:
                    if summary.get("outlier_details"):
                        st.write("**Outlier Types:**")
                        for outlier_type, details in summary["outlier_details"].items():
                            st.write(
                                f"- {outlier_type.replace('_', ' ').title()}: {details['count']} ({details['percentage']:.1f}%)")

            # AI Analysis of Outliers
            st.subheader("🤖 AI Outlier Analysis")
            col_ai1, col_ai2 = st.columns([2, 1])

            with col_ai1:
                st.info(
                    "Get AI-powered insights about your high-confidence outliers")

            with col_ai2:
                if st.button("🔍 Analyze Outliers with AI", type="secondary"):
                    with st.spinner("🤖 AI analyzing outliers... This may take a moment"):
                        try:
                            evaluator = OllamaEvaluator()
                            summary = outlier_pattern.get_outlier_summary()

                            # Get high-confidence outlier data
                            max_score = stats['max_outlier_score']
                            if max_score > 1:  # Only analyze if we have high-confidence outliers
                                # Get outlier indices with max score
                                max_score_indices = [
                                    idx for idx in outlier_pattern.outliers.get('combined', [])
                                    if outlier_pattern.outlier_scores.get(idx, 0) == max_score
                                ]

                                if max_score_indices:
                                    outlier_data = st.session_state.df.loc[max_score_indices]

                                    # Generate AI analysis and store in session state
                                    ai_analysis = evaluator.analyze_outliers(
                                        summary, outlier_data, st.session_state.df, outlier_pattern)
                                    st.session_state.ai_outlier_analysis = ai_analysis
                                    st.rerun()  # Refresh to show the analysis
                                else:
                                    st.session_state.ai_outlier_analysis = "No high-confidence outliers found for detailed analysis"
                                    st.rerun()
                            else:
                                st.session_state.ai_outlier_analysis = "Only low-confidence outliers detected. Consider running on data with more clear anomalies for better AI analysis."
                                st.rerun()

                        except Exception as e:
                            st.session_state.ai_outlier_analysis = f"Error during AI analysis: {str(e)}\n\n💡 Make sure Ollama is running locally (ollama serve) with a model installed"
                            st.rerun()

            # Display stored AI analysis if it exists
            if 'ai_outlier_analysis' in st.session_state and st.session_state.ai_outlier_analysis:
                with st.expander("🎯 AI Insights on High-Confidence Outliers", expanded=True):
                    st.markdown(st.session_state.ai_outlier_analysis)

                    # Add a clear button to remove the analysis
                    if st.button("🗑️ Clear AI Analysis", type="secondary"):
                        del st.session_state.ai_outlier_analysis
                        st.rerun()

            st.success("Outlier detection completed!")

    # Gap Detection Section
    if 'df' in st.session_state and 'current_plot_config' in st.session_state:
        st.divider()
        st.subheader("Gap Detection")

        plot_config = st.session_state['current_plot_config']
        x_col = plot_config['x_col']
        y_col = plot_config['y_col']
        df_selected = plot_config['df_selected']
        
        # Determine if Y is categorical
        y_is_categorical = df_selected[y_col].nunique() <= 60
        
        # Check if X-axis is time-based
        is_time_based = x_col in ['actual_time', 'relative_time', 'relative_ratio', 'logical_time', 'logical_relative'] or \
                       (x_col in df_selected.columns and pd.api.types.is_datetime64_any_dtype(df_selected[x_col]))
        
        # Show detection mode info
        st.info(
            "🔬 **Process-Aware Gap Detection**\n\n"
            "Detects **abnormal gaps** by learning normal transition durations from your process:\n"
            "- Analyzes consecutive events within each case (A → B transitions)\n"
            "- Learns normal duration per transition using statistical analysis (Q1, Q3, IQR, P95)\n"
            "- Marks gaps as abnormal when duration > threshold\n"
            "- Requires at least 5 samples per transition for reliable statistics"
        )
        
        # Optional: Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            min_samples = st.number_input(
                "Minimum samples per transition",
                min_value=3,
                max_value=20,
                value=5,
                step=1,
                help="Transitions with fewer samples will be skipped (not enough data for reliable statistics)"
            )
            
            st.caption(
                "💡 **How it works:** For each activity transition (e.g., 'Check-in → Lab Test'), "
                "the detector computes the distribution of gap durations across all cases. "
                "It then calculates a threshold as max(P95, Q3 + 1.5×IQR). "
                "Gaps exceeding this threshold are marked as abnormal."
            )
        
        # Store settings (for now we only use min_samples in future updates)
        # Currently GapPattern uses hardcoded MIN_SAMPLES_FOR_NORMALITY = 5

        # Gap detection button
        if st.button("Detect Gaps", type="primary"):
            try:
                # Create view configuration for gap detection
                view_config = {
                    'x': x_col,
                    'y': y_col
                }

                # Create gap detector
                with st.spinner("Analyzing process transitions and detecting abnormal gaps..."):
                    gap_detector = GapPattern(
                        view_config=view_config,
                        y_is_categorical=y_is_categorical
                    )
                    
                    # Apply min_samples setting if changed from default
                    if 'min_samples' in locals() and min_samples != 5:
                        gap_detector.MIN_SAMPLES_FOR_NORMALITY = min_samples

                    # Detect gaps
                    gap_detector.detect(df_selected)

                    if gap_detector.detected is None:
                        # Clear gap detector if no gaps found
                        if 'gap_detector' in st.session_state:
                            del st.session_state['gap_detector']
                        st.warning(
                            "No abnormal gaps detected. This could mean:\n"
                            "- All gaps are within normal thresholds for their transitions\n"
                            "- Not enough transitions have sufficient samples (≥5)\n"
                            "- The log doesn't contain 'case_id' or 'activity' columns"
                        )
                    else:
                        # Store gap detection results
                        st.session_state['gap_detector'] = gap_detector
                        # Recreate and store the figure with gap visualization
                        if 'current_plot_config' in st.session_state:
                            # Recreate the base chart
                            plot_config = st.session_state['current_plot_config']
                            df_selected = plot_config['df_selected']
                            x_col = plot_config['x_col']
                            y_col = plot_config['y_col']
                            dots_config_col = plot_config.get('dots_config_col')
                            x_axis_label = plot_config.get('x_axis_label', x_col)
                            y_axis_label = plot_config.get('y_axis_label', y_col)
                            dots_config_label = plot_config.get('dots_config_label', '')
                            total_points = plot_config.get('total_points', len(df_selected))
                            
                            # Recreate the base figure
                            hover_cols = ['activity', 'event_index', 'actual_time']
                            fig = plot_dotted_chart(
                                df=df_selected,
                                x=x_col,
                                y=y_col,
                                color=dots_config_col,
                                title=f"Dotted Chart: {y_axis_label} vs {x_axis_label} ({total_points:,} points)",
                                labels={x_col: x_axis_label, y_col: y_axis_label,
                                        dots_config_col: dots_config_label},
                                hover_data=hover_cols
                            )
                            
                            # Improve visual appearance
                            fig.update_traces(marker=dict(size=5, opacity=0.8))
                            
                            # Layout settings
                            fig.update_layout(
                                showlegend=(dots_config_col is not None and dots_config_col != 'case_id'),
                                hovermode='closest',
                                template='plotly_white',
                                yaxis=dict(autorange='reversed')
                            )
                            
                            # Add gap visualization
                            fig = gap_detector.visualize(df_selected, fig)
                            
                            # Store the updated figure
                            st.session_state['fig'] = fig
                        # Trigger chart refresh to show gaps
                        st.rerun()

            except Exception as e:
                st.error(f"Error during gap detection: {str(e)}")
                st.exception(e)

        # Display chart with gaps if available (after gap detection)
        if 'fig' in st.session_state and 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            st.subheader("📊 Chart with Detected Gaps")
            st.plotly_chart(st.session_state['fig'], use_container_width=True)
            st.divider()

        # Display gap detection results (persistent, outside button block)
        if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            gap_detector = st.session_state['gap_detector']
            plot_config = st.session_state['current_plot_config']
            x_col = plot_config['x_col']
            y_col = plot_config['y_col']
            df_selected = plot_config['df_selected']
            
            # Get Y categorical status
            y_is_categorical = df_selected[y_col].nunique() <= 60
            
            # Get gap summary
            summary = gap_detector.get_gap_summary()
            
            # Helper function to format timestamp (for start/end)
            def format_timestamp(value, x_col):
                """Format timestamp value based on X-axis column type."""
                if x_col == 'actual_time':
                    import pandas as pd
                    # Check if value is already a Timestamp object
                    if isinstance(value, pd.Timestamp):
                        dt = value
                    elif isinstance(value, (int, float)):
                        # Convert nanoseconds to datetime
                        timestamp_ns = int(value)
                        dt = pd.Timestamp(timestamp_ns)
                    else:
                        # Try to convert string or other types
                        dt = pd.Timestamp(value)
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
            total_abnormal = summary.get('total_abnormal_gaps', 0)
            total_transitions = summary.get('total_transitions', 0)
            transitions_with_anomalies = summary.get('transitions_with_anomalies', 0)
            
            st.success(
                f"Process-aware gap detection completed! Found {total_abnormal} abnormal gaps "
                f"across {transitions_with_anomalies} of {total_transitions} transitions")

            # Show gap statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Abnormal Gaps", total_abnormal)
            with col2:
                st.metric("Transitions Analyzed", total_transitions)
            with col3:
                total_magnitude = summary.get('total_magnitude', 0)
                # Format based on magnitude size (duration in seconds)
                if total_magnitude > 86400:  # > 1 day
                    st.metric("Total Duration", f"{total_magnitude/86400:.1f} days")
                elif total_magnitude > 3600:  # > 1 hour
                    st.metric("Total Duration", f"{total_magnitude/3600:.1f} hours")
                else:
                    st.metric("Total Duration", f"{total_magnitude:.1f}s")
            with col4:
                avg_magnitude = summary.get('average_magnitude', 0)
                # Format based on magnitude size
                if avg_magnitude > 86400:  # > 1 day
                    st.metric("Avg Duration", f"{avg_magnitude/86400:.1f} days")
                elif avg_magnitude > 3600:  # > 1 hour
                    st.metric("Avg Duration", f"{avg_magnitude/3600:.1f} hours")
                else:
                    st.metric("Avg Duration", f"{avg_magnitude:.1f}s")

            # Show gap details
            with st.expander("📊 Detailed Gap Information"):
                # Show transition statistics
                if summary.get('transition_stats'):
                    st.write("**Transition Statistics:**")
                    st.caption(f"Learned normality thresholds for {len(summary['transition_stats'])} transitions")
                    
                    # Show top 5 transitions with most samples
                    top_transitions = sorted(
                        summary['transition_stats'].items(),
                        key=lambda x: x[1]['count'],
                        reverse=True
                    )[:5]
                    
                    st.write("Top 5 most frequent transitions:")
                    for trans, stats in top_transitions:
                        st.write(f"- **{trans}**: {stats['count']} occurrences, "
                                f"threshold: {stats['threshold']:.1f}s "
                                f"(P95: {stats['p95']:.1f}s, median: {stats['median']:.1f}s)")
                
                st.write("---")
                
                # Show individual abnormal gaps (limited to first 50 for readability)
                if summary.get('gaps'):
                    st.write("**Abnormal Gaps (Top 50 by Severity):**")
                    st.caption("✅ Process-aware gaps that exceed transition-specific thresholds")
                    
                    # Sort by severity (most severe first)
                    gaps_sorted = sorted(
                        summary['gaps'],
                        key=lambda g: g.get('severity', 0),
                        reverse=True
                    )
                    gaps_to_show = gaps_sorted[:50]  # Show top 50
                    
                    for i, gap in enumerate(gaps_to_show, 1):
                        x_start = format_timestamp(gap.get('x_start', 0), x_col)
                        x_end = format_timestamp(gap.get('x_end', 0), x_col)
                        
                        # Get transition info
                        transition = gap.get('transition', 'Unknown')
                        case_id = gap.get('case_id', 'Unknown')
                        duration = gap.get('duration', 0)
                        threshold = gap.get('threshold', 0)
                        severity = gap.get('severity', 0)
                        
                        # Format duration
                        if duration >= 86400:  # >= 1 day
                            duration_str = f"{duration/86400:.1f} days"
                        elif duration >= 3600:  # >= 1 hour
                            duration_str = f"{duration/3600:.1f} hours"
                        elif duration >= 60:  # >= 1 minute
                            duration_str = f"{duration/60:.1f} minutes"
                        else:
                            duration_str = f"{duration:.1f}s"
                        
                        # Format threshold
                        if threshold >= 86400:
                            threshold_str = f"{threshold/86400:.1f} days"
                        elif threshold >= 3600:
                            threshold_str = f"{threshold/3600:.1f} hours"
                        elif threshold >= 60:
                            threshold_str = f"{threshold/60:.1f} minutes"
                        else:
                            threshold_str = f"{threshold:.1f}s"
                        
                        # Color code by severity
                        if severity >= 5:
                            severity_emoji = "🔴"
                        elif severity >= 3:
                            severity_emoji = "🟠"
                        elif severity >= 2:
                            severity_emoji = "🟡"
                        else:
                            severity_emoji = "🟢"
                        
                        st.write(
                            f"{severity_emoji} **Gap {i}** - Transition: **{transition}**  \n"
                            f"  Case: {case_id} | Time: {x_start} → {x_end}  \n"
                            f"  Duration: {duration_str} | Threshold: {threshold_str} | "
                            f"**Severity: {severity:.2f}x**"
                        )
                    
                    if len(summary['gaps']) > 50:
                        st.info(f"⚠️ Showing top 50 of {len(summary['gaps'])} abnormal gaps (sorted by severity).")

    # Ollama Description (moved to sidebar for better performance)
    with st.sidebar:
        st.subheader(" AI Description")
        if st.button("Describe Chart", disabled=not st.session_state.data_loaded):
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


if __name__ == '__main__':
    main()
