import streamlit as st
from core.data_processing import load_xes_log, DataPreprocessor

from  core.data_processing.loader import compute_variants

from core.evaluation.summary_generator import summarize_event_log
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP

from core.visualization.visualizer import plot_dotted_chart as plot_chart

from core.detection import OutlierDetectionPattern, TemporalClusterPattern
from core.detection.gap_pattern import GapPattern
from core.evaluation.ollama import OllamaEvaluator


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

def load_data_button(xes_path):
    try:
        with st.spinner(f"Loading {xes_path}..."):
            # Use cached loading
            df = cached_load_xes_log(xes_path)

        if df.empty:
            st.warning(
                "The log file was loaded but contains no events.")
            return

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
    if y_axis == 'Variant':
        compute_variants(df_base=df_base)

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

        # Add gap visualization if gaps were detected
        if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            fig = st.session_state['gap_detector'].visualize(df_selected, fig)

        st.plotly_chart(fig, use_container_width=True)

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
    st.session_state['fig'] = fig
    st.session_state['view_config'] = {
        'x_axis': x_col,
        'y_axis': y_col
    }
    st.session_state['chart_plotted'] = True

    st.success("Chart created successfully!")


# region Pattern Detection
def handle_temporal_cluster_detection(x_col, y_col, x_axis_label, y_axis_label, df_selected):
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

def handle_gap_detection():
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

def handle_pattern_detection():
    # Get current plot configuration from session state
    plot_config = st.session_state.get('current_plot_config', {})
    x_col = plot_config.get('x_col')
    y_col = plot_config.get('y_col')
    x_axis_label = plot_config.get('x_axis_label')
    y_axis_label = plot_config.get('y_axis_label')
    df_selected = plot_config.get('df_selected')

    handle_temporal_cluster_detection(x_col, y_col, x_axis_label, y_axis_label, df_selected)

    # Gap Detection Section
    if 'df' in st.session_state and 'current_plot_config' in st.session_state:
        st.divider()
        st.subheader("Gap Detection")

        handle_gap_detection()
                        
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