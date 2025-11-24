import streamlit as st
from core.data_processing import load_xes_log, DataPreprocessor

from  core.data_processing.loader import compute_variants

from core.evaluation.summary_generator import summarize_event_log
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP

from core.visualization.visualizer import plot_dotted_chart as plot_chart

from core.detection import OutlierDetectionPattern, TemporalClusterPattern
from core.detection.gap_pattern import GapPattern
from core.evaluation.ollama import OllamaEvaluator
from core.utils.demo_sampling import sample_small_eventlog

# ⚠️ DEMO MODE - Set to True for fast gap detection during presentations
DEMO_MODE = True


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
            # Show info in UI
            st.info(
                f"🎬 DEMO MODE: Sampled to {len(df):,} events from {len(df_original):,} events "
                f"({df['case_id'].nunique()} cases) for fast gap detection."
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
    df_selected = plot_config['df_selected']
    x_col = plot_config['x_col']
    y_col = plot_config['y_col']
    
    # Determine if Y is categorical
    y_is_categorical = df_selected[y_col].nunique() <= 60
    
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
                    # Trigger chart refresh to show gaps
                    st.rerun()

        except Exception as e:
            st.error(f"Error during gap detection: {str(e)}")
            st.exception(e)

    # Display gap detection results (persistent, outside button block)
    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        gap_detector = st.session_state['gap_detector']
        plot_config = st.session_state['current_plot_config']
        x_col = plot_config['x_col']
        
        # Get gap summary
        summary = gap_detector.get_gap_summary()
        
        # Helper function to format timestamp
        def format_timestamp(value, x_col):
            """Format timestamp value."""
            if x_col == 'actual_time':
                import pandas as pd
                if isinstance(value, pd.Timestamp):
                    return value.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    return str(value)
            else:
                return f"{value:.2f}"
        
        # Display results
        total_abnormal = summary.get('total_abnormal_gaps', 0)
        total_transitions = summary.get('total_transitions', 0)
        transitions_with_anomalies = summary.get('transitions_with_anomalies', 0)
        
        st.success(
            f"Process-aware gap detection completed! Found {total_abnormal} abnormal gaps "
            f"across {transitions_with_anomalies} of {total_transitions} transitions"
        )

        # Show gap statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Abnormal Gaps", total_abnormal)
        with col2:
            st.metric("Transitions Analyzed", total_transitions)
        with col3:
            total_magnitude = summary.get('total_magnitude', 0)
            if total_magnitude > 86400:  # > 1 day
                st.metric("Total Duration", f"{total_magnitude/86400:.1f} days")
            elif total_magnitude > 3600:  # > 1 hour
                st.metric("Total Duration", f"{total_magnitude/3600:.1f} hours")
            else:
                st.metric("Total Duration", f"{total_magnitude:.1f}s")
        with col4:
            avg_magnitude = summary.get('average_magnitude', 0)
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
                    st.write(
                        f"- **{trans}**: {stats['count']} occurrences, "
                        f"threshold: {stats['threshold']:.1f}s "
                        f"(P95: {stats['p95']:.1f}s, median: {stats['median']:.1f}s)"
                    )
            
            st.write("---")
            
            # Show individual abnormal gaps
            if summary.get('gaps'):
                st.write("**Abnormal Gaps (Top 50 by Severity):**")
                st.caption("✅ Process-aware gaps that exceed transition-specific thresholds")
                
                # Sort by severity
                gaps_sorted = sorted(
                    summary['gaps'],
                    key=lambda g: g.get('severity', 0),
                    reverse=True
                )
                gaps_to_show = gaps_sorted[:50]
                
                for i, gap in enumerate(gaps_to_show, 1):
                    x_start = format_timestamp(gap.get('x_start', 0), x_col)
                    x_end = format_timestamp(gap.get('x_end', 0), x_col)
                    
                    transition = gap.get('transition', 'Unknown')
                    case_id = gap.get('case_id', 'Unknown')
                    duration = gap.get('duration', 0)
                    threshold = gap.get('threshold', 0)
                    severity = gap.get('severity', 0)
                    
                    # Format duration
                    if duration >= 86400:
                        duration_str = f"{duration/86400:.1f} days"
                    elif duration >= 3600:
                        duration_str = f"{duration/3600:.1f} hours"
                    elif duration >= 60:
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