import streamlit as st
from core.data_processing import load_xes_log, DataPreprocessor


from core.evaluation.summary_generator import summarize_event_log
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP

from core.visualization.visualizer import plot_dotted_chart as plot_chart

from core.detection import OutlierDetectionPattern, TemporalClusterPattern
from core.detection.gap_pattern import GapPattern
from core.evaluation.ollama import OllamaEvaluator
from core.utils.demo_sampling import sample_small_eventlog
from config.extended_pattern_matrix import is_pattern_meaningful, get_pattern_info


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
    # Initialize layer visibility flags (all layers visible by default)
    if 'visible_gap' not in st.session_state:
        st.session_state.visible_gap = True
    if 'visible_outlier' not in st.session_state:
        st.session_state.visible_outlier = True
    if 'visible_temporal_cluster' not in st.session_state:
        st.session_state.visible_temporal_cluster = True

def load_data_button(xes_path, demo_mode=False):
    try:
        with st.spinner(f"Loading {xes_path}..."):
            # Use cached loading
            df = cached_load_xes_log(xes_path)

        if df.empty:
            st.warning(
                "The log file was loaded but contains no events.")
            return

        # Demo Mode: Sample event log for fast gap detection
        if demo_mode and 'case_id' in df.columns:
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
                f"🎬 **DEMO MODE Active:** Sampled to {len(df):,} events from {len(df_original):,} events "
                f"({df['case_id'].nunique()} cases) for fast gap detection. "
                f"Uncheck 'Demo Mode' to analyze full dataset."
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

        # Note: Visualization overlays will be added by display_chart()
        # Do not display chart here - it will be displayed persistently

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

    # Store the figure and view config for pattern detection (with 3D matrix support)
    st.session_state['fig'] = fig
    st.session_state['view_config'] = {
        'x': x_col,
        'y': y_col,
        'color': dots_config_col  # Include color for 3D matrix consistency
    }
    st.session_state['chart_plotted'] = True
    st.session_state['chart_needs_display'] = True

    st.success("Chart created successfully!")
    
    # Auto-detect all meaningful patterns after plotting
    auto_detect_patterns(x_col, y_col, dots_config_col, x_axis, y_axis, df_plot)

def auto_detect_patterns(x_col, y_col, color_col, x_axis_label, y_axis_label, df_selected):
    """Automatically detect all meaningful patterns after chart is plotted."""
    # Check which patterns are meaningful
    temporal_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'temporal_cluster_x')
    outlier_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'outlier')
    gap_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'gap')
    
    with st.spinner("Auto-detecting patterns..."):
        # Temporal Clusters
        if temporal_meaningful:
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
                    # Initialize visibility to True when pattern is detected
                    st.session_state.visible_temporal_cluster = True
            except Exception as e:
                st.warning(f"Temporal cluster detection skipped: {str(e)}")
        
        # Outlier Detection
        if outlier_meaningful:
            try:
                outlier_pattern = OutlierDetectionPattern(
                    df=st.session_state.df,
                    view_config=st.session_state.view_config
                )
                if outlier_pattern.detect():
                    st.session_state.outlier_pattern = outlier_pattern
                    st.session_state.outlier_detected = True
                    # Initialize visibility to True when pattern is detected
                    st.session_state.visible_outlier = True
            except Exception as e:
                st.warning(f"Outlier detection skipped: {str(e)}")
        
        # Gap Detection
        if gap_meaningful:
            try:
                y_is_categorical = df_selected[y_col].nunique() <= 60
                view_config = {'x': x_col, 'y': y_col}
                gap_detector = GapPattern(
                    view_config=view_config,
                    y_is_categorical=y_is_categorical
                )
                gap_detector.MIN_SAMPLES_FOR_NORMALITY = 5
                gap_detector.detect(df_selected)
                
                if gap_detector.detected is not None and len(gap_detector.detected) > 0:
                    st.session_state['gap_detector'] = gap_detector
                    # Initialize visibility to True when pattern is detected
                    st.session_state.visible_gap = True
            except Exception as e:
                st.warning(f"Gap detection skipped: {str(e)}")
    
    # Trigger chart redisplay with patterns
    st.session_state['chart_needs_display'] = True
    
    # Show success message with detected patterns count
    detected_count = sum([
        st.session_state.get('temporal_detected', False),
        st.session_state.get('outlier_detected', False),
        'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None
    ])
    if detected_count > 0:
        st.success(f"✅ {detected_count} pattern(s) detected and visualized! Open sidebar (←) to toggle layers.")


def display_chart():
    """Display the chart from session state (persistent across reruns)."""
    if not st.session_state.get('chart_plotted', False):
        return
        
    plot_config = st.session_state.get('current_plot_config', {})
    if not plot_config:
        return
    
    df_selected = plot_config['df_selected']
    x_col = plot_config['x_col']
    y_col = plot_config['y_col']
    dots_config_col = plot_config['dots_config_col']
    x_axis = plot_config['x_axis_label']
    y_axis = plot_config['y_axis_label']
    dots_config_label = plot_config['dots_config_label']
    total_points = plot_config['total_points']
    color_col = dots_config_col
    hover_cols = ['activity', 'event_index', 'actual_time']
    
    # Recreate the chart
    fig = plot_chart(
        df=df_selected,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"Dotted Chart: {y_axis} vs {x_axis} ({total_points:,} points)",
        labels={x_col: x_axis, y_col: y_axis, color_col: dots_config_label},
        hover_data=hover_cols
    )
    
    # Improve visual appearance
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    
    # Layout settings
    fig.update_layout(
        showlegend=(color_col is not None and color_col != 'case_id'),
        hovermode='closest',
        template='plotly_white',
        yaxis=dict(autorange='reversed')
    )
    
    # Add gap visualization if gaps were detected AND layer is visible
    if st.session_state.get('visible_gap', True):
        if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
            fig = st.session_state['gap_detector'].visualize(df_selected, fig)
    
    # Add outlier visualization if detected AND layer is visible
    if st.session_state.get('visible_outlier', True):
        if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
            fig = st.session_state.outlier_pattern.visualize(df_selected, fig)
    
    # Add temporal cluster visualization if detected AND layer is visible
    if st.session_state.get('visible_temporal_cluster', True):
        if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
            fig = st.session_state.temporal_clusters.visualize(df_selected, fig)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Update stored figure
    st.session_state['fig'] = fig


def sidebar_pattern_layer_controls():
    """Display pattern layer visibility controls in sidebar."""
    # Check if any pattern was detected
    any_detected = (
        st.session_state.get('temporal_detected', False) or 
        st.session_state.get('outlier_detected', False) or 
        ('gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None)
    )
    
    if not any_detected:
        return
    
    st.subheader("🎨 Pattern Layers")
    st.caption("Toggle pattern visualizations on the chart")
    
    # Initialize visibility flags in session state with separate keys
    # Temporal Clusters
    if st.session_state.get('temporal_detected', False):
        if 'visible_temporal_cluster' not in st.session_state:
            st.session_state.visible_temporal_cluster = True
        
        st.checkbox(
            "⏱️ Temporal Clusters",
            key='visible_temporal_cluster',
            help="Show/hide temporal cluster visualization"
        )
    
    # Outlier Detection
    if st.session_state.get('outlier_detected', False):
        if 'visible_outlier' not in st.session_state:
            st.session_state.visible_outlier = True
        
        st.checkbox(
            "🎯 Outlier Detection",
            key='visible_outlier',
            help="Show/hide outlier detection visualization"
        )
    
    # Gap Detection
    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        if 'visible_gap' not in st.session_state:
            st.session_state.visible_gap = True
        
        st.checkbox(
            "🔬 Gap Detection",
            key='visible_gap',
            help="Show/hide gap detection visualization"
        )
    


# region Pattern Detection
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
                st.session_state['chart_needs_display'] = True
                st.rerun()
            else:
                st.session_state.temporal_detected = False
                st.info(f"No meaningful temporal patterns for {y_axis_label} × {x_axis_label}")
    else:
        st.warning("Please plot a chart first")

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
                st.session_state['chart_needs_display'] = True
                st.rerun()
            else:
                st.session_state.outlier_detected = False
                st.info("No significant outliers detected!")

        except Exception as e:
            st.session_state.outlier_detected = False
            st.error(f"Error during outlier detection: {str(e)}")

def handle_gap_detection_logic(df_selected, x_col, y_col, min_samples=5):
    """Execute gap detection logic."""
    try:
        # Determine if Y is categorical
        y_is_categorical = df_selected[y_col].nunique() <= 60
        
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
            
            # Apply min_samples setting
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
                st.session_state['chart_needs_display'] = True
                st.rerun()

    except Exception as e:
        st.error(f"Error during gap detection: {str(e)}")
        st.exception(e)

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

def handle_pattern_detection():
    # Get current plot configuration from session state
    plot_config = st.session_state.get('current_plot_config', {})
    x_col = plot_config.get('x_col')
    y_col = plot_config.get('y_col')
    color_col = plot_config.get('dots_config_col', 'case_id')  # Default to case_id if not set
    x_axis_label = plot_config.get('x_axis_label')
    y_axis_label = plot_config.get('y_axis_label')
    df_selected = plot_config.get('df_selected')

    # Check which patterns are meaningful for this view (now includes color dimension)
    temporal_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'temporal_cluster_x')
    outlier_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'outlier')
    gap_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'gap')
    
    # Get pattern info for tooltips (now includes color dimension)
    temporal_info = get_pattern_info(x_col, y_col, color_col, 'temporal_cluster_x')
    outlier_info = get_pattern_info(x_col, y_col, color_col, 'outlier')
    gap_info = get_pattern_info(x_col, y_col, color_col, 'gap')
    
    # ========== PATTERN SUMMARY SECTION (patterns auto-detected) ==========
    st.subheader("📋 Pattern Summary")
    st.caption("Patterns are automatically detected after plotting. Toggle visibility in sidebar (←)")
    
    # Check if any pattern was detected
    any_detected = (
        st.session_state.get('temporal_detected', False) or 
        st.session_state.get('outlier_detected', False) or 
        ('gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None)
    )
    
    if not any_detected:
        st.info("💡 Patterns will appear here after chart is plotted.")
        return
    
    # Create tabs for each detected pattern
    tabs_list = []
    tabs_names = []
    
    if st.session_state.get('temporal_detected', False):
        tabs_names.append("⏱️ Temporal Clusters")
        tabs_list.append('temporal')
    if st.session_state.get('outlier_detected', False):
        tabs_names.append("🎯 Outlier Detection")
        tabs_list.append('outlier')
    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        tabs_names.append("🔬 Gap Detection")
        tabs_list.append('gap')
    
    # Create tabs dynamically
    tabs = st.tabs(tabs_names)
    
    for i, (tab, pattern_type) in enumerate(zip(tabs, tabs_list)):
        with tab:
            if pattern_type == 'temporal':
                display_temporal_cluster_tab()
            elif pattern_type == 'outlier':
                display_outlier_tab()
            elif pattern_type == 'gap':
                display_gap_tab()


def display_temporal_cluster_tab():
    """Display Temporal Cluster pattern details in tab."""
    if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
        detector = st.session_state.temporal_clusters
        summary = detector.get_summary()
        
        # Layer visibility is now controlled by sidebar
        layer_visible = st.session_state.get('visible_temporal_cluster', True)
        
        if not layer_visible:
            st.info("👁️‍🗨️ Layer hidden - toggle in sidebar to show on chart")
        
        st.caption("Finds time periods with unusually high or low event activity.")
        st.success(f"✅ {summary['count']} clusters detected")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Clusters", summary['count'])
        with col_m2:
            st.metric("Type", summary['pattern_type'].replace('_', ' ').title())
        
        with st.expander("📊 Details", expanded=False):
            st.text(summary['details']['summary_text'])


def display_outlier_tab():
    """Display Outlier Detection pattern details in tab."""
    if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
        outlier_pattern = st.session_state.outlier_pattern
        summary = outlier_pattern.get_summary()
        
        # Layer visibility is now controlled by sidebar
        layer_visible = st.session_state.get('visible_outlier', True)
        
        if not layer_visible:
            st.info("👁️‍🗨️ Layer hidden - toggle in sidebar to show on chart")
        
        st.caption("Identifies unusual events or cases based on temporal deviations.")
        st.success(f"✅ {summary['count']} outliers detected")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Outliers", summary['count'])
        with col_m2:
            stats = summary['details'].get('statistics', {})
            st.metric("Outlier %", f"{stats.get('outlier_percentage', 0):.1f}%")
        with col_m3:
            st.metric("Methods", f"{stats.get('detection_methods_used', 0)}/6")
        
        with st.expander("📊 Details", expanded=False):
            if summary['details'].get('outlier_details'):
                st.write("**Outlier Types:**")
                for outlier_type, details in summary['details']['outlier_details'].items():
                    st.write(f"- {outlier_type.replace('_', ' ').title()}: {details['count']} ({details['percentage']:.1f}%)")


def display_gap_tab():
    """Display Gap Detection pattern details in tab."""
    if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
        gap_detector = st.session_state['gap_detector']
        summary = gap_detector.get_summary()
        details = summary['details']
        
        # Settings popover in header
        header_col1, header_col2 = st.columns([0.85, 0.15])
        with header_col1:
            st.caption("Learns normal transition durations (A → B) and detects unusually long gaps.")
        with header_col2:
            # Settings popover for gap detection parameters
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
                        # Trigger re-detection
                        plot_config = st.session_state.get('current_plot_config', {})
                        if plot_config:
                            df_selected = plot_config['df_selected']
                            x_col = plot_config['x_col']
                            y_col = plot_config['y_col']
                            handle_gap_detection_logic(df_selected, x_col, y_col, min_samples)
                            st.rerun()
        
        # Layer visibility is now controlled by sidebar
        layer_visible = st.session_state.get('visible_gap', True)
        
        if not layer_visible:
            st.info("👁️‍🗨️ Layer hidden - toggle in sidebar to show on chart")
        
        st.success(f"✅ {summary['count']} abnormal gaps detected")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Gaps", summary['count'])
        with col_m2:
            st.metric("Transitions", details['total_transitions'])
        with col_m3:
            st.metric("Anomalies", details['transitions_with_anomalies'])
        with col_m4:
            total_duration = details['total_magnitude']
            if total_duration > 86400:
                duration_str = f"{total_duration/86400:.1f}d"
            elif total_duration > 3600:
                duration_str = f"{total_duration/3600:.1f}h"
            else:
                duration_str = f"{total_duration:.0f}s"
            st.metric("Duration", duration_str)
        
        with st.expander("📊 Details", expanded=False):
            st.write("**Top Transitions with Anomalies:**")
            trans_stats = details.get('transition_stats', {})
            for trans, stats in list(trans_stats.items())[:5]:
                st.write(f"- **{trans}**: {stats['count']} occurrences, threshold: {stats['threshold']/86400:.1f} days")
            
            st.write("\n**Top 10 Abnormal Gaps by Severity:**")
            abnormal_gaps = sorted(details['abnormal_gaps'], key=lambda x: x.get('severity', 0), reverse=True)[:10]
            for i, gap in enumerate(abnormal_gaps, 1):
                duration_days = gap['duration'] / 86400
                threshold_days = gap['threshold'] / 86400
                st.write(f"{i}. {gap['transition']} - Duration: {duration_days:.1f}d, Threshold: {threshold_days:.1f}d, Severity: {gap['severity']:.2f}x")



                        
def ollama_description_button():
            if st.session_state.get('temporal_detected', False) and 'temporal_clusters' in st.session_state:
                detector = st.session_state.temporal_clusters
                summary = detector.get_summary()
                
                with st.container(border=True):
                    st.markdown("### ⏱️ Temporal Clusters")
                    st.caption("Finds time periods with unusually high or low event activity.")
                    
                    # Layer visibility is now controlled by sidebar
                    layer_visible = st.session_state.get('visible_temporal_cluster', True)
                    
                    if not layer_visible:
                        st.info("👁️‍🗨️ Layer hidden - toggle in sidebar to show on chart")
                    
                    st.success(f"✅ {summary['count']} clusters detected")
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Clusters", summary['count'])
                    with col_m2:
                        st.metric("Type", summary['pattern_type'].replace('_', ' ').title())
                    
                    with st.expander("📊 Details", expanded=False):
                        st.text(summary['details']['summary_text'])
        
        # === OUTLIER DETECTION SUMMARY ===
        with sum_col2:
            if st.session_state.get('outlier_detected', False) and 'outlier_pattern' in st.session_state:
                outlier_pattern = st.session_state.outlier_pattern
                summary = outlier_pattern.get_summary()
                
                with st.container(border=True):
                    st.markdown("### 🎯 Outlier Detection")
                    st.caption("Identifies unusual events or cases based on temporal deviations.")
                    
                    # Layer visibility is now controlled by sidebar
                    layer_visible = st.session_state.get('visible_outlier', True)
                    
                    if not layer_visible:
                        st.info("👁️‍🗨️ Layer hidden - toggle in sidebar to show on chart")
                    
                    st.success(f"✅ {summary['count']} outliers detected")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Outliers", summary['count'])
                    with col_m2:
                        stats = summary['details'].get('statistics', {})
                        st.metric("Outlier %", f"{stats.get('outlier_percentage', 0):.1f}%")
                    with col_m3:
                        st.metric("Methods", f"{stats.get('detection_methods_used', 0)}/6")
                    
                    with st.expander("📊 Details", expanded=False):
                        if summary['details'].get('outlier_details'):
                            st.write("**Outlier Types:**")
                            for outlier_type, details in summary['details']['outlier_details'].items():
                                st.write(f"- {outlier_type.replace('_', ' ').title()}: {details['count']} ({details['percentage']:.1f}%)")
        
        # === GAP DETECTION SUMMARY ===
        with sum_col3:
            if 'gap_detector' in st.session_state and st.session_state['gap_detector'].detected is not None:
                gap_detector = st.session_state['gap_detector']
                summary = gap_detector.get_summary()
                details = summary['details']
                
                with st.container(border=True):
                    header_col1, header_col2 = st.columns([0.85, 0.15])
                    with header_col1:
                        st.markdown("### 🔬 Gap Detection")
                        st.caption("Learns normal transition durations (A → B) and detects unusually long gaps.")
                    with header_col2:
                        # Settings popover for gap detection parameters
                        with st.popover("⚙️"):
                            st.write("**Settings**")
                            current_min_samples = st.session_state.get('gap_min_samples', 5)
                            min_samples = st.number_input(
                                "Min samples per transition",
                                min_value=3,
                                max_value=20,
                                value=current_min_samples,
                                step=1,
                                key="gap_min_samples_input",
                                help="Transitions with fewer samples are skipped"
                            )
                            if min_samples != current_min_samples:
                                if st.button("Apply & Re-detect", use_container_width=True, type="primary"):
                                    st.session_state['gap_min_samples'] = min_samples
                                    # Trigger re-detection
                                    plot_config = st.session_state.get('current_plot_config', {})
                                    if plot_config:
                                        df_selected = plot_config['df_selected']
                                        x_col = plot_config['x_col']
                                        y_col = plot_config['y_col']
                                        handle_gap_detection_logic(df_selected, x_col, y_col, min_samples)
                                        st.rerun()
                    
                    # Layer visibility is now controlled by sidebar
                    layer_visible = st.session_state.get('visible_gap', True)
                    
                    if not layer_visible:
                        st.info("👁️‍🗨️ Layer hidden - toggle in sidebar to show on chart")
                    
                    st.success(f"✅ {summary['count']} abnormal gaps detected")
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Gaps", summary['count'])
                    with col_m2:
                        st.metric("Transitions", details['total_transitions'])
                    with col_m3:
                        st.metric("Anomalies", details['transitions_with_anomalies'])
                    with col_m4:
                        total_duration = details['total_magnitude']
                        if total_duration > 86400:
                            duration_str = f"{total_duration/86400:.1f}d"
                        elif total_duration > 3600:
                            duration_str = f"{total_duration/3600:.1f}h"
                        else:
                            duration_str = f"{total_duration:.0f}s"
                        st.metric("Duration", duration_str)
                    
                    with st.expander("📊 Details", expanded=False):
                        st.write("**Top Transitions with Anomalies:**")
                        trans_stats = details.get('transition_stats', {})
                        for trans, stats in list(trans_stats.items())[:5]:
                            st.write(f"- **{trans}**: {stats['count']} occurrences, threshold: {stats['threshold']/86400:.1f} days")
                        
                        st.write("\n**Top 10 Abnormal Gaps by Severity:**")
                        abnormal_gaps = sorted(details['abnormal_gaps'], key=lambda x: x.get('severity', 0), reverse=True)[:10]
                        for i, gap in enumerate(abnormal_gaps, 1):
                            duration_days = gap['duration'] / 86400
                            threshold_days = gap['threshold'] / 86400
                            st.write(f"{i}. {gap['transition']} - Duration: {duration_days:.1f}d, Threshold: {threshold_days:.1f}d, Severity: {gap['severity']:.2f}x")
                        
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