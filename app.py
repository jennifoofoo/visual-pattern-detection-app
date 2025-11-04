import streamlit as st
import plotly.express as px
from core.evaluation.ollama import OllamaEvaluator
from core.visualization.visualizer import plot_dotted_chart
from core.evaluation.summary_generator import summarize_event_log
from core.data_loader import load_xes_log


# Maps user selection to the column name generated in load_xes_log
X_AXIS_COLUMN_MAP = {
    'Actual time': 'actual_time',               # 0.
    'Relative time': 'relative_time',           # 1. Relative Time (seconds)
    'Relative ratio': 'relative_ratio',         # 2. Relative Ratio (time-based [0, 1])
    'Logical time': 'logical_time',             # 3. Global Logical Time (index)
    'Logical relative': 'logical_relative'      # 4. Logical Relative (global index)
}

# Maps user selection to the column name available in the DataFrame
Y_AXIS_COLUMN_MAP = {
    'Case ID': 'case_id',
    'Activity': 'activity',
    'Event Index': 'event_index', # Used 'event_index_in_trace' in latest load_xes_log
    'Resource': 'resource',                 # Assuming 'resource' is in the log/DataFrame
    'Variant': 'variant'                    # Will be calculated if selected
}

def main():
    st.title('Event Log Dotted Chart')

    xes_path = st.text_input('Enter XES log file path:',
                             'data/Hospital_log.xes', key='xes_path_input')
    
    x_axis_options = list(X_AXIS_COLUMN_MAP.keys())
    y_axis_options = list(Y_AXIS_COLUMN_MAP.keys())
    
    x_axis = st.selectbox('Select x-axis:', x_axis_options)
    y_axis = st.selectbox('Select y-axis:', y_axis_options)

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
            
            # Create a copy for modification if needed (e.g., variant calculation)
            df_selected = df_base.copy() 
            
            # region Varian Calculation Logic 
            # --- Handle Variant Calculation (Your existing logic) ---
            if y_axis == 'Variant':
                if 'variant' not in df_selected.columns:
                    n = 10  # Number of most common variants to show
                    
                    # Combine activities into a string per case
                    case_variants = df_selected.groupby('case_id')['activity'].apply(lambda x: '-'.join(x.astype(str)))
                    variant_counts = case_variants.value_counts()
                    top_variants = variant_counts.head(n).index.tolist()
                    
                    # Map the variant string back to the event DataFrame
                    df_selected['variant'] = df_selected['case_id'].map(case_variants)
                    
                    # Filter to only top n variants for a cleaner chart
                    df_selected = df_selected[df_selected['variant'].isin(top_variants)].copy()
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
                color='case_id',                  # Color by trace
                title=f"Dotted Chart: {y_axis} vs {x_axis}",
                labels={x_col: x_axis, y_col: y_axis, 'case_id': 'Case ID'},
                hover_data=['activity', 'event_index', 'actual_time']
            )
            
            # Optional: Improve visual appearance for better density/clarity
            fig.update_traces(marker=dict(size=5, opacity=0.8))
            fig.update_layout(showlegend=False) # Usually too many case IDs to show legend
            
            st.plotly_chart(fig, use_container_width=True)

    # region legacy

    # # #  if button is pressed then do this
    # # if st.button('Load and Plot'):
    # #     df = load_xes_log(xes_path)
    # #     st.session_state['summary'] = summarize_event_log(df)
    # #     st.write(f"Loaded {len(df)} events.")
    # #     st.subheader('Event Log Summary')
    # #     for k, v in st.session_state['summary'].items():
    # #         st.write(f"**{k}:** {v}")

    #     # Precompute all time representations
    #     # absolute time
    #     time_dfs = {}
    #     df_rel = df.copy()
    #     df_rel['x'] = df_rel['actual_time']
    #     df_rel.attrs['x_label'] = 'actual_time'
    #     time_dfs['Relative time'] = df_rel

    #     df_log = df.copy()
    #     df_log['x'] = df_log['logical_time']
    #     df_log.attrs['x_label'] = 'Logical Time (s)'
    #     time_dfs['Logical time'] = df_log

    #     df_ratio = df.copy()
    #     df_ratio['x'] = df_ratio['relative_ratio']
    #     df_ratio.attrs['x_label'] = 'Relative Ratio'
    #     time_dfs['Relative ratio'] = df_ratio

    #     df_logrel = df.copy()
    #     df_logrel['x'] = df_logrel['logical_relative']
    #     df_logrel.attrs['x_label'] = 'Logical Relative'
    #     time_dfs['Logical relative'] = df_logrel

    #     st.session_state['time_dfs'] = time_dfs

    # # Show chart for selected x and y axis if data is loaded
    # if 'time_dfs' in st.session_state:
    #     df_selected = st.session_state['time_dfs'][x_axis]
    #     y_map = {
    #         'Case ID': 'case_id',
    #         'Activity': 'activity',
    #         'Event Index': 'event_index',
    #         'Resource': 'resource',
    #         'Variant': 'variant'
    #     }
    #     y_col = y_map[y_axis]
    #     # If variant is selected, compute variant column if not present
    #     if y_col == 'variant' and 'variant' not in df_selected.columns:
    #         n = 10  # Number of most common variants to show
    #         case_variants = df_selected.groupby('case_id')['activity'].apply(lambda x: '-'.join(x))
    #         variant_counts = case_variants.value_counts()
    #         top_variants = variant_counts.head(n).index.tolist()
    #         # Assign variant column
    #         df_selected['variant'] = df_selected['case_id'].map(case_variants)
    #         # Filter to only top n variants
    #         df_selected = df_selected[df_selected['variant'].isin(top_variants)]
    #     fig = px.scatter(
    #         df_selected,
    #         x='x',
    #         y=y_col,
    #         color='case_id',
    #         title=f"Dotted Chart ({y_axis} vs {df_selected.attrs.get('x_label', 'Time')})",
    #         labels={'x': df_selected.attrs.get(
    #             'x_label', 'Time'), y_col: y_axis, 'case_id': 'Case ID'}
    #     )
    #     st.plotly_chart(fig)
    # endregion


    # ignore for now
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
