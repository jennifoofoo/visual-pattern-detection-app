
import streamlit as st
import plotly.express as px
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import requests
from core.evaluation.ollama import OllamaEvaluator


def summarize_event_log(df):
    summary = {}
    summary['Number of cases'] = df['case_id'].nunique()
    summary['Number of events'] = len(df)
    summary['Most frequent activity'] = df['activity'].value_counts().idxmax()
    # Most common start and end activities
    start_activities = df.groupby('case_id').first()['activity'].value_counts()
    end_activities = df.groupby('case_id').last()['activity'].value_counts()
    summary['Most common start activity'] = start_activities.idxmax()
    summary['Most common end activity'] = end_activities.idxmax()
    # Average case duration
    durations = df.groupby('case_id').apply(lambda x: (
        x['timestamp'].max() - x['timestamp'].min()).total_seconds())
    summary['Average case duration (s)'] = durations.mean()
    return summary


def load_xes_log(xes_path):
    log = xes_importer.apply(xes_path)
    events = []
    for trace in log:
        case_id = trace.attributes.get('concept:name', None)
        if len(trace) == 0:
            continue
        start_time = trace[0].get('time:timestamp', None)
        end_time = trace[-1].get('time:timestamp', None)
        total_events = len(trace)
        total_logical_time = None
        if start_time and end_time:
            total_logical_time = (end_time - start_time).total_seconds()
        for idx, event in enumerate(trace):
            timestamp = event.get('time:timestamp', None)
            logical_time = None
            logical_relative = None
            relative_ratio = None
            if start_time and timestamp:
                logical_time = (timestamp - start_time).total_seconds()
                if total_logical_time and total_logical_time > 0:
                    logical_relative = logical_time / total_logical_time
            if total_events > 1:
                relative_ratio = idx / (total_events - 1)
            events.append({
                'case_id': case_id,
                'activity': event.get('concept:name', None),
                'timestamp': timestamp,
                'logical_time': logical_time,
                'event_index': idx,
                'relative_ratio': relative_ratio,
                'logical_relative': logical_relative
            })
    return pd.DataFrame(events)


def plot_dotted_chart(df):
    fig = px.scatter(
        df,
        x='x',
        y='activity',
        color='case_id',
        title='Dotted Chart',
        labels={
            'x': df.attrs.get('x_label', 'Time'),
            'activity': 'Activity',
            'case_id': 'Case ID'
        }
    )
    return fig


def main():
    st.title('Event Log Dotted Chart')
    xes_path = st.text_input('Enter XES log file path:',
                             'test_data/Hospital_log.xes', key='xes_path_input')
    x_axis_options = ['Relative time', 'Logical time',
                      'Relative ratio', 'Logical relative']
    y_axis_options = ['Case ID', 'Activity',
                      'Event Index', 'Resource', 'Variant']
    x_axis = st.selectbox('Select x-axis:', x_axis_options)
    y_axis = st.selectbox('Select y-axis:', y_axis_options)

    if st.button('Load and Plot'):
        df = load_xes_log(xes_path)
        st.session_state['summary'] = summarize_event_log(df)
        st.write(f"Loaded {len(df)} events.")
        st.subheader('Event Log Summary')
        for k, v in st.session_state['summary'].items():
            st.write(f"**{k}:** {v}")

        # Precompute all time representations
        time_dfs = {}
        df_rel = df.copy()
        df_rel['x'] = df_rel['timestamp']
        df_rel.attrs['x_label'] = 'Timestamp'
        time_dfs['Relative time'] = df_rel

        df_log = df.copy()
        df_log['x'] = df_log['logical_time']
        df_log.attrs['x_label'] = 'Logical Time (s)'
        time_dfs['Logical time'] = df_log

        df_ratio = df.copy()
        df_ratio['x'] = df_ratio['relative_ratio']
        df_ratio.attrs['x_label'] = 'Relative Ratio'
        time_dfs['Relative ratio'] = df_ratio

        df_logrel = df.copy()
        df_logrel['x'] = df_logrel['logical_relative']
        df_logrel.attrs['x_label'] = 'Logical Relative'
        time_dfs['Logical relative'] = df_logrel

        st.session_state['time_dfs'] = time_dfs

    # Show chart for selected x and y axis if data is loaded
    if 'time_dfs' in st.session_state:
        df_selected = st.session_state['time_dfs'][x_axis]
        y_map = {
            'Case ID': 'case_id',
            'Activity': 'activity',
            'Event Index': 'event_index',
            'Resource': 'resource',
            'Variant': 'variant'
        }
        y_col = y_map[y_axis]
        # If variant is selected, compute variant column if not present
        if y_col == 'variant' and 'variant' not in df_selected.columns:
            n = 10  # Number of most common variants to show
            case_variants = df_selected.groupby('case_id')['activity'].apply(lambda x: '-'.join(x))
            variant_counts = case_variants.value_counts()
            top_variants = variant_counts.head(n).index.tolist()
            # Assign variant column
            df_selected['variant'] = df_selected['case_id'].map(case_variants)
            # Filter to only top n variants
            df_selected = df_selected[df_selected['variant'].isin(top_variants)]
        fig = px.scatter(
            df_selected,
            x='x',
            y=y_col,
            color='case_id',
            title=f"Dotted Chart ({y_axis} vs {df_selected.attrs.get('x_label', 'Time')})",
            labels={'x': df_selected.attrs.get(
                'x_label', 'Time'), y_col: y_axis, 'case_id': 'Case ID'}
        )
        st.plotly_chart(fig)

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
