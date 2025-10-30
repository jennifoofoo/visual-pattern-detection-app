
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
    st.markdown("<h1 style='margin-top:0;'>Event Log Dotted Chart</h1>", unsafe_allow_html=True)
    left_col, right_col = st.columns([2, 2])

    with right_col:
        xes_path = st.text_input('Enter XES log file path:', 'test_data/Hospital_log.xes', key='xes_path_input')
        x_axis_options = ['Relative time', 'Logical time', 'Relative ratio', 'Logical relative']
        y_axis_options = ['Case ID', 'Activity', 'Event Index', 'Resource', 'Variant']
        x_axis = st.selectbox('Select x-axis:', x_axis_options)
        y_axis = st.selectbox('Select y-axis:', y_axis_options)

    if st.button('Load and Plot'):
        df = load_xes_log(xes_path)
        st.session_state['summary'] = summarize_event_log(df)
        st.session_state['df'] = df
        st.write(f"Loaded {len(df)} events.")
    # Show chart for selected x and y axis if data is loaded
    if 'df' in st.session_state:
        df_base = st.session_state['df'].copy()
        # Set x axis
        x_map = {
            'Relative time': ('timestamp', 'Timestamp'),
            'Logical time': ('logical_time', 'Logical Time (s)'),
            'Relative ratio': ('relative_ratio', 'Relative Ratio'),
            'Logical relative': ('logical_relative', 'Logical Relative')
        }
        x_col, x_label = x_map[x_axis]
        df_base['x'] = df_base[x_col]
        df_base.attrs['x_label'] = x_label
        # Set y axis
        y_map = {
            'Case ID': 'case_id',
            'Activity': 'activity',
            'Event Index': 'event_index',
            'Resource': 'resource',
            'Variant': 'variant'
        }
        y_col = y_map[y_axis]
        # If variant is selected, show only n most common variants
        if y_col == 'variant':
            n = 10  # Number of most common variants to show
            case_variants = df_base.groupby('case_id')['activity'].apply(lambda x: '-'.join(x))
            variant_counts = case_variants.value_counts()
            top_variants = variant_counts.head(n).index.tolist()
            df_base['variant'] = df_base['case_id'].map(case_variants)
            df_base = df_base[df_base['variant'].isin(top_variants)]
        fig = px.scatter(
            df_base,
            x='x',
            y=y_col,
            color='case_id',
            title=f"Dotted Chart ({y_axis} vs {x_label})",
            labels={'x': x_label, y_col: y_axis, 'case_id': 'Case ID'}
        )
        st.plotly_chart(fig)


    with left_col:
        st.subheader('Event Log Summary')
        if 'summary' in st.session_state:
            summary = st.session_state['summary']
            compact_text = (
                f"Cases: {summary['Number of cases']} | "
                f"Events: {summary['Number of events']} | \n"
                f"Freq. Activity: \n{summary['Most frequent activity']}\n"
                f"Start: \n{summary['Most common start activity']} | \n"
                f"End: \n{summary['Most common end activity']} | \n"
                f"Avg. Duration: {summary['Average case duration (s)']:.1f}s"
            )
            st.info(compact_text)


    if st.button("Describe Chart with Ollama"):
        OllamaEvaluator_instance = OllamaEvaluator()
        if 'df' in st.session_state:
            df_base = st.session_state['df'].copy()
            # Compute top variants
            case_variants = df_base.groupby('case_id')['activity'].apply(lambda x: '-'.join(x))
            variant_counts = case_variants.value_counts()
            top_n = 10
            top_variants = variant_counts.head(top_n)
            if not top_variants.empty:
                description = OllamaEvaluator_instance.describe_chart(
                    top_variants, df_base)
            else:
                description = "No variants found in the data."
        else:
            description = "No data to describe. Please load a log first."
        st.subheader("Ollama Chart Description")
        st.write(description)


if __name__ == '__main__':
    main()
