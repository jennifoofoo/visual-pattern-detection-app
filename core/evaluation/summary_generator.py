import pandas as pd

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