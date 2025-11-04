import pandas as pd

def summarize_event_log(df):
    summary = {}
    
    # Check if the required timestamp column exists
    timestamp_col = 'actual_time'
    if timestamp_col not in df.columns:
        # Fallback or raise an error if the expected column is missing
        print(f"Warning: DataFrame is missing the required timestamp column '{timestamp_col}'. "
              "Timestamp-based calculations will be skipped or may fail.")
        # Proceed with non-timestamp calculations
        summary['Number of cases'] = df['case_id'].nunique()
        summary['Number of events'] = len(df)
        summary['Most frequent activity'] = df['activity'].value_counts().idxmax()
        return summary
        
    # --- General Statistics ---
    summary['Number of cases'] = df['case_id'].nunique()
    summary['Number of events'] = len(df)
    summary['Most frequent activity'] = df['activity'].value_counts().idxmax()
    
    # --- Start and End Activities ---
    # Note: Use the actual timestamp for reliable sorting if needed, but here .first()/.last() 
    # relies on the DataFrame already being ordered by time within each group, which 
    # load_xes_log ensures.
    start_activities = df.groupby('case_id').first()['activity'].value_counts()
    end_activities = df.groupby('case_id').last()['activity'].value_counts()
    summary['Most common start activity'] = start_activities.idxmax()
    summary['Most common end activity'] = end_activities.idxmax()
    
    # --- Average Case Duration ---
    durations = df.groupby('case_id').apply(lambda x: (
        x[timestamp_col].max() - x[timestamp_col].min()).total_seconds())
    
    # Add start and end dates for the entire log
    summary['Log Start Date'] = df[timestamp_col].min()
    summary['Log End Date'] = df[timestamp_col].max()
    
    summary['Average case duration (s)'] = durations.mean()
    
    return summary