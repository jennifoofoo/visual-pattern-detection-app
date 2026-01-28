import pandas as pd
import pm4py

import streamlit as st

import numpy as np # Used for safe division/handling NaT values

'''
pm4py defaults: 
1. 'case:concept:name' = case id
2. 'concept:name' = activity
3. 'time:timestamp' = actual time

Times Representations from the paper:
0. Actual Time: normal timelime
1. Relative Time: first Events are positioned at time 0
2. Relative Ratio: 1. + last Events are positioned at max(time) 
3. Logical Time: Sequence of all Events across all traces
4. Logical Relative: 3. + first Events are positioned at time 0

XES Event Logs Data that we probably care about:
Traces:
    Case_ID = one actual trace      -> max(case_id) = amount of traces
    start_time = timestamp of first event in trace
    end_time = timestamp of last event in trace
    total_events = amount of events
Events:
    # 
    'case_id'
    'activity'
    'resource'
    
    # time representations
    'actual_time' # which is the same as 'time:timestamp'
    'relative_time'
    'relative_ratio'
    'logical_time'
    'logical_relative'
'''

def load_xes_log(xes_path):
    """
    :param xes_path: Path to XES log file.
    :return: DataFrame containing Event Data:
        'case_id', 'activity', 'resource',
        'actual_time', 'relative_time', 'relative_ratio', 
        'logical_time', 'logical_relative'
    """
    # 1. Load the XES log
    df = pm4py.read_xes(xes_path)

    # Check if all required columns are present
    validate_xes_input(df)  # df contains now case_id, activity, actual time and resource

    # compute differen time attributes
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # 1. relative_time
    df['start_time'] = df.groupby('case:concept:name')['time:timestamp'].transform('min')
    df['relative_time'] = (df['time:timestamp'] - df['start_time']).dt.total_seconds()
    # ToDo: remove this if we need start time of a trace
    df = df.drop(columns=['start_time'])
    # print(df[['case:concept:name', 'concept:name', 'time:timestamp', 'relative_time']].head())

    # 2. relative_ratio
    # maximum relative_time is the total duration because it's relatively the last
    df['trace_duration'] = df.groupby('case:concept:name')['relative_time'].transform('max')
    df['relative_ratio'] = np.where(
        # Condition: duration_in_seconds == 0
        df['trace_duration'] == 0,

        # Value if True: 
        # If duration is 0, the event is the only event or all events are instantaneous.
        # Since relative_time must also be 0 in this case, the ratio is 0.
        0.0,

        # Value if False: relative_ratio = relative_time / duration_in_seconds
        df['relative_time'] / df['trace_duration']
    )
    df = df.drop(columns=['trace_duration'])
    # print(df[['case:concept:name', 'concept:name', 'time:timestamp', 'relative_ratio']].sort_values(by='time:timestamp').head(7))

    # 3. logical_time
    df_sorted = df.sort_values(by='time:timestamp', ascending=True).copy()
    # The order is preserved because the DataFrame is already sorted
    codes, unique_timestamps = pd.factorize(df_sorted['time:timestamp'])
    df_sorted['logical_time'] = codes
    df['logical_time'] = df_sorted['logical_time']
    # print(df[['case:concept:name', 'concept:name', 'time:timestamp', 'logical_time']].sort_values(by='time:timestamp').head(7))

    # 4. logical_relative
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'], ascending=True)
    df['logical_relative'] = (
        df.groupby('case:concept:name')['time:timestamp']
        .transform(lambda x: pd.factorize(x)[0])
    )
    # print(df[['case:concept:name', 'concept:name',  'time:timestamp', 'logical_relative']].head(7))
    rename_df_columns(df)

    return df

def validate_xes_input(dataframe):
    '''
    1. case_id (case:concept:name)
    2. activity (concept:name)
    3. timestamp (time:timestamp)
    4. resource (org:group) - this one is not default by pm4py
    '''
    required_columns_base = [
        'case:concept:name',
        'concept:name',
        'time:timestamp'
    ]
    
    has_base = all(col in dataframe.columns for col in required_columns_base)
    has_resource = 'org:group' in dataframe.columns or 'org:resource' in dataframe.columns
    
    data_is_usable = has_base and has_resource

    if data_is_usable:
        return

    # handle invalid columnnames:
    handle_unreadable_xes_file()


def handle_unreadable_xes_file():
    st.session_state.log_loaded = False
    # # Print the missing ones
    # missing = [col for col in required_columns if col not in dataframe.columns]
    # print(f"\n❌ Missing columns: {missing}")
    # TODO: add streamlit input for the missing columns
    # 
    # dataframe = pm4py.format_dataframe(
    #     dataframe, 
    #     case_id='case:concept:name', # insert streamlit input
    #     activity_key='concept:name', # insert streamlit input
    #     timestamp_key='time:timestamp' # insert streamlit input
    # )

def rename_df_columns(dataframe):
    # Handle resource column renaming with fallback
    if 'org:group' in dataframe.columns:
        resource_col = 'org:group'
    elif 'org:resource' in dataframe.columns:
        resource_col = 'org:resource'
    else:
        resource_col = None

    column_mapping = {
        'case:concept:name': 'case_id',
        'concept:name': 'activity',
        'time:timestamp': 'actual_time',
    }
    
    if resource_col:
        column_mapping[resource_col] = 'resource'
        
    dataframe.rename(columns=column_mapping, inplace=True)




if __name__ == '__main__':
    # load_xes_log("./data/Hospital_log.xes")
    load_xes_log("data/Sepsis Cases - Event Log.xes/Sepsis Cases - Event Log.xes")