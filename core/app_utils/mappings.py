# Maps user selection to the column name generated in load_xes_log
X_AXIS_COLUMN_MAP = {
    # 0. Actual Time (timestamp)
    'Actual time': 'actual_time',
    # 1. Relative Time (seconds)
    'Relative time': 'relative_time',
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
}

# Mapping for Dot Colors (Color/Dots Config)
DOTS_COLOR_MAP = {
    'Case ID (Default)': 'case_id',
    'Activity': 'activity',
    'Resource': 'resource',
    'Event Index (in trace)': 'event_index_in_trace',
    'Global Logical Time': 'timestamp_logical_global',
}