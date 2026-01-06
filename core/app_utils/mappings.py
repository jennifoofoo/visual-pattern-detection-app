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
    # TODO: what shall we do if there is no resource?
    # Assuming 'resource' is in the log/DataFrame
    'Resource': 'resource',
}

# Mapping for Dot Colors (Color/Dots Config)
DOTS_COLOR_MAP = {
    'Activity': 'activity',
    'Case ID': 'case_id',
    'Resource': 'resource',
}