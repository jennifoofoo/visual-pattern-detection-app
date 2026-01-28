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

# View Configuration Presets
# Pattern support: Gap, Burst, Outlier, Cluster, Sequence
VIEW_PRESETS = {
    "Resource Timeline": {
        "x_axis": "Actual time",
        "y_axis": "Resource",
        "color": "Activity",
        "description": "Who is working on what and when?",
        "axes_info": "X: Time | Y: Resource | Color: Activity",
        "patterns": ["Gap", "Burst", "Outlier", "Cluster", "Sequence"],
        "pattern_score": "5/5"
    },
    "Activity Overview": {
        "x_axis": "Actual time",
        "y_axis": "Activity",
        "color": "Resource",
        "description": "Which activities occur when?",
        "axes_info": "X: Time | Y: Activity | Color: Resource",
        "patterns": ["Gap", "Burst", "Outlier", "Cluster", "Sequence"],
        "pattern_score": "5/5"
    },
    "Activity by Case": {
        "x_axis": "Actual time",
        "y_axis": "Activity",
        "color": "Case ID",
        "description": "How do cases flow through activities?",
        "axes_info": "X: Time | Y: Activity | Color: Case ID",
        "patterns": ["Gap", "Burst", "Outlier", "Sequence"],
        "pattern_score": "4/5"
    },
    "Resource by Case": {
        "x_axis": "Actual time",
        "y_axis": "Resource",
        "color": "Case ID",
        "description": "Which resources handle which cases?",
        "axes_info": "X: Time | Y: Resource | Color: Case ID",
        "patterns": ["Gap", "Burst", "Outlier", "Sequence"],
        "pattern_score": "4/5"
    }
}