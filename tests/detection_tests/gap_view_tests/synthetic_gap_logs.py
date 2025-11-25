"""
Minimal, deterministic synthetic event logs for testing GapPattern detection.

Each function creates a DataFrame with a guaranteed detectable gap.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_numeric_y_gap():
    """
    Create synthetic log with numeric X and Y, containing a clear 2D gap.
    
    Gap region: x in [0.4, 0.6] AND y in [0.3, 0.7] is completely empty.
    Four dense clusters around the gap.
    Deterministic dense walls to completely isolate the gap from border.
    
    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'x', 'y']
    """
    np.random.seed(42)
    events = []
    
    # Cluster 1: bottom-left (x < 0.4, y < 0.3)
    for i in range(30):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A',
            'x': np.random.uniform(0.0, 0.4),
            'y': np.random.uniform(0.0, 0.3)
        })
    
    # Cluster 2: bottom-right (x > 0.6, y < 0.3)
    for i in range(30):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A',
            'x': np.random.uniform(0.6, 1.0),
            'y': np.random.uniform(0.0, 0.3)
        })
    
    # Cluster 3: top-left (x < 0.4, y > 0.7)
    for i in range(30):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A',
            'x': np.random.uniform(0.0, 0.4),
            'y': np.random.uniform(0.7, 1.0)
        })
    
    # Cluster 4: top-right (x > 0.6, y > 0.7)
    for i in range(30):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A',
            'x': np.random.uniform(0.6, 1.0),
            'y': np.random.uniform(0.7, 1.0)
        })
    
    # Deterministic dense walls to completely isolate the gap from border:
    # Left wall: x = 0.39 for ALL y values (0 to 1)
    # Use 3000 points evenly spaced to ensure all 150 grid bins are filled
    # Map each y-bin directly: y_bin = i / 150.0 for i in range(150)
    for y_bin_idx in range(150):
        y_val = y_bin_idx / 150.0 + (1.0 / 150.0 / 2.0)  # Center of each bin
        events.append({
            'case_id': f'Case_{y_bin_idx % 5}',
            'activity': 'A',
            'x': 0.39,
            'y': y_val
        })
    
    # Right wall: x = 0.62 for ALL y values (0 to 1)
    # Map each y-bin directly: y_bin = i / 150.0 for i in range(150)
    for y_bin_idx in range(150):
        y_val = y_bin_idx / 150.0 + (1.0 / 150.0 / 2.0)  # Center of each bin
        events.append({
            'case_id': f'Case_{y_bin_idx % 5}',
            'activity': 'A',
            'x': 0.62,
            'y': y_val
        })
    
    # Bottom wall: y = 0.29 for ALL x values between 0.39-0.62
    # Map each x-bin directly between 0.39-0.62
    # Use wider range to ensure complete coverage: 0.38-0.63
    x_left_bin = int(0.38 * 150)  # Bin 57 (wider to ensure coverage)
    x_right_bin = int(0.63 * 150)  # Bin 94 (wider to ensure coverage)
    for x_bin_idx in range(x_left_bin, x_right_bin + 1):
        x_val = x_bin_idx / 150.0 + (1.0 / 150.0 / 2.0)  # Center of each bin
        events.append({
            'case_id': f'Case_{x_bin_idx % 5}',
            'activity': 'A',
            'x': x_val,
            'y': 0.29
        })
        # Add extra point at bin edge to ensure no gaps
        if x_bin_idx < x_right_bin:
            x_val_edge = (x_bin_idx + 1) / 150.0 - 0.0001  # Just before next bin
            events.append({
                'case_id': f'Case_{(x_bin_idx + 1) % 5}',
                'activity': 'A',
                'x': x_val_edge,
                'y': 0.29
            })
    
    # Top wall: y = 0.71 for ALL x values between 0.39-0.62
    # Use same wider range
    for x_bin_idx in range(x_left_bin, x_right_bin + 1):
        x_val = x_bin_idx / 150.0 + (1.0 / 150.0 / 2.0)  # Center of each bin
        events.append({
            'case_id': f'Case_{x_bin_idx % 5}',
            'activity': 'A',
            'x': x_val,
            'y': 0.71
        })
        # Add extra point at bin edge to ensure no gaps
        if x_bin_idx < x_right_bin:
            x_val_edge = (x_bin_idx + 1) / 150.0 - 0.0001  # Just before next bin
            events.append({
                'case_id': f'Case_{(x_bin_idx + 1) % 5}',
                'activity': 'A',
                'x': x_val_edge,
                'y': 0.71
            })
    
    # Gap region [0.4, 0.6] x [0.3, 0.7] is intentionally empty and now isolated
    
    df = pd.DataFrame(events)
    return df


def make_categorical_y_gap():
    """
    Create synthetic log with numeric X and categorical Y.
    
    Category A: has a horizontal gap in x [0.4, 0.6], surrounded by data
    Category B: fully filled (no gap)
    Category C: fully filled (no gap)
    
    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'x', 'y']
    """
    np.random.seed(42)
    events = []
    
    # Category A: dense data before gap (x in [0.0, 0.4])
    for i in range(80):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A',
            'x': np.random.uniform(0.0, 0.4),
            'y': 'A'
        })
    
    # Category A: dense data after gap (x in [0.6, 1.0])
    for i in range(80):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A',
            'x': np.random.uniform(0.6, 1.0),
            'y': 'A'
        })
    
    # Category A: gap in [0.4, 0.6] is intentionally empty
    # The dense data before/after ensures the gap is isolated
    
    # Category B: fully filled across the whole X range (no horizontal gaps)
    for i in range(100):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'B',
            'x': i / 99.0,   # monotonic 0 → 1
            'y': 'B'
        })
    
    # Category C: fully filled across the whole X range (no horizontal gaps)
    for i in range(100):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'C',
            'x': i / 99.0,   # monotonic 0 → 1
            'y': 'C'
        })
    
    df = pd.DataFrame(events)
    return df


def make_actual_time_gap():
    """
    Create synthetic log with actual_time X-axis and a clean 30-minute gap.
    
    Structure:
        - 50 events from t0 to t0+10min (dense)
        - GAP: 30 minutes with absolutely no events
        - 50 events after the gap
    
    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'y']
    """
    np.random.seed(42)
    events = []
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    
    # 50 events before gap (t0 to t0+10min, every 12 seconds)
    for i in range(50):
        event_time = t0 + timedelta(seconds=i * 12)
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'actual_time': event_time,
            'y': 'A' if np.random.rand() < 0.5 else 'B'
        })
    
    # GAP: 30 minutes (t0+10min to t0+40min) - NO EVENTS
    
    # 50 events after gap (t0+40min to t0+50min, every 12 seconds)
    gap_duration = timedelta(minutes=30)
    gap_start_time = t0 + timedelta(minutes=10)
    gap_end_time = gap_start_time + gap_duration
    
    for i in range(50):
        event_time = gap_end_time + timedelta(seconds=i * 12)
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'actual_time': event_time,
            'y': 'A' if np.random.rand() < 0.5 else 'B'
        })
    
    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    return df


def make_relative_time_gap():
    """
    Create synthetic log with relative_time X-axis and a clean gap.
    
    Structure:
        - Dense data: 0-600s (surrounding gap on left)
        - GAP: 600-900s empty
        - Dense data: 900-1500s (surrounding gap on right)
        - Additional data before 0 and after 1500s to ensure gap is isolated
    
    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'relative_time', 'y']
    """
    np.random.seed(42)
    events = []
    
    # Dense data before gap: 0-600s (every 6 seconds)
    for i in range(100):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'relative_time': float(i * 6),
            'y': np.random.uniform(0.0, 1.0)
        })
    
    # GAP: 600-900s - NO EVENTS
    
    # Dense data after gap: 900-1500s (every 6 seconds)
    gap_start = 600
    gap_end = 900
    for i in range(100):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'relative_time': float(gap_end + i * 6),
            'y': np.random.uniform(0.0, 1.0)
        })
    
    df = pd.DataFrame(events)
    return df


def make_relative_ratio_gap():
    """
    Create synthetic log with relative_ratio X-axis and a gap in 0.35-0.50.
    Gap is surrounded by dense data.
    
    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'relative_ratio', 'y']
    """
    np.random.seed(42)
    events = []
    
    # Dense data before gap: 0.0-0.35 (many points)
    for i in range(70):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'relative_ratio': np.random.uniform(0.0, 0.35),
            'y': np.random.uniform(0.0, 1.0)
        })
    
    # GAP: 0.35-0.50 - NO EVENTS
    
    # Dense data after gap: 0.50-1.0 (many points)
    gap_start = 0.35
    gap_end = 0.50
    for i in range(100):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'relative_ratio': np.random.uniform(gap_end, 1.0),
            'y': np.random.uniform(0.0, 1.0)
        })
    
    df = pd.DataFrame(events)
    return df


def make_logical_time_gap():
    """
    Create synthetic log with logical_time X-axis and a gap at indices 100-150.
    Gap is surrounded by dense data.
    
    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'logical_time', 'y']
    """
    np.random.seed(42)
    events = []
    
    # Dense data before gap: 0-100 (every step)
    for i in range(100):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'logical_time': i,
            'y': np.random.uniform(0.0, 1.0)
        })
    
    # GAP: 100-150 - NO EVENTS
    
    # Dense data after gap: 150-250 (every step)
    gap_start = 100
    gap_end = 150
    for i in range(100):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'logical_time': gap_end + i,
            'y': np.random.uniform(0.0, 1.0)
        })
    
    df = pd.DataFrame(events)
    return df


def make_logical_relative_gap():
    """
    Create synthetic log with logical_relative X-axis and a gap in 0.40-0.55.
    Gap is surrounded by dense data.
    
    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'logical_relative', 'y']
    """
    np.random.seed(42)
    events = []
    
    # Dense data before gap: 0.0-0.40 (many points)
    for i in range(80):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'logical_relative': np.random.uniform(0.0, 0.40),
            'y': np.random.uniform(0.0, 1.0)
        })
    
    # GAP: 0.40-0.55 - NO EVENTS
    
    # Dense data after gap: 0.55-1.0 (many points)
    gap_start = 0.40
    gap_end = 0.55
    for i in range(90):
        events.append({
            'case_id': f'Case_{i % 5}',
            'activity': 'A' if np.random.rand() < 0.5 else 'B',
            'logical_relative': np.random.uniform(gap_end, 1.0),
            'y': np.random.uniform(0.0, 1.0)
        })
    
    df = pd.DataFrame(events)
    return df


def list_all_test_logs():
    """
    Return a dictionary of all test logs.
    
    Returns
    -------
    dict
        Dictionary mapping log names to DataFrames
    """
    return {
        "numeric_y": make_numeric_y_gap(),
        "categorical_y": make_categorical_y_gap(),
        "actual_time": make_actual_time_gap(),
        "relative_time": make_relative_time_gap(),
        "relative_ratio": make_relative_ratio_gap(),
        "logical_time": make_logical_time_gap(),
        "logical_relative": make_logical_relative_gap(),
    }
