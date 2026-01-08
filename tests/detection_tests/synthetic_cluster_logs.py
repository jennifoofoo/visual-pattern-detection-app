"""
Minimal, deterministic synthetic event logs for testing TemporalClusterPattern detection.

Each function creates a DataFrame with a guaranteed detectable temporal cluster pattern.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_temporal_bursts():
    """
    Create synthetic log with distinct temporal bursts.

    Three burst periods separated by quiet periods.
    Burst 1: 50 events in first 30 minutes
    Quiet: 30 minutes with 5 events
    Burst 2: 50 events in next 30 minutes
    Quiet: 30 minutes with 5 events
    Burst 3: 50 events in next 30 minutes

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Burst 1: 50 events in first 30 minutes
    for i in range(50):
        events.append({
            'case_id': f'C{i % 10}',
            'activity': f'Activity_{i % 5}',
            'resource': f'R{i % 3}',
            'actual_time': base_time + timedelta(minutes=np.random.randint(0, 30))
        })

    # Quiet period 1: 5 events over 30 minutes
    for i in range(5):
        events.append({
            'case_id': f'C{i}',
            'activity': 'Activity_0',
            'resource': 'R0',
            'actual_time': base_time + timedelta(minutes=30 + i * 6)
        })

    # Burst 2: 50 events in next 30 minutes (60-90 min)
    for i in range(50):
        events.append({
            'case_id': f'C{i % 10}',
            'activity': f'Activity_{i % 5}',
            'resource': f'R{i % 3}',
            'actual_time': base_time + timedelta(minutes=60 + np.random.randint(0, 30))
        })

    # Quiet period 2: 5 events over 30 minutes
    for i in range(5):
        events.append({
            'case_id': f'C{i}',
            'activity': 'Activity_0',
            'resource': 'R0',
            'actual_time': base_time + timedelta(minutes=90 + i * 6)
        })

    # Burst 3: 50 events in next 30 minutes (120-150 min)
    for i in range(50):
        events.append({
            'case_id': f'C{i % 10}',
            'activity': f'Activity_{i % 5}',
            'resource': f'R{i % 3}',
            'actual_time': base_time + timedelta(minutes=120 + np.random.randint(0, 30))
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_activity_time_clusters():
    """
    Create synthetic log with activity-time clustering.

    Different activities cluster at different times of day.
    Activity A: mostly 9-11 AM
    Activity B: mostly 11-13 PM
    Activity C: mostly 13-15 PM

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'relative_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Activity A: 9-11 AM (50 events)
    for i in range(50):
        actual_time = base_time + timedelta(hours=np.random.uniform(0, 2))
        events.append({
            'case_id': f'C{i % 15}',
            'activity': 'Activity_A',
            'resource': f'R{i % 3}',
            'actual_time': actual_time,
            'relative_time': (actual_time - base_time).total_seconds()
        })

    # Activity B: 11 AM - 1 PM (50 events)
    for i in range(50):
        actual_time = base_time + timedelta(hours=2 + np.random.uniform(0, 2))
        events.append({
            'case_id': f'C{i % 15}',
            'activity': 'Activity_B',
            'resource': f'R{i % 3}',
            'actual_time': actual_time,
            'relative_time': (actual_time - base_time).total_seconds()
        })

    # Activity C: 1-3 PM (50 events)
    for i in range(50):
        actual_time = base_time + timedelta(hours=4 + np.random.uniform(0, 2))
        events.append({
            'case_id': f'C{i % 15}',
            'activity': 'Activity_C',
            'resource': f'R{i % 3}',
            'actual_time': actual_time,
            'relative_time': (actual_time - base_time).total_seconds()
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_parallel_cases():
    """
    Create synthetic log with significant case parallelism.

    10 cases running in parallel with overlapping time periods.
    Each case has 5 events spread over 2 hours.
    All cases start within the first 30 minutes.

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Create 10 parallel cases
    for case_idx in range(10):
        case_id = f'C{case_idx}'
        case_start = base_time + \
            timedelta(minutes=case_idx * 3)  # Start within 30 min

        # Each case has 5 events over 2 hours
        for event_idx in range(5):
            events.append({
                'case_id': case_id,
                'activity': f'Activity_{event_idx}',
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 30)
            })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_resource_shift_patterns():
    """
    Create synthetic log with resource shift patterns.

    Three resources working in distinct shifts:
    R1: 9-13 (morning shift, 40 events)
    R2: 13-17 (afternoon shift, 40 events)
    R3: 17-21 (evening shift, 40 events)

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # R1: Morning shift (9-13, 40 events)
    for i in range(40):
        events.append({
            'case_id': f'C{i}',
            'activity': f'Activity_{i % 5}',
            'resource': 'R1',
            'actual_time': base_time + timedelta(hours=np.random.uniform(0, 4))
        })

    # R2: Afternoon shift (13-17, 40 events)
    for i in range(40):
        events.append({
            'case_id': f'C{i + 40}',
            'activity': f'Activity_{i % 5}',
            'resource': 'R2',
            'actual_time': base_time + timedelta(hours=4 + np.random.uniform(0, 4))
        })

    # R3: Evening shift (17-21, 40 events)
    for i in range(40):
        events.append({
            'case_id': f'C{i + 80}',
            'activity': f'Activity_{i % 5}',
            'resource': 'R3',
            'actual_time': base_time + timedelta(hours=8 + np.random.uniform(0, 4))
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_variant_timing_patterns():
    """
    Create synthetic log with variant-specific timing patterns.

    Different process variants occur at different times:
    Variant A->B: mostly morning
    Variant A->C: mostly afternoon

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'variant', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Variant A->B: Morning (20 cases)
    for case_idx in range(20):
        case_id = f'C_AB_{case_idx}'
        case_start = base_time + timedelta(hours=np.random.uniform(0, 3))

        events.append({
            'case_id': case_id,
            'activity': 'A',
            'variant': 'A-B',
            'resource': f'R{case_idx % 3}',
            'actual_time': case_start
        })
        events.append({
            'case_id': case_id,
            'activity': 'B',
            'variant': 'A-B',
            'resource': f'R{case_idx % 3}',
            'actual_time': case_start + timedelta(minutes=30)
        })

    # Variant A->C: Afternoon (20 cases)
    for case_idx in range(20):
        case_id = f'C_AC_{case_idx}'
        case_start = base_time + timedelta(hours=4 + np.random.uniform(0, 3))

        events.append({
            'case_id': case_id,
            'activity': 'A',
            'variant': 'A-C',
            'resource': f'R{case_idx % 3}',
            'actual_time': case_start
        })
        events.append({
            'case_id': case_id,
            'activity': 'C',
            'variant': 'A-C',
            'resource': f'R{case_idx % 3}',
            'actual_time': case_start + timedelta(minutes=30)
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df
