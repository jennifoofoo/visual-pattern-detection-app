"""
Minimal, deterministic synthetic event logs for testing OutlierDetectionPattern.

Each function creates a DataFrame with a guaranteed detectable outlier.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_time_outliers():
    """
    Create synthetic log with time-based outliers.

    Most events occur during business hours (9-17), but a few at unusual times (3 AM).

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_date = datetime(2024, 1, 15)
    events = []

    # Normal business hours events (100 events between 9-17)
    for i in range(100):
        hour = np.random.randint(9, 18)
        events.append({
            'case_id': f'C{i}',
            'activity': 'Process',
            'resource': f'R{i % 5}',
            'actual_time': base_date + timedelta(hours=hour, minutes=np.random.randint(0, 60))
        })

    # Unusual time events (5 events at 3 AM - should be detected as outliers)
    for i in range(5):
        events.append({
            'case_id': f'C_night_{i}',
            'activity': 'Process',
            'resource': 'R0',
            'actual_time': base_date + timedelta(hours=3, minutes=i * 10)
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])

    return df


def make_case_duration_outliers():
    """
    Create synthetic log with case duration outliers.

    Most cases take 1-3 hours, but one case takes 48 hours (extreme outlier).

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Normal duration cases (1-3 hours each)
    for case_idx in range(20):
        case_id = f'C{case_idx}'
        duration_hours = 1 + (case_idx % 3)  # 1, 2, or 3 hours

        # Start event
        events.append({
            'case_id': case_id,
            'activity': 'Start',
            'resource': f'R{case_idx % 3}',
            'actual_time': base_time + timedelta(hours=case_idx * 0.5)
        })
        # Middle event
        events.append({
            'case_id': case_id,
            'activity': 'Process',
            'resource': f'R{case_idx % 3}',
            'actual_time': base_time + timedelta(hours=case_idx * 0.5 + duration_hours * 0.5)
        })
        # End event
        events.append({
            'case_id': case_id,
            'activity': 'End',
            'resource': f'R{case_idx % 3}',
            'actual_time': base_time + timedelta(hours=case_idx * 0.5 + duration_hours)
        })

    # Extremely long case (48 hours - should be an outlier)
    events.append({
        'case_id': 'C_long',
        'activity': 'Start',
        'resource': 'R0',
        'actual_time': base_time
    })
    events.append({
        'case_id': 'C_long',
        'activity': 'Process',
        'resource': 'R0',
        'actual_time': base_time + timedelta(hours=24)
    })
    events.append({
        'case_id': 'C_long',
        'activity': 'End',
        'resource': 'R0',
        'actual_time': base_time + timedelta(hours=48)
    })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])

    return df


def make_activity_frequency_outliers():
    """
    Create synthetic log with rare activity outliers.

    Most activities are common (100 events each), but one activity is extremely rare (1 event).

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    events = []
    base_time = datetime(2024, 1, 15, 9, 0, 0)

    # Common activities (100 events each)
    for i in range(100):
        events.append({
            'case_id': f'C{i}',
            'activity': 'Common_A',
            'resource': f'R{i % 5}',
            'actual_time': base_time + timedelta(minutes=i)
        })
    for i in range(100):
        events.append({
            'case_id': f'C{i + 100}',
            'activity': 'Common_B',
            'resource': f'R{i % 5}',
            'actual_time': base_time + timedelta(minutes=i + 100)
        })

    # Rare activity (should be detected as outlier - less than 1%)
    events.append({
        'case_id': 'C_rare',
        'activity': 'Extremely_Rare_Activity',
        'resource': 'R0',
        'actual_time': base_time + timedelta(minutes=300)
    })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])

    return df


def make_resource_workload_outliers():
    """
    Create synthetic log with resource workload outliers.

    Most resources handle 20 events, but one resource handles 200 events (extreme outlier).

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    events = []
    base_time = datetime(2024, 1, 15, 9, 0, 0)

    # Normal workload resources (20 events each)
    for resource in ['R1', 'R2', 'R3', 'R4', 'R5']:
        for i in range(20):
            events.append({
                'case_id': f'C_{resource}_{i}',
                'activity': 'Process',
                'resource': resource,
                'actual_time': base_time + timedelta(minutes=i * 5)
            })

    # High workload resource (200 events - extreme outlier)
    for i in range(200):
        events.append({
            'case_id': f'C_R_extreme_{i}',
            'activity': 'Process',
            'resource': 'R_Overworked',
            'actual_time': base_time + timedelta(minutes=i)
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])

    return df


def make_sequence_outliers():
    """
    Create synthetic log with unusual sequence outliers.

    Most cases follow A->B->C sequence, but one case follows X->Y (rare sequence).

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    events = []
    base_time = datetime(2024, 1, 15, 9, 0, 0)

    # Common sequence: A -> B -> C (20 cases)
    for i in range(20):
        case_id = f'C{i}'
        for j, activity in enumerate(['A', 'B', 'C']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{i % 3}',
                'actual_time': base_time + timedelta(hours=i, minutes=j * 20)
            })

    # Rare sequence: X -> Y (unique - should be outlier)
    for j, activity in enumerate(['X', 'Y']):
        events.append({
            'case_id': 'C_rare',
            'activity': activity,
            'resource': 'R0',
            'actual_time': base_time + timedelta(hours=25, minutes=j * 20)
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])

    return df


def make_case_complexity_outliers():
    """
    Create synthetic log with case complexity outliers.

    Most cases have 3-5 events, but one case has 50 events (extreme outlier).

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    events = []
    base_time = datetime(2024, 1, 15, 9, 0, 0)

    # Normal complexity cases (3-5 events each)
    for case_idx in range(20):
        case_id = f'C{case_idx}'
        num_events = 3 + (case_idx % 3)  # 3, 4, or 5 events

        for event_idx in range(num_events):
            events.append({
                'case_id': case_id,
                'activity': f'Activity_{event_idx}',
                'resource': f'R{case_idx % 3}',
                'actual_time': base_time + timedelta(hours=case_idx, minutes=event_idx * 10)
            })

    # Extremely complex case (50 events - should be outlier)
    for event_idx in range(50):
        events.append({
            'case_id': 'C_complex',
            'activity': f'Activity_{event_idx}',
            'resource': 'R0',
            'actual_time': base_time + timedelta(hours=25, minutes=event_idx * 5)
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])

    return df
