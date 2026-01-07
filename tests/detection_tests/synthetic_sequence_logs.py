"""
Minimal, deterministic synthetic event logs for testing sequence pattern detection.

Each function creates a DataFrame with guaranteed detectable sequence patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_common_sequence_pattern():
    """
    Create synthetic log with a highly common sequence pattern.

    Most cases (80%) follow sequence A->B->C->D.
    Some cases (20%) follow sequence A->B->E->D.

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Common sequence: A->B->C->D (40 cases = 80%)
    for case_idx in range(40):
        case_id = f'C_common_{case_idx}'
        case_start = base_time + timedelta(hours=case_idx * 0.25)

        for event_idx, activity in enumerate(['A', 'B', 'C', 'D']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    # Alternative sequence: A->B->E->D (10 cases = 20%)
    for case_idx in range(10):
        case_id = f'C_alt_{case_idx}'
        case_start = base_time + timedelta(hours=10 + case_idx * 0.25)

        for event_idx, activity in enumerate(['A', 'B', 'E', 'D']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_rare_sequence_pattern():
    """
    Create synthetic log with rare sequence outliers.

    Most cases follow A->B->C (30 cases).
    One case follows rare sequence X->Y->Z.

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Common sequence: A->B->C (30 cases)
    for case_idx in range(30):
        case_id = f'C{case_idx}'
        case_start = base_time + timedelta(hours=case_idx * 0.5)

        for event_idx, activity in enumerate(['A', 'B', 'C']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 20)
            })

    # Rare sequence: X->Y->Z (1 case)
    case_start = base_time + timedelta(hours=20)
    for event_idx, activity in enumerate(['X', 'Y', 'Z']):
        events.append({
            'case_id': 'C_rare',
            'activity': activity,
            'resource': 'R0',
            'actual_time': case_start + timedelta(minutes=event_idx * 20)
        })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_loop_sequence_pattern():
    """
    Create synthetic log with loop patterns.

    Some cases have repeated activities (loops).
    Case pattern: A->B->B->C (B repeats)

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Cases with loop: A->B->B->C (20 cases)
    for case_idx in range(20):
        case_id = f'C_loop_{case_idx}'
        case_start = base_time + timedelta(hours=case_idx * 0.3)

        for event_idx, activity in enumerate(['A', 'B', 'B', 'C']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    # Normal cases without loop: A->B->C (10 cases)
    for case_idx in range(10):
        case_id = f'C_normal_{case_idx}'
        case_start = base_time + timedelta(hours=10 + case_idx * 0.3)

        for event_idx, activity in enumerate(['A', 'B', 'C']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_skip_sequence_pattern():
    """
    Create synthetic log with skip patterns.

    Most cases follow full sequence A->B->C->D.
    Some cases skip step C: A->B->D (shortcut pattern).

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Full sequence: A->B->C->D (30 cases)
    for case_idx in range(30):
        case_id = f'C_full_{case_idx}'
        case_start = base_time + timedelta(hours=case_idx * 0.25)

        for event_idx, activity in enumerate(['A', 'B', 'C', 'D']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    # Skip sequence: A->B->D (skip C, 10 cases)
    for case_idx in range(10):
        case_id = f'C_skip_{case_idx}'
        case_start = base_time + timedelta(hours=10 + case_idx * 0.25)

        for event_idx, activity in enumerate(['A', 'B', 'D']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_parallel_sequence_pattern():
    """
    Create synthetic log with parallel execution patterns.

    Activities B and C can happen in parallel (interleaved).
    Pattern: A->(B,C)->D where B and C can be in any order.

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Cases with B->C order (15 cases)
    for case_idx in range(15):
        case_id = f'C_BC_{case_idx}'
        case_start = base_time + timedelta(hours=case_idx * 0.2)

        for event_idx, activity in enumerate(['A', 'B', 'C', 'D']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 10)
            })

    # Cases with C->B order (15 cases)
    for case_idx in range(15):
        case_id = f'C_CB_{case_idx}'
        case_start = base_time + timedelta(hours=5 + case_idx * 0.2)

        for event_idx, activity in enumerate(['A', 'C', 'B', 'D']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 10)
            })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df


def make_multi_variant_sequence():
    """
    Create synthetic log with multiple distinct variants.

    Variant 1: A->B->C (40% of cases)
    Variant 2: A->D->E (30% of cases)
    Variant 3: A->F->G->H (20% of cases)
    Variant 4: X->Y (10% of cases, rare)

    Returns
    -------
    pd.DataFrame
        Columns: ['case_id', 'activity', 'actual_time', 'variant', 'resource']
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []

    # Variant 1: A->B->C (20 cases, 40%)
    for case_idx in range(20):
        case_id = f'C_V1_{case_idx}'
        case_start = base_time + timedelta(hours=case_idx * 0.15)

        for event_idx, activity in enumerate(['A', 'B', 'C']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'variant': 'V1',
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    # Variant 2: A->D->E (15 cases, 30%)
    for case_idx in range(15):
        case_id = f'C_V2_{case_idx}'
        case_start = base_time + timedelta(hours=5 + case_idx * 0.15)

        for event_idx, activity in enumerate(['A', 'D', 'E']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'variant': 'V2',
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    # Variant 3: A->F->G->H (10 cases, 20%)
    for case_idx in range(10):
        case_id = f'C_V3_{case_idx}'
        case_start = base_time + timedelta(hours=8 + case_idx * 0.15)

        for event_idx, activity in enumerate(['A', 'F', 'G', 'H']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'variant': 'V3',
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    # Variant 4: X->Y (5 cases, 10%, rare)
    for case_idx in range(5):
        case_id = f'C_V4_{case_idx}'
        case_start = base_time + timedelta(hours=10 + case_idx * 0.15)

        for event_idx, activity in enumerate(['X', 'Y']):
            events.append({
                'case_id': case_id,
                'activity': activity,
                'variant': 'V4',
                'resource': f'R{case_idx % 3}',
                'actual_time': case_start + timedelta(minutes=event_idx * 15)
            })

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    df = df.sort_values('actual_time').reset_index(drop=True)

    return df
