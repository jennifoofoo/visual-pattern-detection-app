import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
from core.utils.demo_sampling import sample_small_eventlog
from core.detection.temporal_cluster import TemporalClusterPattern

# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def event_log_path(test_data_dir) -> Path:
    """Return path to Hospital_log.xes test file."""
    path = test_data_dir / "Hospital_log.xes"
    if not path.exists():
        pytest.skip(f"Test XES file not found: {path}")
    return path


# =============================================================================
# REAL DATA FIXTURES (using sample_small_eventlog)
# =============================================================================

@pytest.fixture(scope="session")
def full_event_log(event_log_path):
    """
    Load full Hospital_log.xes file.

    Session-scoped to avoid reloading for every test.
    Use sample_event_log for most tests (faster).
    """
    from core.data_processing.loader import load_xes_log
    return load_xes_log(str(event_log_path))


@pytest.fixture(scope="session")
def sample_event_log(full_event_log):
    """
    Sampled event log from Hospital_log.xes for fast tests.

    Uses sample_small_eventlog to get 20 cases with max 15 events each.
    This is the main fixture for most tests.
    """
    return sample_small_eventlog(
        full_event_log,
        max_cases=100,
        max_events_per_case=30,
        random_state=42
    )


@pytest.fixture
def sample_event_log_with_variant(sample_event_log):
    """
    Sampled event log with variant column computed.
    """
    df = sample_event_log.copy()

    # Compute variants (activity sequences per case)
    case_variants = df.groupby('case_id')['activity'].apply(
        lambda x: '->'.join(x.astype(str))
    ).to_dict()
    df['variant'] = df['case_id'].map(case_variants)

    return df


@pytest.fixture
def tiny_event_log(full_event_log):
    """
    Very small event log (5 cases) for quick unit tests.
    """
    from core.utils.demo_sampling import sample_small_eventlog
    return sample_small_eventlog(
        full_event_log,
        max_cases=5,
        max_events_per_case=10,
        random_state=42
    )


# =============================================================================
# SYNTHETIC DATA FIXTURES (for edge cases and controlled tests)
# =============================================================================

@pytest.fixture
def synthetic_event_log() -> pd.DataFrame:
    """
    Synthetic event log with known structure for predictable testing.

    Contains 3 cases:
    - Case C1: 4 events (Start -> Process -> Review -> End), duration 3h
    - Case C2: 3 events (Start -> Process -> End), duration 2h
    - Case C3: 2 events (Start -> End), duration 30min
    """
    base_time = datetime(2024, 1, 15, 9, 0, 0)

    events = [
        # Case C1: 4 events
        {'case_id': 'C1', 'event_index': 0, 'activity': 'Start', 'resource': 'R1',
         'actual_time': base_time, 'relative_time': 0.0, 'relative_ratio': 0.0,
         'logical_time': 0, 'logical_relative': 0},
        {'case_id': 'C1', 'event_index': 1, 'activity': 'Process', 'resource': 'R2',
         'actual_time': base_time + timedelta(hours=1), 'relative_time': 3600.0,
         'relative_ratio': 1/3, 'logical_time': 1, 'logical_relative': 1},
        {'case_id': 'C1', 'event_index': 2, 'activity': 'Review', 'resource': 'R1',
         'actual_time': base_time + timedelta(hours=2), 'relative_time': 7200.0,
         'relative_ratio': 2/3, 'logical_time': 2, 'logical_relative': 2},
        {'case_id': 'C1', 'event_index': 3, 'activity': 'End', 'resource': 'R3',
         'actual_time': base_time + timedelta(hours=3), 'relative_time': 10800.0,
         'relative_ratio': 1.0, 'logical_time': 3, 'logical_relative': 3},

        # Case C2: 3 events
        {'case_id': 'C2', 'event_index': 0, 'activity': 'Start', 'resource': 'R2',
         'actual_time': base_time + timedelta(minutes=30), 'relative_time': 0.0,
         'relative_ratio': 0.0, 'logical_time': 4, 'logical_relative': 0},
        {'case_id': 'C2', 'event_index': 1, 'activity': 'Process', 'resource': 'R1',
         'actual_time': base_time + timedelta(hours=1, minutes=30), 'relative_time': 3600.0,
         'relative_ratio': 0.5, 'logical_time': 5, 'logical_relative': 1},
        {'case_id': 'C2', 'event_index': 2, 'activity': 'End', 'resource': 'R2',
         'actual_time': base_time + timedelta(hours=2, minutes=30), 'relative_time': 7200.0,
         'relative_ratio': 1.0, 'logical_time': 6, 'logical_relative': 2},

        # Case C3: 2 events
        {'case_id': 'C3', 'event_index': 0, 'activity': 'Start', 'resource': 'R3',
         'actual_time': base_time + timedelta(hours=1), 'relative_time': 0.0,
         'relative_ratio': 0.0, 'logical_time': 7, 'logical_relative': 0},
        {'case_id': 'C3', 'event_index': 1, 'activity': 'End', 'resource': 'R1',
         'actual_time': base_time + timedelta(hours=1, minutes=30), 'relative_time': 1800.0,
         'relative_ratio': 1.0, 'logical_time': 8, 'logical_relative': 1},
    ]

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    return df


@pytest.fixture
def empty_event_log() -> pd.DataFrame:
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(columns=[
        'case_id', 'event_index', 'activity', 'resource',
        'actual_time', 'relative_time', 'relative_ratio',
        'logical_time', 'logical_relative'
    ])


@pytest.fixture
def single_event_log() -> pd.DataFrame:
    """DataFrame with exactly one event."""
    df = pd.DataFrame([{
        'case_id': 'C1',
        'event_index': 0,
        'activity': 'Start',
        'resource': 'R1',
        'actual_time': datetime(2024, 1, 15, 9, 0, 0),
        'relative_time': 0.0,
        'relative_ratio': 0.0,
        'logical_time': 0,
        'logical_relative': 0
    }])
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    return df


@pytest.fixture
def df_with_gaps() -> pd.DataFrame:
    """
    DataFrame with intentional gaps for gap detection tests.

    Contains events with a 6-hour gap:
    - Morning events: 9:00 - 10:00
    - Afternoon events: 16:00 - 17:00
    """
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []
    global_idx = 0

    # Morning events
    for i in range(5):
        events.append({
            'case_id': f'C{i}', 'event_index': 0, 'activity': 'Morning',
            'resource': 'R1', 'actual_time': base_time + timedelta(minutes=i * 12),
            'relative_time': 0.0, 'relative_ratio': 0.0,
            'logical_time': global_idx, 'logical_relative': 0
        })
        global_idx += 1

    # Afternoon events (6 hour gap)
    afternoon = base_time + timedelta(hours=7)
    for i in range(5):
        events.append({
            'case_id': f'C{i + 5}', 'event_index': 0, 'activity': 'Afternoon',
            'resource': 'R2', 'actual_time': afternoon + timedelta(minutes=i * 12),
            'relative_time': 0.0, 'relative_ratio': 0.0,
            'logical_time': global_idx, 'logical_relative': 0
        })
        global_idx += 1

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    return df


@pytest.fixture
def df_with_clusters() -> pd.DataFrame:
    """
    DataFrame with intentional clusters for cluster detection tests.

    Contains 3 distinct time-based clusters.
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []
    global_idx = 0

    # Cluster 1: Morning
    for i in range(10):
        events.append({
            'case_id': f'Case_{i:03d}', 'event_index': 0, 'activity': 'Process',
            'resource': 'R1',
            'actual_time': base_time + timedelta(minutes=np.random.uniform(-30, 30)),
            'relative_time': 0.0, 'relative_ratio': 0.0,
            'logical_time': global_idx, 'logical_relative': 0
        })
        global_idx += 1

    # Cluster 2: Noon
    noon = base_time + timedelta(hours=3)
    for i in range(10):
        events.append({
            'case_id': f'Case_{i + 10:03d}', 'event_index': 0, 'activity': 'Process',
            'resource': 'R2',
            'actual_time': noon + timedelta(minutes=np.random.uniform(-30, 30)),
            'relative_time': 0.0, 'relative_ratio': 0.0,
            'logical_time': global_idx, 'logical_relative': 0
        })
        global_idx += 1

    # Cluster 3: Afternoon
    afternoon = base_time + timedelta(hours=6)
    for i in range(10):
        events.append({
            'case_id': f'Case_{i + 20:03d}', 'event_index': 0, 'activity': 'Process',
            'resource': 'R3',
            'actual_time': afternoon + timedelta(minutes=np.random.uniform(-30, 30)),
            'relative_time': 0.0, 'relative_ratio': 0.0,
            'logical_time': global_idx, 'logical_relative': 0
        })
        global_idx += 1

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    return df


# =============================================================================
# PREPROCESSOR TEST DATA (from existing test_data_preprocessor.py)
# =============================================================================

@pytest.fixture
def sample_time_data() -> pd.DataFrame:
    """Sample DataFrame for Time View testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
        'activity': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'B', 'A']
    })


@pytest.fixture
def sample_case_data() -> pd.DataFrame:
    """Sample DataFrame for Case View testing."""
    return pd.DataFrame({
        'case_id': ['C1', 'C2', 'C1', 'C3', 'C2', 'C1', 'C3', 'C3', 'C2', 'C1'],
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h')
    })


@pytest.fixture
def sample_resource_data() -> pd.DataFrame:
    """Sample DataFrame for Resource View testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
        'resource': ['R1', 'R2', 'R1', 'R3', 'R2', 'R1', 'R3', 'R3', 'R2', 'R1']
    })


@pytest.fixture
def sample_activity_data() -> pd.DataFrame:
    """Sample DataFrame for Activity View testing."""
    return pd.DataFrame({
        'activity': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'B', 'A'],
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h')
    })


@pytest.fixture
def sample_performance_data() -> pd.DataFrame:
    """Sample DataFrame for Performance View testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
        'case_duration': [100.5, 200.3, 150.7, 300.2, 180.9, 250.1, 120.4, 280.6, 160.8, 220.0]
    })


# =============================================================================
# VIEW CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def time_view_config() -> dict:
    """View config for time view (x=time, y=activity)."""
    return {'x': 'actual_time', 'y': 'activity', 'view': 'time'}


@pytest.fixture
def case_view_config() -> dict:
    """View config for case view (x=case_id, y=time)."""
    return {'x': 'case_id', 'y': 'actual_time', 'view': 'case'}


@pytest.fixture
def resource_view_config() -> dict:
    """View config for resource view (x=time, y=resource)."""
    return {'x': 'actual_time', 'y': 'resource', 'view': 'resource'}


@pytest.fixture
def activity_view_config() -> dict:
    """View config for activity view (x=activity, y=time)."""
    return {'x': 'activity', 'y': 'actual_time', 'view': 'activity'}


@pytest.fixture
def preprocessor_time_config() -> dict:
    """Preprocessor config for time view."""
    return {'x': 'timestamp', 'y': 'activity', 'view': 'time'}


@pytest.fixture
def preprocessor_case_config() -> dict:
    """Preprocessor config for case view."""
    return {'x': 'case_id', 'y': 'timestamp', 'view': 'case'}


@pytest.fixture
def preprocessor_resource_config() -> dict:
    """Preprocessor config for resource view."""
    return {'x': 'timestamp', 'y': 'resource', 'view': 'resource'}


@pytest.fixture
def preprocessor_activity_config() -> dict:
    """Preprocessor config for activity view."""
    return {'x': 'activity', 'y': 'timestamp', 'view': 'activity'}


@pytest.fixture
def preprocessor_performance_config() -> dict:
    """Preprocessor config for performance view."""
    return {'x': 'timestamp', 'y': 'case_duration', 'view': 'performance'}


# Full view config (matching all_views.txt format)
@pytest.fixture
def full_view_config_actual_time_case_activity() -> dict:
    """Full view config matching app.py format."""
    return {
        'x_axis_label': 'Actual time',
        'x_axis_column': 'actual_time',
        'y_axis_label': 'Case ID',
        'y_axis_column': 'case_id',
        'z_color_label': 'Activity',
        'z_color_column': 'activity'
    }


# =============================================================================
# PATTERN DETECTOR FIXTURES
# =============================================================================


# NORMAL CLUSTERING IS NOT AVAILABLE FOR NOW
# @pytest.fixture
# def cluster_detector(time_view_config):
#     """ClusterPattern instance with OPTICS algorithm."""
#     from core.detection.cluster_pattern import ClusterPattern
#     return ClusterPattern(view_config=time_view_config, algorithm='optics')


# @pytest.fixture
# def dbscan_detector(time_view_config):
#     """ClusterPattern instance with DBSCAN algorithm."""
#     from core.detection.cluster_pattern import ClusterPattern
#     return ClusterPattern(view_config=time_view_config, algorithm='dbscan')


@pytest.fixture
def gap_detector(time_view_config):
    """GapPattern instance."""
    from core.detection.gap_pattern import GapPattern
    return GapPattern(view_config=time_view_config)


@pytest.fixture
def gap_detector_grouped(time_view_config):
    """GapPattern instance with group_by_y enabled."""
    from core.detection.gap_pattern import GapPattern
    return GapPattern(view_config=time_view_config, group_by_y=True)


# =============================================================================
# TEMPORAL CLUSTER PATTERN FIXTURES
# =============================================================================

@pytest.fixture
def temporal_cluster_detector_factory():
    """
    Factory fixture to create TemporalClusterPattern with custom parameters.

    Usage:
        def test_temporal(temporal_cluster_detector_factory, sample_event_log):
            detector = temporal_cluster_detector_factory(
                df=sample_event_log,
                x_axis='actual_time',
                y_axis='activity'
            )
            detector.detect()
    """
    def _create_detector(df: pd.DataFrame, x_axis: str = 'actual_time',
                         y_axis: str = 'activity', min_cluster_size: int = 5,
                         temporal_eps: float = None, spatial_eps: float = None):
        return TemporalClusterPattern(
            df=df,
            x_axis=x_axis,
            y_axis=y_axis,
            min_cluster_size=min_cluster_size,
            temporal_eps=temporal_eps,
            spatial_eps=spatial_eps
        )
    return _create_detector


@pytest.fixture
def temporal_burst_detector(sample_event_log):
    """
    TemporalClusterPattern configured for temporal burst detection.

    X=actual_time, Y=activity (detects event concentration periods)
    """
    return TemporalClusterPattern(
        df=sample_event_log,
        x_axis='actual_time',
        y_axis='activity',
        min_cluster_size=3
    )


@pytest.fixture
def activity_cluster_detector(sample_event_log):
    """
    TemporalClusterPattern configured for activity-time clustering.

    X=relative_time, Y=activity (detects activity timing patterns)
    """
    return TemporalClusterPattern(
        df=sample_event_log,
        x_axis='relative_time',
        y_axis='activity',
        min_cluster_size=3
    )


@pytest.fixture
def case_parallelism_detector(sample_event_log):
    """
    TemporalClusterPattern configured for case parallelism detection.

    X=actual_time, Y=case_id (detects concurrent case execution)
    """
    return TemporalClusterPattern(
        df=sample_event_log,
        x_axis='actual_time',
        y_axis='case_id',
        min_cluster_size=3
    )


@pytest.fixture
def resource_pattern_detector(sample_event_log):
    """
    TemporalClusterPattern configured for resource pattern detection.

    X=actual_time, Y=resource (detects resource shift patterns)
    """
    return TemporalClusterPattern(
        df=sample_event_log,
        x_axis='actual_time',
        y_axis='resource',
        min_cluster_size=3
    )


@pytest.fixture
def variant_timing_detector(sample_event_log_with_variant):
    """
    TemporalClusterPattern configured for variant timing detection.

    X=relative_ratio, Y=variant (detects variant timing differences)
    """
    return TemporalClusterPattern(
        df=sample_event_log_with_variant,
        x_axis='relative_ratio',
        y_axis='variant',
        min_cluster_size=3
    )


@pytest.fixture
def df_with_temporal_bursts() -> pd.DataFrame:
    """
    DataFrame with intentional temporal bursts for testing.

    Contains 3 distinct burst periods:
    - Burst 1: 9:00-9:15 (15 events)
    - Quiet period: 9:15-11:00
    - Burst 2: 11:00-11:15 (15 events)
    - Quiet period: 11:15-14:00
    - Burst 3: 14:00-14:15 (15 events)
    """
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []
    global_idx = 0

    activities = ['Process', 'Review', 'Approve']
    resources = ['R1', 'R2', 'R3']

    # Burst 1: 9:00-9:15
    for i in range(15):
        events.append({
            'case_id': f'C{global_idx}',
            'event_index': 0,
            'activity': activities[i % 3],
            'resource': resources[i % 3],
            'actual_time': base_time + timedelta(minutes=i),
            'relative_time': 0.0,
            'relative_ratio': 0.0,
            'logical_time': global_idx,
            'logical_relative': 0
        })
        global_idx += 1

    # Burst 2: 11:00-11:15 (2 hour gap)
    burst2_start = base_time + timedelta(hours=2)
    for i in range(15):
        events.append({
            'case_id': f'C{global_idx}',
            'event_index': 0,
            'activity': activities[i % 3],
            'resource': resources[i % 3],
            'actual_time': burst2_start + timedelta(minutes=i),
            'relative_time': 0.0,
            'relative_ratio': 0.0,
            'logical_time': global_idx,
            'logical_relative': 0
        })
        global_idx += 1

    # Burst 3: 14:00-14:15 (3 hour gap)
    burst3_start = base_time + timedelta(hours=5)
    for i in range(15):
        events.append({
            'case_id': f'C{global_idx}',
            'event_index': 0,
            'activity': activities[i % 3],
            'resource': resources[i % 3],
            'actual_time': burst3_start + timedelta(minutes=i),
            'relative_time': 0.0,
            'relative_ratio': 0.0,
            'logical_time': global_idx,
            'logical_relative': 0
        })
        global_idx += 1

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    return df


@pytest.fixture
def df_with_parallel_cases() -> pd.DataFrame:
    """
    DataFrame with intentional case parallelism for testing.

    Contains 10 cases with overlapping execution times:
    - Peak parallelism: 5 cases running simultaneously
    """
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    events = []
    global_idx = 0

    # Create 10 cases with overlapping durations
    case_configs = [
        # (start_offset_hours, duration_hours)
        (0, 3),      # C0: 9:00 - 12:00
        (0.5, 2.5),  # C1: 9:30 - 12:00
        (1, 3),      # C2: 10:00 - 13:00
        (1.5, 2),    # C3: 10:30 - 12:30
        (2, 2.5),    # C4: 11:00 - 13:30
        (2.5, 2),    # C5: 11:30 - 13:30
        (3, 3),      # C6: 12:00 - 15:00
        (3.5, 2),    # C7: 12:30 - 14:30
        (4, 2),      # C8: 13:00 - 15:00
        (5, 1),      # C9: 14:00 - 15:00
    ]

    for case_idx, (start_offset, duration) in enumerate(case_configs):
        case_id = f'C{case_idx}'
        case_start = base_time + timedelta(hours=start_offset)
        case_end = case_start + timedelta(hours=duration)

        # Add start event
        events.append({
            'case_id': case_id,
            'event_index': 0,
            'activity': 'Start',
            'resource': f'R{case_idx % 3 + 1}',
            'actual_time': case_start,
            'relative_time': 0.0,
            'relative_ratio': 0.0,
            'logical_time': global_idx,
            'logical_relative': 0
        })
        global_idx += 1

        # Add middle event
        events.append({
            'case_id': case_id,
            'event_index': 1,
            'activity': 'Process',
            'resource': f'R{(case_idx + 1) % 3 + 1}',
            'actual_time': case_start + timedelta(hours=duration / 2),
            'relative_time': duration * 1800,  # half duration in seconds
            'relative_ratio': 0.5,
            'logical_time': global_idx,
            'logical_relative': 1
        })
        global_idx += 1

        # Add end event
        events.append({
            'case_id': case_id,
            'event_index': 2,
            'activity': 'End',
            'resource': f'R{(case_idx + 2) % 3 + 1}',
            'actual_time': case_end,
            'relative_time': duration * 3600,  # full duration in seconds
            'relative_ratio': 1.0,
            'logical_time': global_idx,
            'logical_relative': 2
        })
        global_idx += 1

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    return df


@pytest.fixture
def df_with_resource_shifts() -> pd.DataFrame:
    """
    DataFrame with intentional resource shift patterns for testing.

    Contains 3 resources working in distinct time shifts:
    - R1: Morning shift (8:00-12:00)
    - R2: Afternoon shift (12:00-16:00)
    - R3: Evening shift (16:00-20:00)
    """
    base_time = datetime(2024, 1, 15, 8, 0, 0)
    events = []
    global_idx = 0

    shift_configs = [
        ('R1', 0, 4),   # Morning: 8:00-12:00
        ('R2', 4, 4),   # Afternoon: 12:00-16:00
        ('R3', 8, 4),   # Evening: 16:00-20:00
    ]

    for resource, start_offset, duration in shift_configs:
        shift_start = base_time + timedelta(hours=start_offset)

        # Add 20 events spread across the shift
        for i in range(20):
            event_time = shift_start + \
                timedelta(minutes=i * (duration * 60 / 20))
            events.append({
                'case_id': f'C{global_idx}',
                'event_index': 0,
                'activity': 'Process',
                'resource': resource,
                'actual_time': event_time,
                'relative_time': 0.0,
                'relative_ratio': 0.0,
                'logical_time': global_idx,
                'logical_relative': 0
            })
            global_idx += 1

    df = pd.DataFrame(events)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    return df
    
# =============================================================================
# PREPROCESSOR FIXTURE
# =============================================================================


@pytest.fixture
def preprocessor():
    """Fresh DataPreprocessor instance."""
    from core.data_processing.preprocessor import DataPreprocessor
    return DataPreprocessor()


# =============================================================================
# PLOTLY FIGURE FIXTURES
# =============================================================================

@pytest.fixture
def empty_figure() -> go.Figure:
    """Empty Plotly figure for visualization tests."""
    return go.Figure()


@pytest.fixture
def sample_dotted_chart(sample_event_log) -> go.Figure:
    """
    Sample dotted chart with data points.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_event_log['actual_time'],
        y=sample_event_log['case_id'],
        mode='markers',
        marker=dict(size=8),
        text=sample_event_log['activity'],
        hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Case: %{y}<extra></extra>'
    ))
    fig.update_layout(
        title='Sample Dotted Chart',
        xaxis_title='Actual Time',
        yaxis_title='Case ID'
    )
    return fig


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def all_time_columns() -> list:
    """List of all time representation columns."""
    return ['actual_time', 'relative_time', 'relative_ratio', 'logical_time', 'logical_relative']


@pytest.fixture
def all_y_columns() -> list:
    """List of all Y-axis columns."""
    return ['case_id', 'activity', 'resource', 'event_index', 'variant']


@pytest.fixture
def event_log_required_columns() -> list:
    """Required columns for a valid event log DataFrame."""
    return ['case_id', 'event_index', 'activity', 'actual_time',
            'relative_time', 'relative_ratio', 'logical_time', 'logical_relative']


# =============================================================================
# ASSERTION HELPER FIXTURES
# =============================================================================

@pytest.fixture
def assert_dataframe_valid():
    """
    Factory fixture for DataFrame validation.

    Usage:
        def test_something(assert_dataframe_valid, sample_event_log):
            assert_dataframe_valid(sample_event_log, required_columns=['case_id'])
    """
    def _assert_valid(df: pd.DataFrame, required_columns: list = None, min_rows: int = 0):
        assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
        assert len(
            df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"

        if required_columns:
            missing = set(required_columns) - set(df.columns)
            assert not missing, f"Missing columns: {missing}"

    return _assert_valid


@pytest.fixture
def assert_figure_valid():
    """
    Factory fixture for Plotly figure validation.

    Usage:
        def test_viz(assert_figure_valid, some_figure):
            assert_figure_valid(some_figure, min_traces=1)
    """
    def _assert_valid(fig: go.Figure, min_traces: int = 0, has_shapes: bool = False):
        assert isinstance(fig, go.Figure), "Result should be a Plotly Figure"
        assert len(
            fig.data) >= min_traces, f"Expected at least {min_traces} traces"

        if has_shapes:
            shapes = fig.layout.shapes or []
            assert len(shapes) > 0, "Figure should have shapes"

    return _assert_valid


@pytest.fixture
def assert_pattern_detected():
    """
    Factory fixture for pattern detection validation.

    Usage:
        def test_cluster(assert_pattern_detected, cluster_detector):
            assert_pattern_detected(cluster_detector)
    """
    def _assert_detected(detector, has_detected: bool = True):
        if has_detected:
            assert detector.detected is not None, "Pattern should be detected"
        else:
            assert detector.detected is None, "Pattern should NOT be detected"

    return _assert_detected
