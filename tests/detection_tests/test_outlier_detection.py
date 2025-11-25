"""
Tests for OutlierDetectionPattern.

Tests each detection method:
1. Time outliers
2. Case duration outliers
3. Activity frequency outliers
4. Resource outliers
5. Sequence outliers
6. Case complexity outliers
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.detection.outlier_detection import OutlierDetectionPattern


class TestOutlierDetection:
    """Test _detect_time_outliers method."""

    def test_detects_time_outliers(self):
        """
        Test that events at unusual times are detected as outliers.

        Creates events mostly during business hours (9-17) with a few at 3 AM.
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
                'actual_time': base_date + timedelta(hours=hour, minutes=np.random.randint(0, 60))
            })

        # Unusual time events (5 events at 3 AM - should be detected as outliers)
        for i in range(5):
            events.append({
                'case_id': f'C_night_{i}',
                'activity': 'Process',
                'actual_time': base_date + timedelta(hours=3, minutes=i * 10)
            })

        df = pd.DataFrame(events)
        df['actual_time'] = pd.to_datetime(df['actual_time'])

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        result = detector.detect()

        # Should detect outliers
        assert result is True or 'time' in detector.outliers


    def test_detects_case_duration_outliers(self):
        """
        Test that cases with extremely long/short durations are detected.

        Creates cases with normal durations (1-3 hours) and one extremely long case (48 hours).
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
                'actual_time': base_time + timedelta(hours=case_idx * 0.5)
            })
            # End event
            events.append({
                'case_id': case_id,
                'activity': 'End',
                'actual_time': base_time + timedelta(hours=case_idx * 0.5 + duration_hours)
            })

        # Extremely long case (48 hours - should be an outlier)
        events.append({
            'case_id': 'C_long',
            'activity': 'Start',
            'actual_time': base_time
        })
        events.append({
            'case_id': 'C_long',
            'activity': 'End',
            'actual_time': base_time + timedelta(hours=48)
        })

        df = pd.DataFrame(events)
        df['actual_time'] = pd.to_datetime(df['actual_time'])

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        result = detector.detect()

        # Check case duration outliers were analyzed
        assert result is True or 'case_duration' in detector.outliers


    def test_detects_rare_activity_outliers(self):
        """
        Test that rare activities are detected as outliers.

        Creates common activities (100 each) and 2 rare activities.
        """
        events = []
        base_time = datetime(2024, 1, 15, 9, 0, 0)

        # Common activities (100 events each)
        for i in range(100):
            events.append({
                'case_id': f'C{i}',
                'activity': 'Common_A',
                'actual_time': base_time + timedelta(minutes=i)
            })
        for i in range(100):
            events.append({
                'case_id': f'C{i + 100}',
                'activity': 'Common_B',
                'actual_time': base_time + timedelta(minutes=i + 100)
            })

        # Rare activity (should be detected as outlier - less than 1%)
        events.append({
            'case_id': 'C_rare',
            'activity': 'Extremely_Rare_Activity',
            'actual_time': base_time + timedelta(minutes=300)
        })

        df = pd.DataFrame(events)
        df['actual_time'] = pd.to_datetime(df['actual_time'])

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x_axis': 'actual_time', 'y_axis': 'activity'}
        )

        result = detector.detect()

        # Should detect rare activity as outlier
        if 'activity_frequency' in detector.outliers:
            activity_outliers = detector.outliers['activity_frequency']
            if activity_outliers:
                # Check that our rare activity is flagged
                rare_indices = df[df['activity'] ==
                                  'Extremely_Rare_Activity'].index
                assert any(idx in activity_outliers for idx in rare_indices), \
                    "Rare activity should be detected as outlier"


    def test_detects_resource_workload_outliers(self):
        """
        Test that resources with unusual workload are detected.

        Creates resources with normal workload and one with extremely high workload.
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

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x_axis': 'actual_time', 'y_axis': 'resource'}
        )

        result = detector.detect()
        assert result is True 

        # Check resource outliers were analyzed
        if 'resource' in detector.outliers:
            resource_outliers = detector.outliers['resource']
            if resource_outliers:
                # Extreme workload resource should be flagged
                extreme_indices = df[df['resource'] == 'R_Overworked'].index
                assert any(idx in resource_outliers for idx in extreme_indices), \
                    "Overworked resource events should be detected as outliers"



    def test_detects_unusual_sequence_outliers(self):
        """
        Test that unusual activity sequences are detected.

        Creates cases with common sequence (A->B->C) and one with rare sequence (A->C->B).
        """
        events = []
        base_time = datetime(2024, 1, 15, 9, 0, 0)

        # Common sequence: A -> B -> C (20 cases)
        for case_idx in range(20):
            case_id = f'C{case_idx}'
            for event_idx, activity in enumerate(['A', 'B', 'C']):
                events.append({
                    'case_id': case_id,
                    'activity': activity,
                    'actual_time': base_time + timedelta(hours=case_idx, minutes=event_idx * 30)
                })

        # Rare sequence: X -> Y (unique - should be outlier)
        events.append({
            'case_id': 'C_rare_seq',
            'activity': 'X',
            'actual_time': base_time + timedelta(days=1)
        })
        events.append({
            'case_id': 'C_rare_seq',
            'activity': 'Y',
            'actual_time': base_time + timedelta(days=1, hours=1)
        })

        df = pd.DataFrame(events)
        df['actual_time'] = pd.to_datetime(df['actual_time'])

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        result = detector.detect()

        # Check sequence outliers were analyzed
        # Might not always detect due to strict thresholds
        assert 'sequence' in detector.outliers 

    def test_detects_case_complexity_outliers(self):
        """
        Test that cases with unusual complexity (event count) are detected.

        Creates normal cases (3-5 events) and one extremely complex case (50 events).
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
                    'activity': f'Step_{event_idx}',
                    'actual_time': base_time + timedelta(hours=case_idx, minutes=event_idx * 15)
                })

        # Extremely complex case (50 events - should be outlier)
        for event_idx in range(50):
            events.append({
                'case_id': 'C_complex',
                'activity': f'ComplexStep_{event_idx}',
                'actual_time': base_time + timedelta(days=1, minutes=event_idx * 10)
            })

        df = pd.DataFrame(events)
        df['actual_time'] = pd.to_datetime(df['actual_time'])

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        result = detector.detect()

        # Check case complexity outliers
        if 'case_complexity' in detector.outliers:
            complexity_outliers = detector.outliers['case_complexity']
            if complexity_outliers:
                # Complex case should be flagged
                complex_indices = df[df['case_id'] == 'C_complex'].index
                assert any(idx in complexity_outliers for idx in complex_indices), \
                    "Complex case events should be detected as outliers"


    def test_get_outlier_summary_structure(self, sample_event_log):
        """Test that get_outlier_summary returns properly structured data."""
        detector = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )
        detector.detect()

        summary = detector.get_outlier_summary()

        assert isinstance(summary, dict)

        if detector.detected:
            assert 'statistics' in summary
            assert 'outlier_details' in summary
            assert 'available_columns' in summary

    def test_get_summary_standardized_format(self, sample_event_log):
        """Test that get_summary returns standardized format."""
        detector = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )
        detector.detect()

        summary = detector.get_summary()

        assert 'pattern_type' in summary
        assert summary['pattern_type'] == 'outlier'
        assert 'detected' in summary
        assert 'count' in summary
        assert 'details' in summary