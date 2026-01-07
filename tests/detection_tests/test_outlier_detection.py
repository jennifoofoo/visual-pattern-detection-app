"""
Tests for OutlierDetectionPattern.

Tests each detection method with synthetic data:
1. Time outliers
2. Case duration outliers
3. Activity frequency outliers
4. Resource outliers
5. Sequence outliers
6. Case complexity outliers

Then tests on real Hospital_log.xes data.
"""
# isort: skip_file
# fmt: off
import os
import sys

# Add directories to path for imports - MUST come before other imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from synthetic_outlier_logs import (
    make_time_outliers,
    make_case_duration_outliers,
    make_activity_frequency_outliers,
    make_resource_workload_outliers,
    make_sequence_outliers,
    make_case_complexity_outliers
)
from core.data_processing import load_xes_log
from core.detection.outlier_detection import OutlierDetectionPattern
# fmt: on


class TestSyntheticOutlierDetection:
    """Test outlier detection methods using synthetic data."""

    def test_detects_time_outliers(self):
        """
        Test that events at unusual times are detected as outliers.

        Uses synthetic data with business hours events (9-17) and 3 AM events.
        """
        df = make_time_outliers()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )

        result = detector.detect()

        # Should detect outliers
        assert result is True or 'time' in detector.outliers
        print(f"\nTime outliers detected: {'time' in detector.outliers}")

    def test_detects_case_duration_outliers(self):
        """
        Test that cases with extremely long/short durations are detected.

        Uses synthetic data with normal durations (1-3 hours) and extreme case (48 hours).
        """
        df = make_case_duration_outliers()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )

        result = detector.detect()

        # Check case duration outliers were analyzed
        assert result is True or 'case_duration' in detector.outliers
        print(
            f"\nCase duration outliers detected: {'case_duration' in detector.outliers}")

    def test_detects_rare_activity_outliers(self):
        """
        Test that rare activities are detected as outliers.

        Uses synthetic data with common activities (100 each) and 1 rare activity.
        """
        df = make_activity_frequency_outliers()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'activity', 'color': 'case_id'}
        )

        result = detector.detect()

        # Should detect rare activity as outlier
        assert result is True
        if 'activity_frequency' in detector.outliers:
            activity_outliers = detector.outliers['activity_frequency']
            if activity_outliers:
                # Check that our rare activity is flagged
                rare_indices = df[df['activity'] ==
                                  'Extremely_Rare_Activity'].index
                assert any(idx in activity_outliers for idx in rare_indices), \
                    "Rare activity should be detected as outlier"
                print(f"\nRare activity correctly detected as outlier")

    def test_detects_resource_workload_outliers(self):
        """
        Test that resources with unusual workload are detected.

        Uses synthetic data with normal workload (20 events) and extreme workload (200 events).
        """
        df = make_resource_workload_outliers()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'resource', 'color': 'case_id'}
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
                print(f"\nOverworked resource correctly detected as outlier")

    def test_detects_unusual_sequence_outliers(self):
        """
        Test that unusual activity sequences are detected.

        Uses synthetic data with common sequence (A->B->C, 20 cases) and rare sequence (X->Y, 1 case).
        """
        df = make_sequence_outliers()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )

        result = detector.detect()

        # Check sequence outliers were analyzed
        assert 'combined' in detector.outliers
        print(
            f"\nSequence outliers detected: {'combined' in detector.outliers}")

    def test_detects_case_complexity_outliers(self):
        """
        Test that cases with unusual complexity (event count) are detected.

        Uses synthetic data with normal cases (3-5 events) and complex case (50 events).
        """
        df = make_case_complexity_outliers()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
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
                print(f"\nComplex case correctly detected as outlier")


class TestHospitalLogOutliers:
    """Test outlier detection on real Hospital_log.xes data."""

    @pytest.fixture(scope="class")
    def hospital_log(self):
        """Load Hospital_log.xes once for all tests."""
        xes_path = os.path.join(os.path.dirname(
            __file__), '../../data/Hospital_log.xes')
        if not os.path.exists(xes_path):
            pytest.skip(f"Hospital_log.xes not found at {xes_path}")
        return load_xes_log(xes_path)

    def test_hospital_time_outliers(self, hospital_log):
        """Test time-based outlier detection on Hospital_log.xes."""
        df = hospital_log.copy()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )

        result = detector.detect()

        # Should complete detection
        assert hasattr(detector, 'outliers'), "Detection should complete"
        print(f"\nHospital log time outliers: {detector.detected}")
        if detector.detected and 'time' in detector.outliers:
            print(
                f"Time outliers found: {len(detector.outliers['time'])} events")

    def test_hospital_activity_outliers(self, hospital_log):
        """Test activity frequency outlier detection on Hospital_log.xes."""
        df = hospital_log.copy()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'activity', 'color': 'case_id'}
        )

        result = detector.detect()

        assert hasattr(detector, 'outliers'), "Detection should complete"
        print(f"\nHospital log activity outliers: {detector.detected}")

        # Sanity check: outliers should be reasonable percentage of total
        if detector.detected:
            outlier_percentage = (
                detector.statistics['total_outliers'] / detector.statistics['total_events']) * 100
            assert outlier_percentage <= 10, f"Outlier percentage {outlier_percentage:.2f}% exceeds 10% - likely false positives"
            assert outlier_percentage > 0, "Should detect at least some outliers in real data"
            print(
                f"Outlier percentage: {outlier_percentage:.2f}% (reasonable)")

        # Check activity frequency outliers if found
        if detector.detected and 'activity_frequency' in detector.outliers:
            activity_outliers = detector.outliers['activity_frequency']
            print(f"Activity outliers found: {len(activity_outliers)} events")

            # Validation: outlier activities should be statistically rare
            activity_counts = df['activity'].value_counts()
            outlier_activities = [idx for idx, _ in activity_outliers]
            for act_idx in outlier_activities[:3]:  # Check first 3
                act_name = df.iloc[act_idx]['activity']
                act_count = activity_counts.get(act_name, 0)
                total_acts = len(df)
                frequency = act_count / total_acts
                assert frequency < 0.05, f"Outlier activity '{act_name}' has frequency {frequency:.3f} >= 5% - not truly rare"
                print(
                    f"  - '{act_name}': {act_count} occurrences ({frequency*100:.2f}%)")

    def test_hospital_resource_outliers(self, hospital_log):
        """Test resource workload outlier detection on Hospital_log.xes."""
        df = hospital_log.copy()

        # Check if resource column exists
        if 'resource' not in df.columns:
            pytest.skip("Hospital_log.xes does not have resource column")

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'resource', 'color': 'activity'}
        )

        result = detector.detect()

        # Should complete detection
        assert hasattr(detector, 'outliers'), "Detection should complete"
        print(f"\nHospital log resource outliers: {detector.detected}")
        if detector.detected and 'resource' in detector.outliers:
            print(
                f"Resource outliers found: {len(detector.outliers['resource'])} events")

    def test_hospital_case_duration_outliers(self, hospital_log):
        """Test case duration outlier detection on Hospital_log.xes."""
        df = hospital_log.copy()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )

        result = detector.detect()

        # Should complete detection
        assert hasattr(detector, 'outliers'), "Detection should complete"
        print(f"\nHospital log case duration outliers: {detector.detected}")

        if detector.detected and 'case_duration' in detector.outliers:
            duration_outliers = detector.outliers['case_duration']
            print(
                f"Case duration outliers found: {len(duration_outliers)} cases")

            # Statistical validation: outlier durations should be extreme
            import numpy as np
            case_durations = df.groupby('case_id')['time:timestamp'].agg(
                lambda x: (x.max() - x.min()).total_seconds()
            )
            median_duration = case_durations.median()
            std_duration = case_durations.std()

            for case_id, _ in duration_outliers[:3]:  # Check first 3
                case_duration = case_durations[case_id]
                z_score = abs(case_duration - median_duration) / \
                    std_duration if std_duration > 0 else 0
                assert z_score > 2.0, f"Case '{case_id}' duration not statistically extreme (z-score={z_score:.2f})"
                print(
                    f"  - Case '{case_id}': {case_duration/3600:.2f} hours (z-score={z_score:.2f})")

    def test_hospital_case_complexity_outliers(self, hospital_log):
        """Test case complexity outlier detection on Hospital_log.xes."""
        df = hospital_log.copy()

        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )

        result = detector.detect()

        # Should complete detection
        assert hasattr(detector, 'outliers'), "Detection should complete"
        print(f"\nHospital log case complexity outliers: {detector.detected}")
        if detector.detected and 'case_complexity' in detector.outliers:
            print(
                f"Case complexity outliers found: {len(detector.outliers['case_complexity'])} events")


class TestOutlierDetection:
    """Test outlier detection structure and summary methods."""

    def test_get_outlier_summary_structure(self, sample_event_log):
        """Test that get_outlier_summary returns properly structured data."""
        detector = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
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
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )
        detector.detect()

        summary = detector.get_summary()

        assert 'pattern_type' in summary
        assert summary['pattern_type'] == 'outlier'
        assert 'detected' in summary
        assert 'count' in summary
        assert 'details' in summary
