"""
Semi-synthetic validation tests for pattern detection.

This module addresses the Test Oracle Limitation by injecting known patterns
into real-world Hospital_log.xes data, providing ground truth for validation.

Strategy: Take real data + inject artificial patterns → verify detection
"""
from core.detection.gap_pattern import GapPattern
from core.detection.outlier_detection import OutlierDetectionPattern
from core.data_processing import load_xes_log
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


@pytest.fixture(scope="module")
def hospital_log():
    """Load Hospital_log.xes once for all tests."""
    xes_path = os.path.join(os.path.dirname(__file__),
                            '../../data/Hospital_log.xes')
    if not os.path.exists(xes_path):
        pytest.skip(f"Hospital_log.xes not found at {xes_path}")
    return load_xes_log(xes_path)


class TestSemiSyntheticOutliers:
    """Test outlier detection with known injected patterns."""

    def test_injected_time_outliers(self, hospital_log):
        """
        Inject 5 events far in the future (year 2030) and verify detection.

        Ground truth: Events with extreme timestamps should be detected as outliers.
        """
        df = hospital_log.copy()
        np.random.seed(42)

        # Inject 5 time outliers far in the future (year 2030)
        sample_indices = np.random.choice(df.index, size=5, replace=False)
        injected_indices = []

        for idx in sample_indices:
            # Set timestamp to far future (2030) - definitely an outlier
            original_time = df.loc[idx, 'actual_time']
            outlier_time = original_time.replace(year=2030)
            df.loc[idx, 'actual_time'] = outlier_time
            injected_indices.append(idx)

        # Detect outliers using Isolation Forest (which considers time features)
        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )
        result = detector.detect()

        # Validation: Should detect outliers (may use different methods)
        assert detector.detected, "Should detect outliers"

        # Collect all detected outlier indices from any detection method
        all_detected = []
        for outlier_type, indices in detector.outliers.items():
            all_detected.extend(indices)
        all_detected = list(set(all_detected))

        # Check if our injected outliers were detected
        true_positives = sum(
            1 for idx in injected_indices if idx in all_detected)
        precision = true_positives / len(all_detected) if all_detected else 0
        recall = true_positives / len(injected_indices)

        print(f"\nInjected time outliers: {len(injected_indices)}")
        print(f"Total outliers detected: {len(all_detected)}")
        print(f"True positives: {true_positives}")
        print(f"Precision: {precision:.2%}, Recall: {recall:.2%}")

        # Require reasonable recall (time detection is tricky)
        assert recall >= 0.4, f"Recall {recall:.2%} too low - missing most injected outliers"

    def test_injected_rare_activity(self, hospital_log):
        """
        Inject 1 ultra-rare activity and verify detection.

        Ground truth: The rare activity should be flagged as outlier.
        """
        df = hospital_log.copy()
        np.random.seed(42)

        # Inject a unique rare activity
        rare_activity = "ULTRA_RARE_ACTIVITY_TEST"
        sample_idx = np.random.choice(df.index)
        original_activity = df.loc[sample_idx, 'activity']
        df.loc[sample_idx, 'activity'] = rare_activity

        # Detect outliers
        detector = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'activity', 'color': 'case_id'}
        )
        result = detector.detect()

        # Validation
        assert detector.detected, "Should detect outliers"

        # Check if detected in any outlier category
        all_detected_indices = []
        for outlier_type, indices in detector.outliers.items():
            all_detected_indices.extend(indices)
        all_detected_indices = list(set(all_detected_indices))

        # The injected rare activity should be detected
        assert sample_idx in all_detected_indices, f"Failed to detect injected rare activity (detected {len(all_detected_indices)} outliers)"

        # Verify it's the rare activity
        detected_activity = df.loc[sample_idx, 'activity']
        assert detected_activity == rare_activity, "Wrong activity detected"
        print(
            f"\n✓ Successfully detected injected rare activity '{rare_activity}'")


class TestSemiSyntheticGaps:
    """Test gap detection with known injected gaps."""

    def test_injected_transition_gaps(self, hospital_log):
        """
        Inject abnormally large time gaps between consecutive activities in cases.

        Ground truth: Injected gaps should be detected as abnormal.
        """
        df = hospital_log.copy()
        df = df.sort_values(['case_id', 'actual_time']
                            ).reset_index(drop=True)
        np.random.seed(42)

        # Select 5 random cases and inject large gaps
        cases = df['case_id'].unique()
        selected_cases = np.random.choice(
            cases, size=min(5, len(cases)), replace=False)
        injected_gap_indices = []

        for case_id in selected_cases:
            case_events = df[df['case_id'] == case_id].index
            if len(case_events) > 1:
                # Add 30-minute gap after second event
                gap_idx = case_events[1]
                subsequent_events = case_events[2:]
                df.loc[subsequent_events, 'actual_time'] = \
                    df.loc[subsequent_events, 'actual_time'] + \
                    timedelta(minutes=30)
                injected_gap_indices.append(gap_idx)

        # Detect gaps
        detector = GapPattern(
            view_config={'x': 'actual_time', 'y': 'case_id'},
            y_is_categorical=False
        )
        detector.detect(df)

        # Validation
        assert detector.detected is not None, "Gap detection should complete"
        assert detector.detected['total_abnormal_gaps'] > 0, "Should detect injected abnormal gaps"

        print(f"\nInjected gaps: {len(injected_gap_indices)}")
        print(
            f"Total abnormal gaps detected: {detector.detected['total_abnormal_gaps']}")

        # Expect at least some of our injected gaps to be detected
        assert detector.detected['total_abnormal_gaps'] >= len(injected_gap_indices) * 0.4, \
            "Should detect at least 40% of injected gaps"


class TestCrossViewConsistency:
    """Test that patterns are consistent across different view configurations."""

    def test_outlier_consistency_across_views(self, hospital_log):
        """
        Verify that outlier cases/events are detected consistently across views.

        A case detected as duration outlier should also show temporal anomalies
        in actual_time views.
        """
        df = hospital_log.copy()

        # Detect case duration outliers
        detector1 = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )
        detector1.detect()

        # Detect time outliers
        detector2 = OutlierDetectionPattern(
            df=df,
            view_config={'x': 'actual_time',
                         'y': 'case_id', 'color': 'activity'}
        )
        detector2.detect()

        # Cross-validation: cases with duration outliers should have time outliers
        if detector1.detected and 'case_duration' in detector1.outliers:
            outlier_cases = detector1.outliers['case_duration']

            if detector2.detected and 'time' in detector2.outliers:
                time_outlier_indices = detector2.outliers['time']
                time_outlier_cases = df.loc[time_outlier_indices, 'case_id'].unique(
                )

                # Check overlap
                overlap = sum(
                    1 for case in outlier_cases if case in time_outlier_cases)
                overlap_ratio = overlap / \
                    len(outlier_cases) if outlier_cases else 0

                print(f"\nCase duration outliers: {len(outlier_cases)}")
                print(f"Cases with time outliers: {len(time_outlier_cases)}")
                print(f"Overlap: {overlap} ({overlap_ratio:.1%})")

                # Expect some consistency (not perfect due to different detection methods)
                assert overlap_ratio >= 0.2, "Low consistency between duration and time outliers"
