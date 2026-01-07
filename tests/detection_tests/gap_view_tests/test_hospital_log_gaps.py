"""
Pytest tests for gap detection on the real Hospital_log.xes dataset.
Tests various X-axis and Y-axis combinations.
"""

from core.detection.gap_pattern import GapPattern
from core.data_processing import load_xes_log
import pytest
import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


@pytest.fixture(scope="module")
def hospital_log():
    """Load Hospital_log.xes once for all tests."""
    xes_path = os.path.join(os.path.dirname(__file__),
                            '../../../data/Hospital_log.xes')
    if not os.path.exists(xes_path):
        pytest.skip(f"Hospital_log.xes not found at {xes_path}")
    return load_xes_log(xes_path)


def test_hospital_actual_time_case_id(hospital_log):
    """Test gap detection with actual_time × case_id (time-based, numeric Y)."""
    df = hospital_log.copy()

    view_config = {'x': 'actual_time', 'y': 'case_id'}
    y_is_categorical = df['case_id'].nunique() <= 60  # False (1143 unique)

    detector = GapPattern(
        view_config=view_config,
        min_gap_area=None,  # Use default (0.001 for time-based)
        y_is_categorical=y_is_categorical
    )
    detector.detect(df)

    # Should detect gaps (time-based with default threshold)
    assert detector.detected is not None, "Should detect gaps for actual_time × case_id"
    print(
        f"\nactual_time × case_id: {detector.detected['total_gaps']} gaps detected")


def test_hospital_actual_time_resource(hospital_log):
    """Test gap detection with actual_time × resource (time-based, categorical Y)."""
    df = hospital_log.copy()

    view_config = {'x': 'actual_time', 'y': 'resource'}
    y_is_categorical = df['resource'].nunique() <= 60  # True (42 unique)

    detector = GapPattern(
        view_config=view_config,
        min_gap_x_width=0.001,  # Use lower threshold for time-based
        y_is_categorical=y_is_categorical
    )
    detector.detect(df)

    # Validation: gaps should have abnormal characteristics
    if detector.detected and detector.detected['total_abnormal_gaps'] > 0:
        print(
            f"\nactual_time × resource: {detector.detected['total_gaps']} gaps, {detector.detected['total_abnormal_gaps']} abnormal")

        # Statistical validation: abnormal gaps should be significantly larger than normal
        if 'gap_severity_scores' in detector.detected:
            severity_scores = [
                s for s in detector.detected['gap_severity_scores'] if s > 1.0]
            if severity_scores:
                avg_severity = sum(severity_scores) / len(severity_scores)
                assert avg_severity > 1.5, f"Average gap severity {avg_severity:.2f} not significant enough"
                print(
                    f"Average abnormal gap severity: {avg_severity:.2f}x threshold (validates abnormality)")

        assert detector.detected["total_abnormal_gaps"] >= 10, "Should detect multiple abnormal gaps in real data"
    else:
        print("\nactual_time × resource: No abnormal gaps detected")


def test_hospital_relative_time_activity(hospital_log):
    """Test gap detection with relative_time × activity (time-based, numeric Y)."""
    df = hospital_log.copy()

    view_config = {'x': 'relative_time', 'y': 'activity'}
    y_is_categorical = df['activity'].nunique() <= 60  # False (624 unique)

    detector = GapPattern(
        view_config=view_config,
        min_gap_area=None,  # Use default (0.001 for time-based)
        y_is_categorical=y_is_categorical
    )
    detector.detect(df)

    # Should detect gaps (time-based with default threshold)
    assert detector.detected is not None, "Should detect gaps for relative_time × activity"
    print(
        f"\nrelative_time × activity: {detector.detected['total_gaps']} gaps detected")

    # Sanity check: gaps should be reasonable percentage of transitions
    if 'total_transitions' in detector.detected:
        total_transitions = detector.detected['total_transitions']
        total_gaps = detector.detected['total_gaps']
        gap_percentage = (total_gaps / total_transitions *
                          100) if total_transitions > 0 else 0
        assert gap_percentage <= 50, f"Gap percentage {gap_percentage:.1f}% too high - likely over-detection"
        print(
            f"Gap percentage: {gap_percentage:.1f}% of transitions (reasonable)")

    assert detector.detected['total_gaps'] >= 10, "Expected at least 10 gaps in real process data"


def test_hospital_relative_ratio_resource(hospital_log):
    """Test gap detection with relative_ratio × resource (time-based, categorical Y)."""
    df = hospital_log.copy()

    view_config = {'x': 'relative_ratio', 'y': 'resource'}
    y_is_categorical = df['resource'].nunique() <= 60  # True (42 unique)

    detector = GapPattern(
        view_config=view_config,
        min_gap_x_width=0.001,  # Use lower threshold for time-based
        y_is_categorical=y_is_categorical
    )
    detector.detect(df)

    # Detection should work (may or may not find gaps depending on data)
    print(
        f"\nrelative_ratio × resource: {detector.detected['total_gaps'] if detector.detected else 0} gaps detected")
    assert detector.detected['total_gaps'] >= 100, "Expected at least 1 gap"


def test_hospital_logical_time_case_id(hospital_log):
    """Test gap detection with logical_time × case_id (time-based, numeric Y)."""
    df = hospital_log.copy()

    view_config = {'x': 'logical_time', 'y': 'case_id'}
    y_is_categorical = df['case_id'].nunique() <= 60  # False (1143 unique)

    detector = GapPattern(
        view_config=view_config,
        # Use very low threshold (logical_time has many unique values)
        min_gap_area=0.0001,
        y_is_categorical=y_is_categorical
    )
    detector.detect(df)

    # Detection should work (may or may not find gaps depending on data)
    print(
        f"\nlogical_time × case_id: {detector.detected['total_gaps'] if detector.detected else 0} gaps detected")
    assert detector.detected['total_gaps'] >= 100, "Expected at least 1 gap"


def test_hospital_relative_ratio_activity_2d(hospital_log):
    """Test 2D gap detection with relative_ratio × activity (numeric X, numeric Y)."""
    df = hospital_log.copy()

    view_config = {'x': 'relative_ratio', 'y': 'activity'}
    y_is_categorical = df['activity'].nunique() <= 60  # False (624 unique)

    # For numeric X + numeric Y, we need to check if it's time-based
    # relative_ratio is time-based, so it will use 1D detection
    # But let's test with a small threshold to see if 1D detection would work
    detector = GapPattern(
        view_config=view_config,
        min_gap_area=0.001,  # Use lower threshold for time-based
        y_is_categorical=y_is_categorical
    )
    detector.detect(df)

    # This should use 1D detection because relative_ratio is time-based
    # Detection should work (may or may not find gaps depending on data)
    print(
        f"\nrelative_ratio × activity (1D time-based): {detector.detected['total_gaps'] if detector.detected else 0} gaps detected")
    assert detector.detected['total_gaps'] >= 100, "Expected at least 1 gap"
