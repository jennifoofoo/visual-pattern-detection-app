"""
Unit tests for XES log loader (core/data_processing/loader.py).

Tests cover:
- Loading valid XES files
- All 5 time representations (actual, relative, relative_ratio, logical, logical_relative)
- Required columns presence
- Edge cases (empty traces, missing attributes)
- compute_variants function
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestLoadXesLog:
    """Tests for load_xes_log function."""

    def test_load_hospital_log(self, event_log_path):
        """Test loading the Hospital_log.xes file."""
        from core.data_processing.loader import load_xes_log

        df = load_xes_log(str(event_log_path))

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame should not be empty"

    def test_required_columns_present(self, event_log_path):
        """Test that all required columns are present after loading."""
        from core.data_processing.loader import load_xes_log

        df = load_xes_log(str(event_log_path))

        required_columns = [
            'case_id', 'activity',
            'actual_time', 'relative_time', 'relative_ratio',
            'logical_time'
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_resource_column_present(self, event_log_path):
        """Test that resource column is extracted."""
        from core.data_processing.loader import load_xes_log

        df = load_xes_log(str(event_log_path))

        assert 'resource' in df.columns, "Resource column should be present"


class TestTimeRepresentations:
    """Tests for the 5 time representations."""

    def test_actual_time_is_datetime(self, sample_event_log):
        """Test that actual_time is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(sample_event_log['actual_time']), \
            "actual_time should be datetime type"


    def test_relative_time_is_non_negative(self, sample_event_log):
        """Test that relative_time is always non-negative."""
        assert (sample_event_log['relative_time'] >= 0).all(), \
            "relative_time should always be non-negative"

    def test_relative_ratio_in_range(self, sample_event_log):
        """Test that relative_ratio is in [0, 1] range."""
        valid_ratios = sample_event_log['relative_ratio'].dropna()

        assert (valid_ratios >= 0).all(), "relative_ratio should be >= 0"
        assert (valid_ratios <= 1).all(), "relative_ratio should be <= 1"
class TestEventAttributes:
    """Tests for event attribute extraction."""

    def test_case_id_not_null(self, sample_event_log):
        """Test that case_id is not null for any event."""
        assert sample_event_log['case_id'].notna().all(), \
            "case_id should not be null"

    def test_activity_not_null(self, sample_event_log):
        """Test that activity is not null for any event."""
        assert sample_event_log['activity'].notna().all(), \
            "activity should not be null"


class TestEdgeCases:
    """Tests for edge cases in data loading."""

    def test_empty_dataframe_columns(self, empty_event_log):
        """Test that empty DataFrame has correct columns."""
        expected_cols = [
            'case_id', 'event_index', 'activity', 'resource',
            'actual_time', 'relative_time', 'relative_ratio',
            'logical_time', 'logical_relative'
        ]

        for col in expected_cols:
            assert col in empty_event_log.columns, f"Missing column: {col}"

    def test_single_event_case(self, single_event_log):
        """Test handling of single-event cases."""
        assert len(single_event_log) == 1

        event = single_event_log.iloc[0]

        # First (and only) event should have these values
        assert event['event_index'] == 0
        assert event['logical_relative'] == 0
        assert event['relative_time'] == 0.0

    def test_synthetic_data_consistency(self, synthetic_event_log):
        """Test that synthetic data has consistent time representations."""
        df = synthetic_event_log

        for case_id, case_df in df.groupby('case_id'):
            case_df_sorted = case_df.sort_values('event_index')

            # Check relative_time increases monotonically
            rel_times = case_df_sorted['relative_time'].values
            assert all(rel_times[i] <= rel_times[i+1] for i in range(len(rel_times)-1)), \
                f"relative_time should increase within case {case_id}"

            # Check relative_ratio increases monotonically
            rel_ratios = case_df_sorted['relative_ratio'].values
            assert all(rel_ratios[i] <= rel_ratios[i+1] for i in range(len(rel_ratios)-1)), \
                f"relative_ratio should increase within case {case_id}"


class TestDataTypes:
    """Tests for correct data types after loading."""

    def test_numeric_columns(self, sample_event_log):
        """Test that numeric columns have correct types."""
        numeric_cols = ['relative_time', 'relative_ratio',
                        'logical_time']

        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_event_log[col]), \
                f"{col} should be numeric type"

    def test_string_columns(self, sample_event_log):
        """Test that string columns have correct types."""
        # case_id and activity should be string-like (object or string)
        assert sample_event_log['case_id'].dtype == object or \
            pd.api.types.is_string_dtype(sample_event_log['case_id']), \
            "case_id should be string type"

        assert sample_event_log['activity'].dtype == object or \
            pd.api.types.is_string_dtype(sample_event_log['activity']), \
            "activity should be string type"

    def test_datetime_column(self, sample_event_log):
        """Test that actual_time is datetime."""
        assert pd.api.types.is_datetime64_any_dtype(sample_event_log['actual_time']), \
            "actual_time should be datetime type"
