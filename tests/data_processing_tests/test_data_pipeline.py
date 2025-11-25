"""
Integration tests for data processing pipeline (loader + preprocessor).

Tests the complete data flow:
1. Load XES file -> DataFrame with time representations
2. Preprocess DataFrame -> Encoded/scaled columns for pattern detection

These tests ensure data consistency across the pipeline and verify that
the preprocessor handles all column types produced by the loader.
"""

import pytest
import pandas as pd
import numpy as np
from core.data_processing.loader import load_xes_log
from core.data_processing.preprocessor import DataPreprocessor


class TestLoaderPreprocessorIntegration:
    """Integration tests for loader -> preprocessor pipeline."""


    def test_full_pipeline_actual_time_activity(self, event_log_path):
        """Test: Load XES -> Preprocess with actual_time x activity view."""
        # Load
        df = load_xes_log(str(event_log_path))
        assert len(df) > 0

        # Preprocess
        preprocessor = DataPreprocessor()
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(df, view_config)

        # Verify new columns created
        assert 'activity_code' in df_processed.columns
        assert 'actual_time_scaled' in df_processed.columns

        # Verify original columns preserved
        assert 'actual_time' in df_processed.columns
        assert 'activity' in df_processed.columns

        # Verify scaled values in [0, 1] range
        assert df_processed['actual_time_scaled'].min() >= 0
        assert df_processed['actual_time_scaled'].max() <= 1

    def test_full_pipeline_actual_time_case(self, event_log_path):
        """Test: Load XES -> Preprocess with actual_time x case_id view."""
        df = load_xes_log(str(event_log_path))

        preprocessor = DataPreprocessor()
        view_config = {'x': 'case_id', 'y': 'actual_time', 'view': 'case'}
        df_processed = preprocessor.process(df, view_config)

        assert 'case_id_code' in df_processed.columns
        assert 'actual_time_scaled' in df_processed.columns

    def test_full_pipeline_actual_time_resource(self, event_log_path):
        """Test: Load XES -> Preprocess with actual_time x resource view."""
        df = load_xes_log(str(event_log_path))

        # Skip if no resource data
        if df['resource'].isna().all():
            pytest.skip("No resource data in test file")

        preprocessor = DataPreprocessor()
        view_config = {'x': 'actual_time', 'y': 'resource', 'view': 'resource'}
        df_processed = preprocessor.process(df, view_config)

        assert 'resource_code' in df_processed.columns
        assert 'actual_time_scaled' in df_processed.columns
#TODO: check this test
    # def test_pipeline_with_sampled_data(self, sample_event_log):
    #     """Test pipeline with sampled data (faster, no file I/O)."""
    #     preprocessor = DataPreprocessor()

    #     # Test all 5 time representations
    #     time_cols = ['actual_time', 'relative_time', 'relative_ratio',
    #                  'logical_time', 'logical_relative']

    #     for time_col in time_cols:
    #         view_config = {'x': time_col, 'y': 'activity', 'view': 'time'}
    #         df_processed = preprocessor.process(
    #             sample_event_log.copy(), view_config)

    #         assert f'{time_col}_scaled' in df_processed.columns, \
    #             f"Missing scaled column for {time_col}"
    #         assert 'activity_code' in df_processed.columns


class TestAllTimeRepresentationsWithPreprocessor:
    """Test that all time representations from loader work with preprocessor."""

    def test_actual_time_preprocessing(self, sample_event_log, preprocessor):
        """Test preprocessing actual_time column."""
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert 'actual_time_scaled' in df_processed.columns
        assert df_processed['actual_time_scaled'].notna().all()

        # Datetime should be converted to numeric for scaling
        assert pd.api.types.is_numeric_dtype(
            df_processed['actual_time_scaled'])

    def test_relative_time_preprocessing(self, sample_event_log, preprocessor):
        """Test preprocessing relative_time column."""
        view_config = {'x': 'relative_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert 'relative_time_scaled' in df_processed.columns
        # relative_time is already numeric (seconds)
        assert df_processed['relative_time_scaled'].notna().all()

    def test_relative_ratio_preprocessing(self, sample_event_log, preprocessor):
        """Test preprocessing relative_ratio column."""
        view_config = {'x': 'relative_ratio', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert 'relative_ratio_scaled' in df_processed.columns
        # relative_ratio is already [0, 1], but MinMax should still work
        assert df_processed['relative_ratio_scaled'].notna().all()

    def test_logical_time_preprocessing(self, sample_event_log, preprocessor):
        """Test preprocessing logical_time column."""
        view_config = {'x': 'logical_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert 'logical_time_scaled' in df_processed.columns
        # logical_time is integer
        assert df_processed['logical_time_scaled'].notna().all()

    def test_logical_relative_preprocessing(self, sample_event_log, preprocessor):
        """Test preprocessing logical_relative column."""
        view_config = {'x': 'logical_relative', 'y': 'case_id', 'view': 'case'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert 'logical_relative_scaled' in df_processed.columns
        assert 'case_id_code' in df_processed.columns


class TestAllYAxisColumnsWithPreprocessor:
    """Test that all Y-axis column types from loader work with preprocessor."""

    def test_case_id_encoding(self, sample_event_log, preprocessor):
        """Test encoding case_id column."""
        view_config = {'x': 'case_id', 'y': 'actual_time', 'view': 'case'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert 'case_id_code' in df_processed.columns
        # Code should be integer
        assert pd.api.types.is_integer_dtype(df_processed['case_id_code'])
        # Each unique case_id should have unique code
        assert df_processed.groupby(
            'case_id')['case_id_code'].nunique().eq(1).all()

    def test_activity_encoding(self, sample_event_log, preprocessor):
        """Test encoding activity column."""
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert 'activity_code' in df_processed.columns
        assert pd.api.types.is_integer_dtype(df_processed['activity_code'])

    def test_resource_encoding(self, sample_event_log, preprocessor):
        """Test encoding resource column."""
        # Skip if no resource data
        if sample_event_log['resource'].isna().all():
            pytest.skip("No resource data")

        view_config = {'x': 'actual_time', 'y': 'resource', 'view': 'resource'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert 'resource_code' in df_processed.columns


class TestDataConsistencyAcrossPipeline:
    """Test data consistency after loader -> preprocessor pipeline."""

    def test_row_count_preserved(self, sample_event_log, preprocessor):
        """Test that preprocessing doesn't change row count."""
        original_count = len(sample_event_log)

        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert len(df_processed) == original_count

    def test_original_columns_preserved(self, sample_event_log, preprocessor):
        """Test that original columns are preserved after preprocessing."""
        original_columns = set(sample_event_log.columns)

        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        # All original columns should still exist
        assert original_columns.issubset(set(df_processed.columns))

    def test_original_values_unchanged(self, sample_event_log, preprocessor):
        """Test that original column values are not modified."""
        original_activities = sample_event_log['activity'].tolist()
        original_case_ids = sample_event_log['case_id'].tolist()

        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert df_processed['activity'].tolist() == original_activities
        assert df_processed['case_id'].tolist() == original_case_ids

    def test_index_preserved(self, sample_event_log, preprocessor):
        """Test that DataFrame index is preserved."""
        original_index = sample_event_log.index.tolist()

        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        assert df_processed.index.tolist() == original_index

    def test_encoding_consistency_across_cases(self, sample_event_log, preprocessor):
        """Test that same activity gets same code across all cases."""
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        # Group by activity and check code is unique per activity
        activity_codes = df_processed.groupby(
            'activity')['activity_code'].unique()
        for activity, codes in activity_codes.items():
            assert len(codes) == 1, \
                f"Activity '{activity}' has multiple codes: {codes}"

    def test_scaled_values_order_preserved(self, sample_event_log, preprocessor):
        """Test that scaling preserves relative ordering of values."""
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(sample_event_log, view_config)

        # Convert original timestamps to numeric for comparison
        original_numeric = sample_event_log['actual_time'].astype('int64')
        scaled = df_processed['actual_time_scaled']

        # For any two points, if original[i] < original[j], then scaled[i] < scaled[j]
        for i in range(min(10, len(df_processed) - 1)):
            for j in range(i + 1, min(10, len(df_processed))):
                if original_numeric.iloc[i] < original_numeric.iloc[j]:
                    assert scaled.iloc[i] < scaled.iloc[j], \
                        "Scaling should preserve relative ordering"
                elif original_numeric.iloc[i] > original_numeric.iloc[j]:
                    assert scaled.iloc[i] > scaled.iloc[j], \
                        "Scaling should preserve relative ordering"


class TestEdgeCasesInPipeline:
    """Test edge cases in the loader -> preprocessor pipeline."""

    def test_empty_dataframe(self, empty_event_log, preprocessor):
        """Test preprocessing empty DataFrame."""
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(empty_event_log, view_config)

        assert len(df_processed) == 0
        assert isinstance(df_processed, pd.DataFrame)

    def test_single_event(self, single_event_log, preprocessor):
        """Test preprocessing single-event DataFrame."""
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df_processed = preprocessor.process(single_event_log, view_config)

        assert len(df_processed) == 1
        assert 'activity_code' in df_processed.columns
        assert 'actual_time_scaled' in df_processed.columns

    def test_missing_resource_handling(self, sample_event_log, preprocessor):
        """Test handling of events with missing resource."""
        df = sample_event_log.copy()
        # Set some resources to None
        df.loc[df.index[:5], 'resource'] = None

        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        # This should not fail even with missing resources
        df_processed = preprocessor.process(df, view_config)

        assert len(df_processed) == len(df)


class TestMultiplePreprocessorCalls:
    """Test preprocessor behavior across multiple calls (statefulness)."""

    def test_consistent_encoding_across_calls(self, sample_event_log):
        """Test that same preprocessor gives consistent encodings."""
        preprocessor = DataPreprocessor()
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}

        # First call
        df1 = preprocessor.process(sample_event_log.copy(), view_config)

        # Second call with same data
        df2 = preprocessor.process(sample_event_log.copy(), view_config)

        # Encodings should be identical
        assert df1['activity_code'].tolist() == df2['activity_code'].tolist()

    def test_new_values_get_new_codes(self, sample_event_log):
        """Test that new categorical values get new codes."""
        preprocessor = DataPreprocessor()
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}

        # First call
        df1 = preprocessor.process(sample_event_log.copy(), view_config)
        existing_codes = set(df1['activity_code'].unique())

        # Add new activity
        df_new = sample_event_log.copy()
        new_row = df_new.iloc[0].copy()
        new_row['activity'] = 'BRAND_NEW_ACTIVITY'
        df_new = pd.concat(
            [df_new, pd.DataFrame([new_row])], ignore_index=True)

        # Second call with new activity
        df2 = preprocessor.process(df_new, view_config)
        new_activity_code = df2[df2['activity'] ==
                                'BRAND_NEW_ACTIVITY']['activity_code'].iloc[0]

        # New activity should have a new code
        assert new_activity_code not in existing_codes


class TestViewConfigurationsWithRealData:
    """Test all view configurations work with data from loader."""

    def test_time_view_config(self, sample_event_log, preprocessor):
        """Test time view: x=time (scaled), y=activity (encoded)."""
        view_config = {'x': 'actual_time', 'y': 'activity', 'view': 'time'}
        df = preprocessor.process(sample_event_log, view_config)

        assert 'actual_time_scaled' in df.columns
        assert 'activity_code' in df.columns

    def test_case_view_config(self, sample_event_log, preprocessor):
        """Test case view: x=case_id (encoded), y=time (scaled)."""
        view_config = {'x': 'case_id', 'y': 'actual_time', 'view': 'case'}
        df = preprocessor.process(sample_event_log, view_config)

        assert 'case_id_code' in df.columns
        assert 'actual_time_scaled' in df.columns

    def test_resource_view_config(self, sample_event_log, preprocessor):
        """Test resource view: x=time (scaled), y=resource (encoded)."""
        if sample_event_log['resource'].isna().all():
            pytest.skip("No resource data")

        view_config = {'x': 'actual_time', 'y': 'resource', 'view': 'resource'}
        df = preprocessor.process(sample_event_log, view_config)

        assert 'actual_time_scaled' in df.columns
        assert 'resource_code' in df.columns

    def test_activity_view_config(self, sample_event_log, preprocessor):
        """Test activity view: x=activity (encoded), y=time (scaled)."""
        view_config = {'x': 'activity', 'y': 'actual_time', 'view': 'activity'}
        df = preprocessor.process(sample_event_log, view_config)

        assert 'activity_code' in df.columns
        assert 'actual_time_scaled' in df.columns
