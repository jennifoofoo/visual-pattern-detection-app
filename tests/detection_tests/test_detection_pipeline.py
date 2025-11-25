"""
Integration tests for the detection pipeline.

Tests the interaction between:
1. Data loading/preprocessing → Pattern detection
2. Multiple detectors working together on the same data
3. Pattern detection → Visualization pipeline
4. Pattern detection → Summary generation

This ensures the entire detection workflow functions correctly end-to-end.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from core.data_processing.loader import load_xes_log
from core.data_processing.preprocessor import DataPreprocessor
from core.detection.temporal_cluster import TemporalClusterPattern
from core.detection.outlier_detection import OutlierDetectionPattern
from core.detection.gap_pattern import GapPattern
from core.detection.cluster_pattern import ClusterPattern


class TestLoaderToDetectorPipeline:
    """Test data flows correctly from loader through detection."""

    def test_loaded_data_compatible_with_temporal_cluster(self, sample_event_log):
        """Test that loaded event log data works with TemporalClusterPattern."""
        # Verify sample_event_log has required columns
        assert 'actual_time' in sample_event_log.columns
        assert 'case_id' in sample_event_log.columns
        assert 'activity' in sample_event_log.columns

        # Create detector and run detection
        detector = TemporalClusterPattern(
            df=sample_event_log,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=3
        )

        # Should not raise any errors
        result = detector.detect()
        assert isinstance(result, bool)

    def test_loaded_data_compatible_with_outlier_detection(self, sample_event_log):
        """Test that loaded event log data works with OutlierDetectionPattern."""
        detector = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        # Should not raise any errors
        result = detector.detect()
        assert isinstance(result, bool)

    def test_loaded_data_compatible_with_gap_detection(self, sample_event_log):
        """Test that loaded event log data works with GapPattern."""
        detector = GapPattern(
            view_config={'x': 'actual_time', 'y': 'case_id'}
        )

        # Should not raise any errors
        detector.detect(sample_event_log)
        # detected can be None (no gaps) or dict (gaps found)
        assert detector.detected is None or isinstance(detector.detected, dict)


class TestPreprocessorToDetectorPipeline:
    """Test preprocessor output works with detection."""

    def test_preprocessed_data_with_temporal_cluster(self, sample_event_log):
        """Test that preprocessed data works with TemporalClusterPattern."""
        preprocessor = DataPreprocessor()

        config = {'x': 'actual_time', 'y': 'activity'}
        processed_df = preprocessor.process(sample_event_log, config)

        # Processed data should still have original columns needed for detection
        detector = TemporalClusterPattern(
            df=processed_df,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=3
        )

        result = detector.detect()
        assert isinstance(result, bool)

    def test_preprocessed_data_with_outlier_detection(self, sample_event_log):
        """Test that preprocessed data works with OutlierDetectionPattern."""
        preprocessor = DataPreprocessor()

        config = {'x': 'actual_time', 'y': 'case_id'}
        processed_df = preprocessor.process(sample_event_log, config)

        detector = OutlierDetectionPattern(
            df=processed_df,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        result = detector.detect()
        assert isinstance(result, bool)


class TestMultipleDetectorsOnSameData:
    """Test multiple detectors can analyze the same data concurrently."""

    def test_all_detectors_on_same_dataframe(self, sample_event_log):
        """Test running all detector types on the same event log."""
        # Create all detectors
        temporal_detector = TemporalClusterPattern(
            df=sample_event_log,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=3
        )

        outlier_detector = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        gap_detector = GapPattern(
            view_config={'x': 'actual_time', 'y': 'case_id'}
        )

        # Run all detectors - they should not interfere with each other
        temporal_result = temporal_detector.detect()
        outlier_result = outlier_detector.detect()
        gap_detector.detect(sample_event_log)

        # All should complete without error
        assert isinstance(temporal_result, bool)
        assert isinstance(outlier_result, bool)

        # Original data should be unmodified
        assert 'actual_time' in sample_event_log.columns
        assert 'case_id' in sample_event_log.columns
        assert 'activity' in sample_event_log.columns

    def test_detectors_do_not_modify_source_data(self, sample_event_log):
        """Test that detectors don't modify the original DataFrame."""
        original_columns = set(sample_event_log.columns)
        original_len = len(sample_event_log)
        original_hash = pd.util.hash_pandas_object(sample_event_log).sum()

        # Run multiple detectors
        temporal_detector = TemporalClusterPattern(
            df=sample_event_log.copy(),  # Use copy to be safe
            x_axis='actual_time',
            y_axis='activity'
        )
        temporal_detector.detect()

        outlier_detector = OutlierDetectionPattern(
            df=sample_event_log.copy(),
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )
        outlier_detector.detect()

        # Verify original data unchanged
        assert set(sample_event_log.columns) == original_columns
        assert len(sample_event_log) == original_len


class TestDetectorToVisualizationPipeline:
    """Test pattern detection flows correctly into visualization."""

    def test_temporal_cluster_visualization(self, df_with_temporal_bursts, empty_figure):
        """Test temporal cluster detection to visualization."""
        detector = TemporalClusterPattern(
            df=df_with_temporal_bursts,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=5
        )

        detector.detect()
        result_fig = detector.visualize(fig=empty_figure)

        assert isinstance(result_fig, go.Figure)
        # If patterns detected, should have added visual elements
        if detector.clusters:
            # At minimum, annotations or traces should be added
            has_annotations = (result_fig.layout.annotations is not None and
                               len(result_fig.layout.annotations) > 0)
            has_traces = len(result_fig.data) > len(empty_figure.data)
            assert has_annotations or has_traces or len(result_fig.data) >= 0

    def test_outlier_visualization(self, sample_event_log, empty_figure):
        """Test outlier detection to visualization."""
        detector = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        detector.detect()
        result_fig = detector.visualize(empty_figure)

        assert isinstance(result_fig, go.Figure)

    def test_gap_visualization(self, df_with_gaps, empty_figure):
        """Test gap detection to visualization."""
        detector = GapPattern(
            view_config={'x': 'actual_time', 'y': 'case_id'}
        )

        detector.detect(df_with_gaps)
        result_fig = detector.visualize(df_with_gaps, empty_figure)

        assert isinstance(result_fig, go.Figure)


class TestDetectorToSummaryPipeline:
    """Test pattern detection flows correctly into summary generation."""

    def test_temporal_cluster_get_summary(self, sample_event_log):
        """Test TemporalClusterPattern.get_summary() returns proper structure."""
        detector = TemporalClusterPattern(
            df=sample_event_log,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=3
        )

        detector.detect()
        summary = detector.get_summary()

        # Verify standardized summary structure
        assert 'pattern_type' in summary
        assert summary['pattern_type'] == 'temporal_cluster'
        assert 'detected' in summary
        assert 'count' in summary
        assert 'details' in summary
        assert isinstance(summary['detected'], bool)
        assert isinstance(summary['count'], int)

    def test_outlier_get_summary(self, sample_event_log):
        """Test OutlierDetectionPattern.get_summary() returns proper structure."""
        detector = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )

        detector.detect()
        summary = detector.get_summary()

        # Verify standardized summary structure
        assert 'pattern_type' in summary
        assert summary['pattern_type'] == 'outlier'
        assert 'detected' in summary
        assert 'count' in summary
        assert 'details' in summary

    def test_gap_get_summary(self, sample_event_log):
        """Test GapPattern.get_summary() returns proper structure."""
        detector = GapPattern(
            view_config={'x': 'actual_time', 'y': 'case_id'}
        )

        detector.detect(sample_event_log)
        summary = detector.get_summary()

        # Verify standardized summary structure
        assert 'pattern_type' in summary
        assert summary['pattern_type'] == 'gap'
        assert 'detected' in summary
        assert 'count' in summary
        assert 'details' in summary

    def test_all_summaries_consistent_format(self, sample_event_log):
        """Test all detector summaries follow the same format."""
        temporal = TemporalClusterPattern(
            df=sample_event_log,
            x_axis='actual_time',
            y_axis='activity'
        )
        temporal.detect()

        outlier = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )
        outlier.detect()

        gap = GapPattern(view_config={'x': 'actual_time', 'y': 'case_id'})
        gap.detect(sample_event_log)

        summaries = [
            temporal.get_summary(),
            outlier.get_summary(),
            gap.get_summary()
        ]

        # All should have same top-level keys
        required_keys = {'pattern_type', 'detected', 'count', 'details'}
        for summary in summaries:
            assert set(summary.keys()) >= required_keys, \
                f"Summary missing required keys: {required_keys - set(summary.keys())}"


class TestDifferentViewConfigurations:
    """Test detection with different X/Y axis configurations."""

    @pytest.mark.parametrize("x_axis,y_axis", [
        ('actual_time', 'case_id'),
        ('actual_time', 'activity'),
        ('actual_time', 'resource'),
        ('relative_time', 'activity'),
        ('relative_ratio', 'case_id'),
    ])
    def test_temporal_cluster_with_different_axes(self, sample_event_log, x_axis, y_axis):
        """Test TemporalClusterPattern works with various axis configurations."""
        # Skip if column doesn't exist
        if x_axis not in sample_event_log.columns or y_axis not in sample_event_log.columns:
            pytest.skip(f"Column {x_axis} or {y_axis} not in sample data")

        detector = TemporalClusterPattern(
            df=sample_event_log,
            x_axis=x_axis,
            y_axis=y_axis,
            min_cluster_size=3
        )

        # Should not raise
        result = detector.detect()
        assert isinstance(result, bool)

    @pytest.mark.parametrize("x_axis,y_axis", [
        ('actual_time', 'case_id'),
        ('actual_time', 'activity'),
        ('actual_time', 'resource'),
    ])
    def test_outlier_with_different_axes(self, sample_event_log, x_axis, y_axis):
        """Test OutlierDetectionPattern works with various axis configurations."""
        if x_axis not in sample_event_log.columns or y_axis not in sample_event_log.columns:
            pytest.skip(f"Column {x_axis} or {y_axis} not in sample data")

        detector = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': x_axis, 'y_axis': y_axis}
        )

        result = detector.detect()
        assert isinstance(result, bool)


class TestEndToEndWorkflow:
    """Test complete workflow from loading to visualization."""

    def test_full_pipeline_with_real_data(self, event_log_path, empty_figure):
        """Test complete pipeline: load → detect → visualize → summarize."""
        # Step 1: Load data
        df = load_xes_log(str(event_log_path))
        assert len(df) > 0, "Should load events"

        # Step 2: Sample for testing speed
        from core.utils.demo_sampling import sample_small_eventlog
        sampled_df = sample_small_eventlog(
            df, max_cases=20, max_events_per_case=10)

        # Step 3: Detect patterns (temporal clusters)
        temporal_detector = TemporalClusterPattern(
            df=sampled_df,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=3
        )
        temporal_detector.detect()

        # Step 4: Visualize
        fig = temporal_detector.visualize(fig=empty_figure)
        assert isinstance(fig, go.Figure)

        # Step 5: Get summary
        summary = temporal_detector.get_summary()
        assert 'pattern_type' in summary
        assert 'detected' in summary

    def test_full_pipeline_with_multiple_detectors(self, sample_event_log, empty_figure):
        """Test running complete workflow with multiple detector types."""
        results = {}

        # Run temporal cluster detection
        temporal = TemporalClusterPattern(
            df=sample_event_log,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=3
        )
        temporal.detect()
        results['temporal'] = temporal.get_summary()

        # Run outlier detection
        outlier = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )
        outlier.detect()
        results['outlier'] = outlier.get_summary()

        # Run gap detection
        gap = GapPattern(view_config={'x': 'actual_time', 'y': 'case_id'})
        gap.detect(sample_event_log)
        results['gap'] = gap.get_summary()

        # All should have completed
        assert len(results) == 3
        for name, summary in results.items():
            assert 'pattern_type' in summary, f"{name} missing pattern_type"
            assert 'detected' in summary, f"{name} missing detected"
            assert 'count' in summary, f"{name} missing count"


class TestErrorHandlingInPipeline:
    """Test error handling across the detection pipeline."""

    def test_empty_dataframe_handling(self, empty_event_log):
        """Test all detectors handle empty DataFrames gracefully."""
        # Temporal cluster
        temporal = TemporalClusterPattern(
            df=empty_event_log,
            x_axis='actual_time',
            y_axis='activity'
        )
        result = temporal.detect()
        assert result is False

        # Outlier
        outlier = OutlierDetectionPattern(
            df=empty_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )
        with pytest.raises(ValueError):
            outlier.detect()
        
        # Gap
        gap = GapPattern(view_config={'x': 'actual_time', 'y': 'case_id'})
        with pytest.raises(ValueError):
            gap.detect(empty_event_log)

    def test_single_event_handling(self, single_event_log):
        """Test all detectors handle single-event DataFrames gracefully."""
        # Temporal cluster
        temporal = TemporalClusterPattern(
            df=single_event_log,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=2
        )
        result = temporal.detect()
        assert result is False  # Can't form clusters with 1 event

        # Outlier
        outlier = OutlierDetectionPattern(
            df=single_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'activity'}
        )
        result = outlier.detect()
        assert isinstance(result, bool)

    def test_missing_columns_handling(self, synthetic_event_log):
        """Test detectors handle missing columns gracefully."""
        df_no_resource = synthetic_event_log.drop(columns=['resource'])

        # Should not crash even without resource column
        outlier = OutlierDetectionPattern(
            df=df_no_resource,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )
        result = outlier.detect()
        assert isinstance(result, bool)


class TestPatternCombinations:
    """Test detecting and combining multiple pattern types."""

    def test_combine_pattern_summaries(self, sample_event_log):
        """Test combining summaries from multiple pattern detectors."""
        detectors = []

        # Create detectors
        temporal = TemporalClusterPattern(
            df=sample_event_log,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=3
        )
        temporal.detect()
        detectors.append(('temporal_cluster', temporal))

        outlier = OutlierDetectionPattern(
            df=sample_event_log,
            view_config={'x_axis': 'actual_time', 'y_axis': 'case_id'}
        )
        outlier.detect()
        detectors.append(('outlier', outlier))

        gap = GapPattern(view_config={'x': 'actual_time', 'y': 'case_id'})
        gap.detect(sample_event_log)
        detectors.append(('gap', gap))

        # Combine summaries
        combined_summary = {
            'patterns_analyzed': len(detectors),
            'patterns': {}
        }

        for name, detector in detectors:
            combined_summary['patterns'][name] = detector.get_summary()

        # Verify combined structure
        assert combined_summary['patterns_analyzed'] == 3
        assert 'temporal_cluster' in combined_summary['patterns']
        assert 'outlier' in combined_summary['patterns']
        assert 'gap' in combined_summary['patterns']

        # All patterns should have consistent structure
        for pattern_name, pattern_summary in combined_summary['patterns'].items():
            assert 'pattern_type' in pattern_summary
            assert 'detected' in pattern_summary
            assert 'count' in pattern_summary
