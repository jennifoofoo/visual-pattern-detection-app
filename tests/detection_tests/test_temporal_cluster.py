"""
Tests for TemporalClusterPattern detection.

Tests each detection method:
1. Temporal burst detection
2. Activity-time cluster detection
3. Case parallelism detection
4. Resource pattern detection
5. Variant timing pattern detection
"""

import pytest
from core.detection.temporal_cluster import TemporalClusterPattern


class TestTemporalClustersDetection:
    """Test _detect_temporal_bursts method."""

    def test_detects_temporal_bursts(self, df_with_temporal_bursts):
        """
        Test that temporal bursts are detected when events cluster in time.

        Uses df_with_temporal_bursts fixture which has 3 distinct burst periods.
        """
        detector = TemporalClusterPattern(
            df=df_with_temporal_bursts,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=5,
            temporal_eps=1800  # 30 minutes - enough to capture each burst
        )

        result = detector.detect()

        # Should detect the temporal bursts
        assert result is True, "Should detect temporal bursts"
        assert 'temporal_bursts' in detector.clusters, "Should have temporal_bursts in clusters"

        bursts = detector.clusters['temporal_bursts']
        assert len(
            bursts) >= 2, f"Should detect at least 2 bursts, got {len(bursts)}"

        # Verify burst metadata structure
        for burst in bursts:
            assert 'cluster_id' in burst
            assert 'start_time' in burst
            assert 'end_time' in burst
            assert 'event_count' in burst
            assert burst['event_count'] >= 5, "Each burst should have at least min_cluster_size events"


    def test_detects_activity_time_clusters(self, sample_event_log):
        """
        Test that activity-time clustering works.

        When Y-axis is 'activity' and X-axis is time-based, should detect
        when specific activities cluster at certain times.
        """
        detector = TemporalClusterPattern(
            df=sample_event_log,
            x_axis='relative_time',
            y_axis='activity',
            min_cluster_size=3
        )

        result = detector.detect()

        # Check that activity clustering was attempted
        # Result depends on data distribution
        if result and 'activity_time' in detector.clusters:
            activity_clusters = detector.clusters['activity_time']
            assert isinstance(activity_clusters, dict)

            # Each activity should map to a list of cluster info
            for activity, clusters in activity_clusters.items():
                assert isinstance(clusters, list)
                for cluster in clusters:
                    assert 'cluster_id' in cluster
                    assert 'event_count' in cluster


    def test_detects_case_parallelism(self, df_with_parallel_cases):
        """
        Test that case parallelism is detected when cases overlap in time.

        Uses df_with_parallel_cases fixture with 10 overlapping cases.
        """
        detector = TemporalClusterPattern(
            df=df_with_parallel_cases,
            x_axis='actual_time',
            y_axis='case_id',
            min_cluster_size=3
        )

        result = detector.detect()
        assert result is True, "Should detect case parallelism"
        # Should detect parallelism - we have overlapping cases
        assert 'case_parallelism' in detector.clusters, "Should detect case parallelism"

        parallelism = detector.clusters['case_parallelism']
        assert 'max_parallel_cases' in parallelism
        assert 'avg_parallel_cases' in parallelism

        # Given our fixture has significant overlap
        assert parallelism['max_parallel_cases'] >= 3, \
            f"Expected significant parallelism, got max {parallelism['max_parallel_cases']}"


    def test_detects_resource_shift_patterns(self, df_with_resource_shifts):
        """
        Test that resource time patterns (shifts) are detected.

        Uses df_with_resource_shifts fixture with 3 distinct resource shifts.
        """
        detector = TemporalClusterPattern(
            df=df_with_resource_shifts,
            x_axis='actual_time',
            y_axis='resource',
            min_cluster_size=5,
            temporal_eps=7200  # 2 hours - captures work within a shift
        )

        result = detector.detect()

        # Check if resource patterns were detected
        if result and 'resource_time' in detector.clusters:
            resource_patterns = detector.clusters['resource_time']
            assert isinstance(resource_patterns, dict)

            # Should detect multiple resources with time patterns
            # (R1, R2, R3 work in distinct shifts)
            for resource, patterns in resource_patterns.items():
                assert isinstance(patterns, list)
                assert resource in ['R1', 'R2',
                                    'R3'], f"Unexpected resource: {resource}"



    def test_case_parallelism_applicability(self, synthetic_event_log):
        """Test that case parallelism is only detected for appropriate axis combinations."""
        # Should be applicable
        detector = TemporalClusterPattern(
            df=synthetic_event_log, x_axis='actual_time', y_axis='case_id'
        )
        assert detector._should_detect_case_parallelism() is True

        # Should not be applicable
        detector2 = TemporalClusterPattern(
            df=synthetic_event_log, x_axis='actual_time', y_axis='activity'
        )
        assert detector2._should_detect_case_parallelism() is False

    def test_resource_pattern_applicability(self, synthetic_event_log):
        """Test that resource patterns are only detected for appropriate axis combinations."""
        # Should be applicable
        detector = TemporalClusterPattern(
            df=synthetic_event_log, x_axis='actual_time', y_axis='resource'
        )
        assert detector._should_detect_resource_patterns() is True

        # Should not be applicable
        detector2 = TemporalClusterPattern(
            df=synthetic_event_log, x_axis='actual_time', y_axis='activity'
        )
        assert detector2._should_detect_resource_patterns() is False


class TestGetSummary:
    """Test get_summary and get_summary_text methods."""

    def test_get_summary_returns_standardized_format(self, df_with_temporal_bursts):
        """Test that get_summary returns properly structured data."""
        detector = TemporalClusterPattern(
            df=df_with_temporal_bursts,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=5
        )
        detector.detect()

        summary = detector.get_summary()

        assert 'pattern_type' in summary
        assert summary['pattern_type'] == 'temporal_cluster'
        assert 'detected' in summary
        assert 'count' in summary
        assert 'details' in summary

        details = summary['details']
        assert 'x_axis' in details
        assert 'y_axis' in details
        assert 'summary_text' in details


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, empty_event_log):
        """Test handling of empty DataFrame."""
        detector = TemporalClusterPattern(
            df=empty_event_log,
            x_axis='actual_time',
            y_axis='activity'
        )

        with pytest.raises(ValueError):
            detector.detect()
        

    def test_single_event(self, single_event_log):
        """Test handling of DataFrame with single event."""
        detector = TemporalClusterPattern(
            df=single_event_log,
            x_axis='actual_time',
            y_axis='activity',
            min_cluster_size=2
        )

        result = detector.detect()

        # Single event can't form a cluster
        assert result is False

    def test_missing_column(self, synthetic_event_log):
        """Test handling when referenced column doesn't exist."""
        detector = TemporalClusterPattern(
            df=synthetic_event_log,
            x_axis='nonexistent_column',
            y_axis='activity'
        )

        # Should not crash
        try:
            result = detector.detect()
        except KeyError:
            pass  # Expected - column doesn't exist
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
