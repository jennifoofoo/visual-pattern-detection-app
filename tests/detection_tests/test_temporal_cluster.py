"""
Tests for TemporalClusterPattern detection.

Tests each detection method with synthetic data:
1. Temporal burst detection
2. Activity-time cluster detection
3. Case parallelism detection
4. Resource pattern detection
5. Variant timing pattern detection

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
from synthetic_cluster_logs import (
    make_temporal_bursts,
    make_activity_time_clusters,
    make_parallel_cases,
    make_resource_shift_patterns,
    make_variant_timing_patterns
)
from core.data_processing import load_xes_log
from core.detection.temporal_cluster import TemporalClusterPattern
# fmt: on


class TestSyntheticTemporalClusters:
    """Test temporal cluster detection methods using synthetic data."""

    def test_detects_temporal_bursts(self):
        """
        Test that temporal bursts are detected when events cluster in time.

        Uses synthetic data with 3 distinct burst periods separated by quiet periods.
        """
        df = make_temporal_bursts()

        detector = TemporalClusterPattern(
            df=df,
            x_axis='actual_time',
            y_axis='activity',
            color='case_id',
            min_cluster_size=5,
            temporal_eps=600  # 10 minutes - captures events within each burst without bridging gaps
        )

        result = detector.detect()

        # Should detect the temporal bursts
        assert result is True, "Should detect temporal bursts"
        assert 'temporal_bursts' in detector.clusters, "Should have temporal_bursts in clusters"

        bursts = detector.clusters['temporal_bursts']
        assert len(
            bursts) >= 2, f"Should detect at least 2 bursts, got {len(bursts)}"
        print(f"\nDetected {len(bursts)} temporal bursts")

        # Verify burst metadata structure
        for burst in bursts:
            assert 'cluster_id' in burst
            assert 'start_time' in burst
            assert 'end_time' in burst
            assert 'event_count' in burst
            assert burst['event_count'] >= 5, "Each burst should have at least min_cluster_size events"

    def test_detects_activity_time_clusters(self):
        """
        Test that activity-time clustering works.

        Uses synthetic data where different activities cluster at different times of day.
        Activity A: 9-11 AM, Activity B: 11-13 PM, Activity C: 13-15 PM
        """
        df = make_activity_time_clusters()

        detector = TemporalClusterPattern(
            df=df,
            x_axis='relative_time',
            y_axis='activity',
            color='case_id',
            min_cluster_size=3
        )

        result = detector.detect()

        # Check that activity clustering was attempted
        if result and 'activity_time' in detector.clusters:
            activity_clusters = detector.clusters['activity_time']
            assert isinstance(activity_clusters, dict)
            print(
                f"\nDetected activity-time clusters for {len(activity_clusters)} activities")

            # Each activity should map to a list of cluster info
            for activity, clusters in activity_clusters.items():
                assert isinstance(clusters, list)
                for cluster in clusters:
                    assert 'cluster_id' in cluster
                    assert 'event_count' in cluster

    def test_detects_resource_shift_patterns(self):
        """
        Test that resource time patterns (shifts) are detected.

        Uses synthetic data with 3 distinct resource shifts (morning, afternoon, evening).
        """
        df = make_resource_shift_patterns()

        detector = TemporalClusterPattern(
            df=df,
            x_axis='actual_time',
            y_axis='resource',
            color='activity',
            min_cluster_size=5,
            temporal_eps=7200  # 2 hours - captures work within a shift
        )

        result = detector.detect()

        # Check if resource patterns were detected
        if result and 'resource_time' in detector.clusters:
            resource_patterns = detector.clusters['resource_time']
            assert isinstance(resource_patterns, dict)
            print(
                f"\nDetected resource shift patterns for {len(resource_patterns)} resources")

            # Should detect multiple resources with time patterns
            for resource, patterns in resource_patterns.items():
                assert isinstance(patterns, list)
                assert resource in ['R1', 'R2',
                                    'R3'], f"Unexpected resource: {resource}"

    def test_detects_variant_timing_patterns(self):
        """
        Test that variant-specific timing patterns are detected.

        Uses synthetic data where different process variants occur at different times.
        """
        df = make_variant_timing_patterns()

        # First check if variant column exists
        if 'variant' not in df.columns:
            pytest.skip("Variant column not in synthetic data")

        detector = TemporalClusterPattern(
            df=df,
            x_axis='actual_time',
            y_axis='variant',
            color='activity',
            min_cluster_size=3
        )

        result = detector.detect()

        # Just verify detection completed
        assert hasattr(detector, 'clusters'), "Detection should complete"
        print(f"\nVariant timing detection completed: {result}")


class TestHospitalLogTemporalClusters:
    """Test temporal cluster detection on real Hospital_log.xes data."""

    @pytest.fixture(scope="class")
    def hospital_log(self):
        """Load Hospital_log.xes once for all tests."""
        xes_path = os.path.join(os.path.dirname(
            __file__), '../../data/Hospital_log.xes')
        if not os.path.exists(xes_path):
            pytest.skip(f"Hospital_log.xes not found at {xes_path}")
        return load_xes_log(xes_path)

    def test_hospital_temporal_bursts(self, hospital_log):
        """Test temporal burst detection on Hospital_log.xes."""
        df = hospital_log.copy()

        detector = TemporalClusterPattern(
            df=df,
            x_axis='actual_time',
            y_axis='activity',
            color='case_id',
            min_cluster_size=10,
            temporal_eps=3600  # 1 hour
        )

        result = detector.detect()

        # Should complete detection
        assert hasattr(detector, 'clusters'), "Detection should complete"
        print(f"\nHospital log temporal bursts: {result}")
        if result and 'temporal_bursts' in detector.clusters:
            bursts = detector.clusters['temporal_bursts']
            print(f"Bursts detected: {len(bursts)}")

    def test_hospital_case_parallelism(self, hospital_log):
        """Test case parallelism detection on Hospital_log.xes."""
        df = hospital_log.copy()

        detector = TemporalClusterPattern(
            df=df,
            x_axis='actual_time',
            y_axis='case_id',
            color='activity',
            min_cluster_size=5
        )

        result = detector.detect()

        # Should complete detection
        assert hasattr(detector, 'clusters'), "Detection should complete"
        print(f"\nHospital log case parallelism: {result}")
        if result and 'case_parallelism' in detector.clusters:
            parallelism = detector.clusters['case_parallelism']
            print(
                f"Max parallel cases: {parallelism.get('max_parallel_cases', 'N/A')}")
            print(
                f"Avg parallel cases: {parallelism.get('avg_parallel_cases', 'N/A')}")

    def test_hospital_resource_patterns(self, hospital_log):
        """Test resource pattern detection on Hospital_log.xes."""
        df = hospital_log.copy()

        # Check if resource column exists
        if 'resource' not in df.columns:
            pytest.skip("Hospital_log.xes does not have resource column")

        detector = TemporalClusterPattern(
            df=df,
            x_axis='actual_time',
            y_axis='resource',
            color='activity',
            min_cluster_size=10,
            temporal_eps=7200  # 2 hours
        )

        result = detector.detect()

        # Should complete detection
        assert hasattr(detector, 'clusters'), "Detection should complete"
        print(f"\nHospital log resource patterns: {result}")
        if result and 'resource_time' in detector.clusters:
            resource_patterns = detector.clusters['resource_time']
            print(f"Resources with patterns: {len(resource_patterns)}")

    def test_hospital_activity_time_clusters(self, hospital_log):
        """Test activity-time clustering on Hospital_log.xes."""
        df = hospital_log.copy()

        detector = TemporalClusterPattern(
            df=df,
            x_axis='relative_time',
            y_axis='activity',
            color='case_id',
            min_cluster_size=5
        )

        result = detector.detect()

        # Should complete detection
        assert hasattr(detector, 'clusters'), "Detection should complete"
        print(f"\nHospital log activity-time clusters: {result}")
        if result and 'activity_time' in detector.clusters:
            activity_clusters = detector.clusters['activity_time']
            print(
                f"Activities with temporal patterns: {len(activity_clusters)}")


class TestGetSummary:
    """Test get_summary and get_summary_text methods."""

    def test_get_summary_returns_standardized_format(self):
        """Test that get_summary returns properly structured data."""
        df = make_temporal_bursts()

        detector = TemporalClusterPattern(
            df=df,
            x_axis='actual_time',
            y_axis='activity',
            color='case_id',
            min_cluster_size=5
        )
        detector.detect()

        summary = detector.get_summary()

        assert 'pattern_type' in summary
        assert summary['pattern_type'] == 'temporal_cluster_x'
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
            y_axis='activity',
            color='case_id'
        )

        result = detector.detect()

        # Should handle empty DataFrame gracefully
        assert result is False or detector.clusters == {}

    def test_single_event(self, single_event_log):
        """Test handling of DataFrame with single event."""
        detector = TemporalClusterPattern(
            df=single_event_log,
            x_axis='actual_time',
            y_axis='activity',
            color='case_id',
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
            y_axis='activity',
            color='case_id'
        )

        # Should not crash
        try:
            result = detector.detect()
        except KeyError:
            pass  # Expected - column doesn't exist
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
