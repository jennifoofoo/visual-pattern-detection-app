"""
Pattern detection module for visual pattern detection in dotted charts.
"""

from .pattern_base import Pattern
from .cluster_pattern import ClusterPattern
from .outlier_detection import OutlierDetectionPattern
from .temporal_cluster import TemporalClusterPattern
from .gap_pattern import GapPattern
from .trend_pattern import TrendPattern
from .case_arrival_trend_pattern import CaseArrivalTrendPattern
from .sequence_detector import HorizontalSequencePatternDetector

__all__ = ['Pattern', 'ClusterPattern',
           'OutlierDetectionPattern', 'TemporalClusterPattern', 'GapPattern', 'TrendPattern',
           'CaseArrivalTrendPattern', 'HorizontalSequencePatternDetector']
