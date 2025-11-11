"""
Pattern detection module for visual pattern detection in dotted charts.
"""

from .pattern_base import Pattern
from .cluster_pattern import ClusterPattern
from .outlier_detection import OutlierDetectionPattern
from .temporal_cluster import TemporalClusterPattern
from .gap_pattern import GapPattern

__all__ = ['Pattern', 'ClusterPattern',
           'OutlierDetectionPattern', 'TemporalClusterPattern', 'GapPattern']
