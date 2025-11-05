"""
Pattern detection module for visual pattern detection in dotted charts.
"""

from .pattern_base import Pattern
from .cluster_pattern import ClusterPattern
from .gap_pattern import GapPattern

__all__ = ['Pattern', 'ClusterPattern', 'GapPattern']
