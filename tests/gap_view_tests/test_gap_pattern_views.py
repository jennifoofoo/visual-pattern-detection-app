"""
Pytest tests for gap detection across all X-views and Y-views.
"""

import pytest
import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.gap_view_tests.synthetic_gap_logs import (
    make_numeric_y_gap,
    make_categorical_y_gap,
    make_actual_time_gap,
    make_relative_time_gap,
    make_relative_ratio_gap,
    make_logical_time_gap,
    make_logical_relative_gap
)
from core.detection.gap_pattern import GapPattern


def test_numeric_y_gap():
    """Test gap detection with numeric X and Y (2D gap)."""
    df = make_numeric_y_gap()
    
    view_config = {'x': 'x', 'y': 'y'}
    detector = GapPattern(
        view_config=view_config,
        min_gap_area=0.01,
        y_is_categorical=False
    )
    detector.detect(df)
    
    assert detector.detected is not None, "Should detect gaps for numeric Y"
    assert detector.detected['total_gaps'] >= 1, f"Expected at least 1 gap, got {detector.detected['total_gaps']}"


def test_categorical_y_gap():
    """Test gap detection with categorical Y (horizontal gap in category A)."""
    df = make_categorical_y_gap()
    
    view_config = {'x': 'x', 'y': 'y'}
    detector = GapPattern(
        view_config=view_config,
        min_gap_x_width=0.01,
        y_is_categorical=True
    )
    detector.detect(df)
    
    assert detector.detected is not None, "Should detect gaps for categorical Y"
    assert detector.detected['total_gaps'] >= 1, f"Expected at least 1 gap, got {detector.detected['total_gaps']}"


def test_actual_time_gap():
    """Test gap detection with actual_time X-axis."""
    df = make_actual_time_gap()
    
    # Test with categorical Y
    view_config = {'x': 'actual_time', 'y': 'y'}
    detector = GapPattern(
        view_config=view_config,
        min_gap_x_width=0.01,
        y_is_categorical=True
    )
    detector.detect(df)
    
    assert detector.detected is not None, "Should detect gaps for actual_time × categorical Y"
    assert detector.detected['total_gaps'] >= 1, f"Expected at least 1 gap, got {detector.detected['total_gaps']}"


def test_relative_time_gap():
    """Test gap detection with relative_time X-axis."""
    df = make_relative_time_gap()
    
    # Test with numeric Y
    view_config = {'x': 'relative_time', 'y': 'y'}
    detector = GapPattern(
        view_config=view_config,
        min_gap_area=0.01,
        y_is_categorical=False
    )
    detector.detect(df)
    
    assert detector.detected is not None, "Should detect gaps for relative_time × numeric Y"
    assert detector.detected['total_gaps'] >= 1, f"Expected at least 1 gap, got {detector.detected['total_gaps']}"


def test_relative_ratio_gap():
    """Test gap detection with relative_ratio X-axis."""
    df = make_relative_ratio_gap()
    
    # Test with numeric Y
    view_config = {'x': 'relative_ratio', 'y': 'y'}
    detector = GapPattern(
        view_config=view_config,
        min_gap_area=0.01,
        y_is_categorical=False
    )
    detector.detect(df)
    
    assert detector.detected is not None, "Should detect gaps for relative_ratio × numeric Y"
    assert detector.detected['total_gaps'] >= 1, f"Expected at least 1 gap, got {detector.detected['total_gaps']}"


def test_logical_time_gap():
    """Test gap detection with logical_time X-axis."""
    df = make_logical_time_gap()
    
    # Test with numeric Y
    view_config = {'x': 'logical_time', 'y': 'y'}
    detector = GapPattern(
        view_config=view_config,
        min_gap_area=0.01,
        y_is_categorical=False
    )
    detector.detect(df)
    
    assert detector.detected is not None, "Should detect gaps for logical_time × numeric Y"
    assert detector.detected['total_gaps'] >= 1, f"Expected at least 1 gap, got {detector.detected['total_gaps']}"


def test_logical_relative_gap():
    """Test gap detection with logical_relative X-axis."""
    df = make_logical_relative_gap()
    
    # Test with numeric Y
    view_config = {'x': 'logical_relative', 'y': 'y'}
    detector = GapPattern(
        view_config=view_config,
        min_gap_area=0.01,
        y_is_categorical=False
    )
    detector.detect(df)
    
    assert detector.detected is not None, "Should detect gaps for logical_relative × numeric Y"
    assert detector.detected['total_gaps'] >= 1, f"Expected at least 1 gap, got {detector.detected['total_gaps']}"
