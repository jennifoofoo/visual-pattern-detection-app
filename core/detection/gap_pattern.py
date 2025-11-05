"""
Gap pattern detection for Dotted Charts.

Detects empty regions (gaps) in the visual dotted chart space.
Works on the visual/meta level - detecting gaps in the actual chart coordinates
as they appear in the visualization, not on raw data features.
"""

from .pattern_base import Pattern
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


class GapPattern(Pattern):
    """
    Detects gaps (empty regions) in the visual dotted chart space.
    
    Works on the visual/meta level - detecting empty regions in the 2D chart
    coordinates (X, Y) as they appear in the dotted chart visualization.
    
    Gaps represent visual empty regions where no points are plotted, which can indicate:
    - Bottlenecks or waiting times (if X-axis is time)
    - Missing data periods
    - Process delays or interruptions
    - Off-hours or breaks
    - Visual patterns in the chart layout
    """
    
    def __init__(
        self, 
        view_config: Dict[str, str],
        min_gap_duration: Optional[float] = None,
        group_by_y: bool = False,
        **kwargs
    ):
        """
        Initialize gap detector.
        
        Parameters
        ----------
        view_config : dict
            Configuration with "x" and "y" keys for chart dimensions
            Should include "view" key if available
        min_gap_duration : float, optional
            Minimum gap duration to detect. If None, automatically determined
            based on data distribution using IQR method.
        group_by_y : bool, default False
            If True, detects gaps separately for each Y-axis value (e.g., per activity).
            If False, detects global gaps across all points.
        **kwargs : dict
            Additional parameters (not used, but kept for compatibility)
        """
        super().__init__("Gap Detection", view_config)
        self.min_gap_duration = min_gap_duration
        self.group_by_y = group_by_y
        self.detected = None
    
    def _calculate_min_gap_duration(self, x_values: pd.Series) -> float:
        """
        Automatically calculate minimum gap duration based on data distribution.
        
        Uses IQR (Interquartile Range) method to determine threshold for gap detection.
        
        Parameters
        ----------
        x_values : pd.Series
            Sorted X-axis coordinate values
            
        Returns
        -------
        float
            Minimum gap duration threshold
        """
        if len(x_values) < 2:
            return 0.0
        
        # Calculate differences between consecutive points
        diffs = x_values.diff().dropna()
        
        if len(diffs) == 0:
            return 0.0
        
        # Use IQR method to find outliers (large gaps)
        q1 = diffs.quantile(0.25)
        q3 = diffs.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            # If IQR is 0, use median * 2
            median_gap = diffs.median()
            return max(median_gap * 2, 0.0) if median_gap > 0 else 0.0
        
        # Threshold: Q3 + 1.5 * IQR
        threshold = q3 + 1.5 * iqr
        
        # Ensure minimum threshold is at least 2x the median spacing
        median_gap = diffs.median()
        return max(threshold, median_gap * 2) if threshold > 0 else median_gap * 2
    
    def detect(self, df: pd.DataFrame) -> None:
        """
        Detect gaps in the visual dotted chart space.
        
        This method works on the visual level - detecting gaps in the actual
        chart coordinates (X, Y) as they appear in the dotted chart, not on
        raw data features. Gaps represent empty regions in the 2D visual space.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe with x and y coordinates (as shown in chart)
        """
        if df.empty:
            self.detected = None
            return
        
        try:
            x_col = self.view_config['x']
            y_col = self.view_config['y']
            
            # Ensure required columns exist
            if x_col not in df.columns:
                raise ValueError(f"X-axis column '{x_col}' not found in DataFrame")
            if y_col not in df.columns:
                raise ValueError(f"Y-axis column '{y_col}' not found in DataFrame")
            
            # Get visual coordinates (as they appear in the chart)
            # These are the actual X/Y values that will be plotted
            x_values = pd.to_numeric(df[x_col], errors='coerce')
            y_values = df[y_col]  # Keep original for grouping
            
            # Remove NaN values
            valid_mask = pd.notna(x_values)
            if not valid_mask.any():
                self.detected = None
                return
            
            df_clean = df[valid_mask].copy()
            x_values_clean = x_values[valid_mask]
            
            # Determine minimum gap duration based on visual X-axis spacing
            # This is calculated on the visual coordinates, not raw time values
            if self.min_gap_duration is None:
                min_gap = self._calculate_min_gap_duration(x_values_clean)
            else:
                min_gap = self.min_gap_duration
                # Convert user input to match the actual data type
                # If actual_time is datetime, user input is in seconds but data is in nanoseconds
                # Store original column for conversion check
                original_x_col = df[x_col]
                if pd.api.types.is_datetime64_any_dtype(original_x_col):
                    # User input is in seconds, but actual_time will be in nanoseconds when converted
                    # Convert seconds to nanoseconds
                    min_gap = min_gap * 1_000_000_000
            
            gaps = []
            
            if self.group_by_y:
                # Detect gaps separately for each Y-axis value (visual grouping)
                # This creates gaps per visual row/category in the chart
                for y_value in df_clean[y_col].unique():
                    df_group = df_clean[df_clean[y_col] == y_value].copy()
                    group_x_values = pd.to_numeric(df_group[x_col], errors='coerce').sort_values()
                    
                    if len(group_x_values) < 2:
                        continue
                    
                    # Find gaps within this visual group
                    # Gaps are detected in the visual X-coordinate space
                    group_gaps = self._find_gaps(group_x_values, min_gap, y_value)
                    gaps.extend(group_gaps)
            else:
                # Detect global gaps across all visual points
                # This finds empty regions in the entire chart space
                sorted_x_values = x_values_clean.sort_values()
                
                if len(sorted_x_values) < 2:
                    self.detected = None
                    return
                
                # Gaps are detected in the visual X-coordinate space
                gaps = self._find_gaps(sorted_x_values, min_gap, None)
            
            if not gaps:
                self.detected = None
                return
            
            # Store results
            self.detected = {
                'gaps': gaps,
                'min_gap_duration': min_gap,
                'group_by_y': self.group_by_y,
                'total_gaps': len(gaps),
                'total_gap_duration': sum(gap['duration'] for gap in gaps)
            }
            
        except Exception as e:
            print(f"Error during gap detection: {e}")
            import traceback
            traceback.print_exc()
            self.detected = None
    
    def _find_gaps(
        self, 
        sorted_x_values: pd.Series, 
        min_gap: float, 
        y_value: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """
        Find gaps in sorted visual X-coordinates.
        
        Detects empty regions in the visual chart space where no points exist
        between consecutive points in the sorted X-axis coordinate sequence.
        
        Parameters
        ----------
        sorted_x_values : pd.Series
            Sorted X-axis coordinate values (visual coordinates)
        min_gap : float
            Minimum gap size to consider (in visual coordinate units)
        y_value : optional
            Y-axis value if grouping by Y
            
        Returns
        -------
        list of dict
            List of gap dictionaries with 'start', 'end', 'duration', 'y_value'
            All values are in visual coordinate space
        """
        gaps = []
        
        for i in range(len(sorted_x_values) - 1):
            gap_duration = sorted_x_values.iloc[i + 1] - sorted_x_values.iloc[i]
            
            if gap_duration > min_gap:
                gap = {
                    'start': sorted_x_values.iloc[i],
                    'end': sorted_x_values.iloc[i + 1],
                    'duration': gap_duration,
                    'y_value': y_value
                }
                gaps.append(gap)
        
        return gaps
    
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Overlay detected gaps on a Plotly figure.
        
        Adds rectangles to highlight empty regions (gaps) in the chart.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        fig : go.Figure
            Plotly figure to annotate
            
        Returns
        -------
        go.Figure
            Figure with gap rectangles added
        """
        if self.detected is None or not self.detected.get('gaps'):
            return fig
        
        x_col = self.view_config['x']
        y_col = self.view_config['y']
        gaps = self.detected['gaps']
        
        # Get Y-axis range for rectangles
        y_values = df[y_col]
        if pd.api.types.is_numeric_dtype(y_values):
            y_min = y_values.min()
            y_max = y_values.max()
            y_range = y_max - y_min
            # Add padding
            y_min = y_min - y_range * 0.05
            y_max = y_max + y_range * 0.05
        else:
            # For categorical Y, use indices
            unique_y = df[y_col].unique()
            y_min = -0.5
            y_max = len(unique_y) - 0.5
        
        # Add rectangles for each gap
        for i, gap in enumerate(gaps):
            x_start = gap['start']
            x_end = gap['end']
            
            # If group_by_y, use specific Y range for this gap
            if self.group_by_y and gap.get('y_value') is not None:
                y_val = gap['y_value']
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    y_gap_min = y_val - 0.4
                    y_gap_max = y_val + 0.4
                else:
                    # Find index of y_value in unique values
                    unique_y = sorted(df[y_col].unique())
                    try:
                        y_idx = unique_y.index(y_val)
                        y_gap_min = y_idx - 0.4
                        y_gap_max = y_idx + 0.4
                    except ValueError:
                        y_gap_min = y_min
                        y_gap_max = y_max
            else:
                y_gap_min = y_min
                y_gap_max = y_max
            
            fig.add_shape(
                type="rect",
                x0=x_start,
                y0=y_gap_min,
                x1=x_end,
                y1=y_gap_max,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line=dict(color="rgba(255, 0, 0, 0.3)", width=1),
                layer="below"
            )
        
        return fig
    
    def get_gap_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected gaps.
        
        Returns
        -------
        dict
            Summary dictionary with gap statistics and details
        """
        if self.detected is None:
            return {
                'total_gaps': 0,
                'total_gap_duration': 0,
                'average_gap_duration': 0,
                'min_gap_threshold': self.min_gap_duration or 0,
                'group_by_y': self.group_by_y,
                'gaps': []
            }
        
        gaps = self.detected['gaps']
        total_gaps = len(gaps)
        total_duration = sum(gap['duration'] for gap in gaps)
        avg_duration = total_duration / total_gaps if total_gaps > 0 else 0
        
        return {
            'total_gaps': total_gaps,
            'total_gap_duration': total_duration,
            'average_gap_duration': avg_duration,
            'min_gap_threshold': self.detected['min_gap_duration'],
            'group_by_y': self.group_by_y,
            'gaps': gaps
        }

