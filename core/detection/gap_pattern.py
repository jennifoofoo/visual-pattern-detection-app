"""
Process-Aware Gap Detection for Dotted Charts.

Detects abnormal gaps using transition-specific normality:
- Extracts gaps within cases (case-aware)
- Learns normal gap duration per transition (Activity A → Activity B)
- Identifies abnormal gaps that exceed statistical thresholds
- Computes gap severity (duration / threshold)
"""

from .pattern_base import Pattern
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


class GapPattern(Pattern):
    """
    Process-aware gap detector using transition-specific normality.
    
    Detects abnormal gaps by:
    1. Extracting gaps within cases between consecutive events
    2. Learning normal gap duration per transition (A → B)
    3. Computing statistical thresholds (Q3 + 1.5*IQR, P95)
    4. Identifying gaps that exceed transition-specific thresholds
    5. Computing gap severity (duration / threshold)
    6. Pre-computing visual Y-positions for stable rendering
    
    Works on raw coordinates with case-aware, activity-aware semantics.
    """
    
    MIN_SAMPLES_FOR_NORMALITY = 5  # Minimum samples needed for statistical threshold
    
    def __init__(
        self, 
        view_config: Dict[str, str],
        y_is_categorical: bool = False,
        **kwargs
    ):
        """
        Initialize process-aware gap detector.
        
        Parameters
        ----------
        view_config : dict
            Configuration with "x" and "y" keys for chart dimensions
        y_is_categorical : bool, default False
            Whether Y-axis is categorical
        """
        super().__init__("Process-Aware Gap Detection", view_config)
        self.y_is_categorical = y_is_categorical
        self.detected = None
        self.transition_stats = None
        self.y_categories = None
        self.y_to_index = None
    
    def _is_time_like(self, x_series: pd.Series, x_col: str) -> bool:
        """
        Check if X-axis is time-like (required for gap detection).
        
        Parameters
        ----------
        x_series : pd.Series
            X-axis data series
        x_col : str
            Column name for X-axis
            
        Returns
        -------
        bool
            True if X is datetime or time-like column
        """
        # Check if datetime dtype
        if pd.api.types.is_datetime64_any_dtype(x_series):
            return True
        
        # Check column name
        time_like_names = [
            "actual_time", "relative_time", "relative_ratio",
            "logical_time", "logical_relative"
        ]
        return x_col in time_like_names
    
    def _extract_transition_gaps(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ) -> List[Dict[str, Any]]:
        """
        Extract gaps within cases between consecutive events.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        x_col : str
            X-axis column name (time-like)
        y_col : str
            Y-axis column name
            
        Returns
        -------
        list of dict
            List of gaps with transition information
        """
        if 'case_id' not in df.columns:
            return []
        
        if 'activity' not in df.columns:
            return []
        
        # Sort by case and time
        df_sorted = df.sort_values(['case_id', x_col]).copy()
        
        # Check if X is datetime
        x_is_datetime = pd.api.types.is_datetime64_any_dtype(df_sorted[x_col])
        
        gaps = []
        
        # Group by case
        for case_id, case_df in df_sorted.groupby('case_id'):
            case_df = case_df.reset_index(drop=True)
            
            if len(case_df) < 2:
                continue
            
            # Extract consecutive event pairs
            for i in range(len(case_df) - 1):
                event_a = case_df.iloc[i]
                event_b = case_df.iloc[i + 1]
                
                activity_a = event_a['activity']
                activity_b = event_b['activity']
                
                x_start = event_a[x_col]
                x_end = event_b[x_col]
                
                y_value_a = event_a[y_col]
                y_value_b = event_b[y_col]
                
                # Calculate duration
                if x_is_datetime:
                    # Ensure timestamps
                    if not isinstance(x_start, pd.Timestamp):
                        x_start = pd.Timestamp(int(x_start) if isinstance(x_start, (int, float)) else x_start)
                    if not isinstance(x_end, pd.Timestamp):
                        x_end = pd.Timestamp(int(x_end) if isinstance(x_end, (int, float)) else x_end)
                    
                    duration = (x_end - x_start).total_seconds()
                else:
                    duration = float(x_end - x_start)
                
                # Skip negative or zero durations (data quality issue)
                if duration <= 0:
                    continue
                
                gaps.append({
                    'case_id': case_id,
                    'activity_from': activity_a,
                    'activity_to': activity_b,
                    'transition': f"{activity_a} → {activity_b}",
                    'x_start': x_start,
                    'x_end': x_end,
                    'y_value_from': y_value_a,
                    'y_value_to': y_value_b,
                    'duration': duration
                })
        
        return gaps
    
    def _compute_normality_per_transition(
        self,
        gaps: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical normality thresholds per transition.
        
        Only computes thresholds for transitions with >= MIN_SAMPLES_FOR_NORMALITY.
        
        Parameters
        ----------
        gaps : list of dict
            List of gaps with transition information
            
        Returns
        -------
        dict
            Mapping from transition to statistics:
            {
                'transition_name': {
                    'count': int,
                    'median': float,
                    'q1': float,
                    'q3': float,
                    'iqr': float,
                    'p95': float,
                    'threshold': float
                }
            }
        """
        # Group gaps by transition
        transition_durations = {}
        for gap in gaps:
            transition = gap['transition']
            if transition not in transition_durations:
                transition_durations[transition] = []
            transition_durations[transition].append(gap['duration'])
        
        # Compute statistics per transition
        transition_stats = {}
        
        for transition, durations in transition_durations.items():
            durations_array = np.array(durations)
            count = len(durations)
            
            # Skip transitions with insufficient samples
            if count < self.MIN_SAMPLES_FOR_NORMALITY:
                continue
            
            median = np.median(durations_array)
            q1 = np.percentile(durations_array, 25)
            q3 = np.percentile(durations_array, 75)
            iqr = q3 - q1
            p95 = np.percentile(durations_array, 95)
            
            # Compute threshold: max(P95, Q3 + 1.5*IQR)
            threshold = max(p95, q3 + 1.5 * iqr)
            
            transition_stats[transition] = {
                'count': count,
                'median': median,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'p95': p95,
                'threshold': threshold
            }
        
        return transition_stats
    
    def _compute_y_position(
        self,
        gap: Dict[str, Any],
        df: pd.DataFrame,
        y_col: str
    ) -> tuple:
        """
        Compute visual Y-position for gap visualization.
        
        For categorical Y: Always shows gap at the FROM-resource row.
        This is semantically correct because the gap represents waiting time
        at/after the FROM activity, regardless of where the TO activity happens.
        
        For numeric Y: Uses the full Y-range of the plot.
        
        Parameters
        ----------
        gap : dict
            Gap information
        df : pd.DataFrame
            Event log dataframe
        y_col : str
            Y-axis column name
            
        Returns
        -------
        tuple
            (y_low, y_high) for visualization
        """
        if self.y_is_categorical:
            # Use precomputed category index mapping
            # ALWAYS show gap at the FROM-resource (where the waiting happens)
            y_value_from = gap['y_value_from']
            
            if y_value_from in self.y_to_index:
                idx = self.y_to_index[y_value_from]
                y_low = idx - 0.4
                y_high = idx + 0.4
            else:
                # Fallback
                y_low = 0
                y_high = 1
        else:
            # Numeric Y: will be computed from plot range during visualization
            # For now, use df range as placeholder
            y_low = df[y_col].min()
            y_high = df[y_col].max()
        
        return y_low, y_high
    
    def detect(self, df: pd.DataFrame) -> None:
        """
        Detect abnormal gaps using process-aware transition analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe with case_id and activity columns
        """
        if df.empty:
            self.detected = None
            return
        
        try:
            x_col = self.view_config['x']
            y_col = self.view_config['y']
            
            if x_col not in df.columns or y_col not in df.columns:
                self.detected = None
                return
            
            # Check if X is time-like (required for gap detection)
            if not self._is_time_like(df[x_col], x_col):
                self.detected = None
                return
            
            # Store Y categories if categorical
            if self.y_is_categorical:
                self.y_categories = list(pd.unique(df[y_col]))
                self.y_to_index = {cat: idx for idx, cat in enumerate(self.y_categories)}
            
            # Extract transition gaps
            all_gaps = self._extract_transition_gaps(df, x_col, y_col)
            
            if not all_gaps:
                self.detected = None
                return
            
            # Compute normality per transition
            self.transition_stats = self._compute_normality_per_transition(all_gaps)
            
            if not self.transition_stats:
                # No transitions with sufficient samples
                self.detected = None
                return
            
            # Identify abnormal gaps
            abnormal_gaps = []
            
            for gap in all_gaps:
                transition = gap['transition']
                
                # Skip transitions without computed thresholds
                if transition not in self.transition_stats:
                    continue
                
                duration = gap['duration']
                threshold = self.transition_stats[transition]['threshold']
                
                if duration > threshold:
                    # Compute severity
                    severity = duration / threshold
                    
                    # Compute Y position for visualization
                    y_low, y_high = self._compute_y_position(gap, df, y_col)
                    
                    # Build complete abnormal gap structure
                    abnormal_gap = {
                        'case_id': gap['case_id'],
                        'transition': transition,
                        'activity_from': gap['activity_from'],
                        'activity_to': gap['activity_to'],
                        'x_start': gap['x_start'],
                        'x_end': gap['x_end'],
                        'duration': duration,
                        'threshold': threshold,
                        'severity': severity,
                        'y_low': y_low,
                        'y_high': y_high,
                        'y_value_from': gap['y_value_from'],
                        'y_value_to': gap['y_value_to']
                    }
                    
                    abnormal_gaps.append(abnormal_gap)
            
            if not abnormal_gaps:
                self.detected = None
                return
            
            # Build result summary
            total_gaps = len(all_gaps)
            total_abnormal = len(abnormal_gaps)
            total_transitions = len(self.transition_stats)
            transitions_with_anomalies = len(set(g['transition'] for g in abnormal_gaps))
            
            self.detected = {
                'total_gaps': total_gaps,
                'total_abnormal_gaps': total_abnormal,
                'total_transitions': total_transitions,
                'transitions_with_anomalies': transitions_with_anomalies,
                'abnormal_gaps': abnormal_gaps,
                'transition_stats': self.transition_stats
            }
            
        except Exception as e:
            self.detected = None
            raise
    
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Overlay abnormal gaps on a Plotly figure using precomputed positions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        fig : go.Figure
            Plotly figure to annotate
            
        Returns
        -------
        go.Figure
            Figure with abnormal gap rectangles added
        """
        if self.detected is None or not self.detected.get('abnormal_gaps'):
            return fig
        
        abnormal_gaps = self.detected['abnormal_gaps']
        y_col = self.view_config['y']
        
        # For numeric Y: try to get actual plot range
        if not self.y_is_categorical:
            # Try to extract Y range from figure layout
            if fig.layout.yaxis and fig.layout.yaxis.range:
                y_min_plot = fig.layout.yaxis.range[0]
                y_max_plot = fig.layout.yaxis.range[1]
            else:
                # Extract from figure traces
                all_y_values = []
                if fig.data:
                    for trace in fig.data:
                        if hasattr(trace, 'y') and trace.y is not None:
                            all_y_values.extend([y for y in trace.y if y is not None])
                
                if all_y_values:
                    y_min_plot = min(all_y_values)
                    y_max_plot = max(all_y_values)
                else:
                    # Final fallback
                    y_min_plot = df[y_col].min()
                    y_max_plot = df[y_col].max()
            
            # Update all gaps to use actual plot range
            for gap in abnormal_gaps:
                gap['y_low'] = y_min_plot
                gap['y_high'] = y_max_plot
        
        # Draw gaps
        for gap in abnormal_gaps:
            x_start = gap['x_start']
            x_end = gap['x_end']
            y_low = gap['y_low']
            y_high = gap['y_high']
            
            # Draw red rectangle for abnormal gap
            fig.add_shape(
                type="rect",
                x0=x_start,
                y0=y_low,
                x1=x_end,
                y1=y_high,
                fillcolor="rgba(255, 0, 0, 0.25)",
                line=dict(color="rgba(255, 0, 0, 0.6)", width=2),
                layer="below"
            )
        
        return fig
    
    def get_gap_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected abnormal gaps.
        
        Returns
        -------
        dict
            Summary dictionary with gap statistics and transition info
        """
        if self.detected is None:
            return {
                'total_gaps': 0,
                'total_abnormal_gaps': 0,
                'total_transitions': 0,
                'transitions_with_anomalies': 0,
                'total_magnitude': 0,
                'average_magnitude': 0,
                'abnormal_gaps': [],
                'gaps': [],  # Alias for backward compatibility
                'transition_stats': {}
            }
        
        # Calculate total and average duration for UI display
        abnormal_gaps = self.detected['abnormal_gaps']
        total_duration = sum(gap['duration'] for gap in abnormal_gaps) if abnormal_gaps else 0
        avg_duration = total_duration / len(abnormal_gaps) if abnormal_gaps else 0
        
        return {
            'total_gaps': self.detected['total_gaps'],
            'total_abnormal_gaps': self.detected['total_abnormal_gaps'],
            'total_transitions': self.detected['total_transitions'],
            'transitions_with_anomalies': self.detected['transitions_with_anomalies'],
            'total_magnitude': total_duration,  # Total duration of abnormal gaps
            'average_magnitude': avg_duration,  # Average duration of abnormal gaps
            'abnormal_gaps': abnormal_gaps,
            'gaps': abnormal_gaps,  # Alias for backward compatibility with UI
            'transition_stats': self.detected['transition_stats']
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get standardized pattern summary.
        
        Returns
        -------
        Dict[str, Any]
            Standardized summary with pattern_type, detected, count, and details
        """
        gap_summary = self.get_gap_summary()
        
        return {
            'pattern_type': 'gap',
            'detected': self.detected is not None and gap_summary['total_abnormal_gaps'] > 0,
            'count': gap_summary['total_abnormal_gaps'],
            'details': gap_summary
        }
