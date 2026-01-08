"""
Trend Detection Pattern for Dotted Charts.

Detects monotonic trends (increasing/decreasing) in event frequency over time.
Uses Mann-Kendall test for statistical significance and LOWESS for visualization.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from .pattern_base import Pattern
from config.extended_pattern_matrix import is_pattern_meaningful, get_pattern_info


class TrendPattern(Pattern):
    """
    Detects trends in event frequency over time in Dotted Charts.
    
    Features:
    - Mann-Kendall test for statistical trend significance
    - Sen's slope estimator for trend magnitude
    - LOWESS smoothing for trend line visualization
    - Optional: per-category trend analysis (e.g., per resource)
    """
    
    # Time column mappings
    TIME_COLUMNS = ['actual_time', 'relative_time']
    
    def __init__(
        self,
        view_config: Dict[str, str],
        aggregation_period: str = 'D',  # 'D' = daily, 'H' = hourly, 'W' = weekly
        min_periods: int = 5,
        significance_level: float = 0.05,
        analyze_per_category: bool = True
    ):
        """
        Initialize TrendPattern detector.
        
        Parameters
        ----------
        view_config : dict
            View configuration with 'x' and 'y' keys
        aggregation_period : str
            Pandas frequency string for time aggregation ('D', 'H', 'W')
        min_periods : int
            Minimum number of time periods required for trend analysis
        significance_level : float
            Alpha level for Mann-Kendall test (default 0.05)
        analyze_per_category : bool
            If True, analyze trends per Y-axis category (e.g., per resource)
        """
        super().__init__(name="Trend", view_config=view_config)
        self.aggregation_period = aggregation_period
        self.min_periods = min_periods
        self.significance_level = significance_level
        self.analyze_per_category = analyze_per_category
        
        # Results storage
        self.global_trend = None
        self.category_trends = {}
        self.trend_line_data = None
    
    def _is_time_based(self, x_col: str) -> bool:
        """Check if X-axis is time-based."""
        return x_col in self.TIME_COLUMNS or 'time' in x_col.lower()
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform Mann-Kendall trend test.
        
        The Mann-Kendall test is a non-parametric test for monotonic trend.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data (event counts per period)
            
        Returns
        -------
        dict
            Test results with keys: trend, p_value, z_score, s_statistic, slope
        """
        n = len(data)
        
        if n < self.min_periods:
            return {
                'trend': 'insufficient_data',
                'p_value': 1.0,
                'z_score': 0.0,
                's_statistic': 0,
                'slope': 0.0,
                'slope_percent': 0.0
            }
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = data[j] - data[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1
        
        # Calculate variance of S
        # Account for ties
        unique, counts = np.unique(data, return_counts=True)
        tie_sum = sum(t * (t - 1) * (2 * t + 5) for t in counts if t > 1)
        
        var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18
        
        # Calculate Z score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Sen's slope estimator (median of all slopes between pairs)
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j - i != 0:
                    slopes.append((data[j] - data[i]) / (j - i))
        
        slope = np.median(slopes) if slopes else 0.0
        
        # Calculate slope as percentage of mean
        mean_val = np.mean(data)
        slope_percent = (slope / mean_val * 100) if mean_val > 0 else 0.0
        
        # Determine trend direction
        # Require BOTH statistical significance AND practical significance (slope > 0.5%)
        MIN_PRACTICAL_SLOPE = 0.5  # Minimum slope % to be considered a real trend
        
        if p_value <= self.significance_level and abs(slope_percent) >= MIN_PRACTICAL_SLOPE:
            if z > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        elif p_value <= self.significance_level and abs(slope_percent) < MIN_PRACTICAL_SLOPE:
            trend = 'stable'  # Statistically significant but practically negligible
        else:
            trend = 'no_trend'
        
        return {
            'trend': trend,
            'p_value': p_value,
            'z_score': z,
            's_statistic': s,
            'slope': slope,
            'slope_percent': slope_percent
        }
    
    def _lowess_smooth(self, x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> np.ndarray:
        """
        Apply LOWESS (Locally Weighted Scatterplot Smoothing).
        
        Simple implementation using weighted local regression.
        
        Parameters
        ----------
        x : np.ndarray
            X values (time indices)
        y : np.ndarray
            Y values (event counts)
        frac : float
            Fraction of data to use for local regression (0 to 1)
            
        Returns
        -------
        np.ndarray
            Smoothed Y values
        """
        n = len(x)
        k = max(int(frac * n), 3)  # Number of neighbors
        smoothed = np.zeros(n)
        
        for i in range(n):
            # Calculate distances to all points
            distances = np.abs(x - x[i])
            
            # Get k nearest neighbors
            indices = np.argsort(distances)[:k]
            
            # Tricube weight function
            max_dist = distances[indices[-1]]
            if max_dist == 0:
                weights = np.ones(k)
            else:
                u = distances[indices] / max_dist
                weights = (1 - u**3)**3
            
            # Weighted linear regression
            X_local = np.vstack([np.ones(k), x[indices]]).T
            W = np.diag(weights)
            
            try:
                beta = np.linalg.lstsq(W @ X_local, W @ y[indices], rcond=None)[0]
                smoothed[i] = beta[0] + beta[1] * x[i]
            except Exception:
                smoothed[i] = y[i]
        
        return smoothed
    
    def _aggregate_events(self, df: pd.DataFrame, x_col: str, y_col: str = None) -> pd.DataFrame:
        """
        Aggregate events by time period.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        x_col : str
            Time column name
        y_col : str, optional
            Category column for per-category aggregation
            
        Returns
        -------
        pd.DataFrame
            Aggregated event counts per period
        """
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
            df = df.copy()
            df[x_col] = pd.to_datetime(df[x_col])
        
        # Set time as index for resampling
        df_temp = df.set_index(x_col)
        
        if y_col:
            # Per-category aggregation
            counts = df_temp.groupby(y_col).resample(self.aggregation_period).size()
            counts = counts.unstack(level=0, fill_value=0)
        else:
            # Global aggregation
            counts = df_temp.resample(self.aggregation_period).size()
            counts = counts.to_frame(name='event_count')
        
        return counts
    
    def detect(self, df: pd.DataFrame) -> bool:
        """
        Detect trends in the event log.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
            
        Returns
        -------
        bool
            True if any significant trend was detected
        """
        x_col = self.view_config.get('x', '')
        y_col = self.view_config.get('y', '')
        
        # Check if time-based
        if not self._is_time_based(x_col):
            self.detected = None
            return False
        
        # Check if time column exists and has valid data
        if x_col not in df.columns:
            self.detected = None
            return False
        
        try:
            # Aggregate events by time period
            event_counts = self._aggregate_events(df, x_col)
            
            if len(event_counts) < self.min_periods:
                self.detected = None
                return False
            
            # Global trend analysis
            global_counts = event_counts.sum(axis=1) if isinstance(event_counts, pd.DataFrame) and len(event_counts.columns) > 1 else event_counts.iloc[:, 0] if isinstance(event_counts, pd.DataFrame) else event_counts
            
            if isinstance(global_counts, pd.DataFrame):
                global_counts = global_counts.iloc[:, 0]
            
            self.global_trend = self._mann_kendall_test(global_counts.values)
            self.global_trend['time_index'] = global_counts.index
            self.global_trend['values'] = global_counts.values
            
            # Calculate LOWESS trend line for visualization
            x_numeric = np.arange(len(global_counts))
            self.trend_line_data = {
                'time': global_counts.index,
                'smoothed': self._lowess_smooth(x_numeric, global_counts.values),
                'original': global_counts.values
            }
            
            # Per-category trend analysis
            # Check if Y-axis supports category trends using the pattern matrix
            # Category trends make sense for resource/activity axes, not for case_id
            color_col = self.view_config.get('color', 'case_id')
            trend_info = get_pattern_info(x_col, y_col, color_col, 'trend')
            
            # Category trends are meaningful if Y-axis is not case_id (unique per case)
            # The matrix tells us if the view config is valid for trends
            supports_category_trends = (
                trend_info is not None and 
                y_col.lower() not in ['case_id', 'case:concept:name']
            )
            
            if self.analyze_per_category and y_col in df.columns and supports_category_trends:
                category_counts = self._aggregate_events(df, x_col, y_col)
                
                for category in category_counts.columns:
                    cat_values = category_counts[category].values
                    if len(cat_values) >= self.min_periods and cat_values.sum() > 0:
                        trend_result = self._mann_kendall_test(cat_values)
                        trend_result['values'] = cat_values
                        trend_result['time_index'] = category_counts.index
                        self.category_trends[category] = trend_result
            
            # Store detected results
            self.detected = {
                'global': self.global_trend,
                'categories': self.category_trends,
                'has_significant_trend': (
                    self.global_trend['trend'] != 'no_trend' or
                    any(t['trend'] != 'no_trend' for t in self.category_trends.values())
                )
            }
            
            return self.detected['has_significant_trend']
            
        except Exception:
            self.detected = None
            return False
    
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Visualize detected trends on the chart.
        
        Adds a trend annotation box in the top-right corner.
        Note: We don't draw a trend line through the chart because the Y-axis
        is typically categorical (case_id, activity, resource) - the trend
        is about event FREQUENCY over time, not Y-axis values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        fig : go.Figure
            Plotly figure to annotate
            
        Returns
        -------
        go.Figure
            Figure with trend visualization
        """
        return fig

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of detected trends.
        
        Returns
        -------
        dict
            Summary with global trend and top category trends
        """
        if self.detected is None:
            return {'count': 0, 'details': None}
        
        # Count significant trends
        significant_categories = [
            (cat, info) for cat, info in self.category_trends.items()
            if info['trend'] != 'no_trend'
        ]
        
        # Sort by absolute slope
        significant_categories.sort(key=lambda x: abs(x[1]['slope_percent']), reverse=True)
        
        # Format slope with more precision for small values
        slope_pct = self.global_trend['slope_percent']
        if abs(slope_pct) < 0.1:
            slope_str = f"{slope_pct:+.3f}%" if slope_pct != 0 else "~0%"
        else:
            slope_str = f"{slope_pct:+.1f}%"
        
        summary_text = f"Global: {self.global_trend['trend']} ({slope_str}/period)"
        
        if significant_categories:
            summary_text += f"\nTop changes:"
            for cat, info in significant_categories[:3]:
                arrow = '↗' if info['trend'] == 'increasing' else '↘'
                cat_slope = info['slope_percent']
                if abs(cat_slope) < 0.1:
                    cat_slope_str = f"{cat_slope:+.3f}%" if cat_slope != 0 else "~0%"
                else:
                    cat_slope_str = f"{cat_slope:+.1f}%"
                summary_text += f"\n  {arrow} {cat}: {cat_slope_str}"
        
        return {
            'count': len(significant_categories) + (1 if self.global_trend['trend'] != 'no_trend' else 0),
            'global_trend': self.global_trend['trend'],
            'global_change': self.global_trend['slope_percent'],
            'p_value': self.global_trend['p_value'],
            'significant_categories': len(significant_categories),
            'details': {
                'summary_text': summary_text,
                'category_trends': significant_categories[:5]
            }
        }

