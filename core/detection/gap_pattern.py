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
import streamlit as st
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

    MIN_SAMPLES_FOR_NORMALITY = 15  # Minimum samples for stable threshold estimation
    MAX_GAPS_TO_DISPLAY = 50  # Limit visualization to top N gaps by severity

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
                        x_start = pd.Timestamp(int(x_start) if isinstance(
                            x_start, (int, float)) else x_start)
                    if not isinstance(x_end, pd.Timestamp):
                        x_end = pd.Timestamp(int(x_end) if isinstance(
                            x_end, (int, float)) else x_end)

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
        """Detect process-aware gaps in the event log."""
        # Validate DataFrame is not empty
        if df is None or len(df) == 0:
            raise ValueError("Cannot detect gaps: DataFrame is empty")

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
                self.y_to_index = {cat: idx for idx,
                                   cat in enumerate(self.y_categories)}

            # Extract transition gaps
            all_gaps = self._extract_transition_gaps(df, x_col, y_col)

            if not all_gaps:
                self.detected = None
                return

            # Compute normality per transition
            self.transition_stats = self._compute_normality_per_transition(
                all_gaps)

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
            transitions_with_anomalies = len(
                set(g['transition'] for g in abnormal_gaps))

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

    @staticmethod
    def _severity_to_color(severity: float) -> str:
        """Map severity to color gradient: yellow → orange → red → darkred."""
        if severity < 2:
            return 'rgba(255, 193, 7, 0.7)'    # Yellow - mild
        elif severity < 3:
            return 'rgba(255, 152, 0, 0.8)'   # Orange - moderate
        elif severity < 5:
            return 'rgba(220, 53, 69, 0.85)'  # Red - severe
        else:
            return 'rgba(139, 0, 0, 0.9)'     # Dark red - critical

    @staticmethod
    def _severity_to_width(severity: float) -> float:
        """Map severity to line width: more severe = thicker."""
        return min(1.5 + severity * 0.4, 5)  # 1.9px to max 5px

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        hours = seconds / 3600
        if hours < 1:
            return f"{hours * 60:.0f}min"
        elif hours < 24:
            return f"{hours:.1f}h"
        else:
            return f"{hours / 24:.1f}d"

    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Overlay abnormal gaps with severity-based colors and line widths.

        Features:
        - Color gradient: yellow (mild) → red (critical)
        - Line width scales with severity
        - Top-N filtering to avoid clutter
        - Grouped by severity category for legend control
        """
        if self.detected is None or not self.detected.get('abnormal_gaps'):
            return fig

        abnormal_gaps = self.detected['abnormal_gaps']

        # Filter by selected transitions if set
        selected_transitions = st.session_state.get('selected_gap_transitions')
        if selected_transitions is not None:
            abnormal_gaps = [g for g in abnormal_gaps if g['transition'] in selected_transitions]

        if not abnormal_gaps:
            return fig

        # Sort by severity and limit to top N
        gaps_sorted = sorted(abnormal_gaps, key=lambda g: g['severity'], reverse=True)
        gaps_to_show = gaps_sorted[:self.MAX_GAPS_TO_DISPLAY]

        # Group gaps by severity category
        severity_groups = {
            'Critical (>5x)': [],
            'Severe (3-5x)': [],
            'Moderate (2-3x)': [],
            'Mild (1-2x)': []
        }

        for gap in gaps_to_show:
            sev = gap['severity']
            if sev >= 5:
                severity_groups['Critical (>5x)'].append(gap)
            elif sev >= 3:
                severity_groups['Severe (3-5x)'].append(gap)
            elif sev >= 2:
                severity_groups['Moderate (2-3x)'].append(gap)
            else:
                severity_groups['Mild (1-2x)'].append(gap)

        # Draw each severity group as separate trace (allows legend toggle)
        group_colors = {
            'Critical (>5x)': 'rgba(139, 0, 0, 0.9)',
            'Severe (3-5x)': 'rgba(220, 53, 69, 0.85)',
            'Moderate (2-3x)': 'rgba(255, 152, 0, 0.8)',
            'Mild (1-2x)': 'rgba(255, 193, 7, 0.7)'
        }

        for group_name, gaps in severity_groups.items():
            if not gaps:
                continue

            x_coords, y_coords, hover_texts = [], [], []
            avg_severity = sum(g['severity'] for g in gaps) / len(gaps)

            for gap in gaps:
                x_coords.extend([gap['x_start'], gap['x_end'], None])
                y_coords.extend([gap['y_value_from'], gap['y_value_to'], None])

                dur_str = self._format_duration(gap['duration'])
                thresh_str = self._format_duration(gap['threshold'])
                hover = (f"<b>{gap['transition']}</b><br>"
                        f"Duration: {dur_str} (threshold: {thresh_str})<br>"
                        f"Severity: {gap['severity']:.1f}x<br>"
                        f"Case: {gap['case_id']}")
                hover_texts.extend([hover, hover, None])

            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                line=dict(
                    color=group_colors[group_name],
                    width=self._severity_to_width(avg_severity),
                    dash='dot'
                ),
                marker=dict(
                    size=6 + avg_severity,
                    color=group_colors[group_name],
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                hoverinfo='text',
                hovertext=hover_texts,
                name=f'{group_name} ({len(gaps)})',
                showlegend=True,
                legendgroup='gaps'
            ))

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
        total_duration = sum(gap['duration']
                             for gap in abnormal_gaps) if abnormal_gaps else 0
        avg_duration = total_duration / \
            len(abnormal_gaps) if abnormal_gaps else 0

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
