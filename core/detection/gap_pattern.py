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
from typing import Dict, Any, List, Optional, Literal


class GapPattern(Pattern):
    """
    Gap detector supporting two distinct modes:

    1. TRANSITION mode (default): Process-aware gaps within cases
       - Extracts gaps between consecutive events within a case
       - Learns normal duration per transition (Activity A → Activity B)
       - Detects abnormal delays in process flow

    2. RESOURCE_INACTIVITY mode: Resource-timeline gaps
       - Extracts gaps between consecutive events per resource
       - Learns normal activity frequency per resource
       - Detects periods of unusual resource inactivity
       - NOT a process-flow gap - purely resource availability analysis

    Both modes use the same statistical approach:
    - Threshold: max(P95, Q3 + 1.5*IQR)
    - Severity: duration / threshold
    """

    MIN_SAMPLES_FOR_NORMALITY = 15  # Minimum samples for stable threshold estimation
    MAX_GAPS_TO_DISPLAY = 50  # Limit visualization to top N gaps by severity

    def __init__(
        self,
        view_config: Dict[str, str],
        y_is_categorical: bool = False,
        gap_mode: Literal["transition", "resource_inactivity"] = "transition",
        **kwargs
    ):
        """
        Initialize gap detector.

        Parameters
        ----------
        view_config : dict
            Configuration with "x" and "y" keys for chart dimensions
        y_is_categorical : bool, default False
            Whether Y-axis is categorical
        gap_mode : str, default "transition"
            Detection mode:
            - "transition": Case-internal gaps between activities (process flow)
            - "resource_inactivity": Resource-timeline gaps (not process flow)
        """
        mode_names = {
            "transition": "Transition Gap Detection",
            "resource_inactivity": "Resource Inactivity Detection"
        }
        super().__init__(mode_names.get(gap_mode, "Gap Detection"), view_config)
        self.y_is_categorical = y_is_categorical
        self.gap_mode = gap_mode
        self.detected = None
        self.transition_stats = None  # Used for both modes (keyed by transition or resource)
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

    def _extract_resource_gaps(
        self,
        df: pd.DataFrame,
        x_col: str
    ) -> List[Dict[str, Any]]:
        """
        Extract gaps within resource timelines (NOT process-flow gaps).

        Groups events by resource and finds periods of inactivity.
        This is fundamentally different from transition gaps:
        - Transition gaps: delays within a case's process flow
        - Resource gaps: periods when a resource has no events

        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe with 'resource' column
        x_col : str
            X-axis column name (time-like)

        Returns
        -------
        list of dict
            List of resource inactivity gaps
        """
        if 'resource' not in df.columns:
            return []

        # Sort by resource and time
        df_sorted = df.sort_values(['resource', x_col]).copy()

        # Check if X is datetime
        x_is_datetime = pd.api.types.is_datetime64_any_dtype(df_sorted[x_col])

        gaps = []

        # Group by resource (NOT by case)
        for resource, resource_df in df_sorted.groupby('resource'):
            resource_df = resource_df.reset_index(drop=True)

            if len(resource_df) < 2:
                continue

            # Extract consecutive event pairs within this resource's timeline
            for i in range(len(resource_df) - 1):
                event_a = resource_df.iloc[i]
                event_b = resource_df.iloc[i + 1]

                x_start = event_a[x_col]
                x_end = event_b[x_col]

                # Calculate duration
                if x_is_datetime:
                    if not isinstance(x_start, pd.Timestamp):
                        x_start = pd.Timestamp(int(x_start) if isinstance(
                            x_start, (int, float)) else x_start)
                    if not isinstance(x_end, pd.Timestamp):
                        x_end = pd.Timestamp(int(x_end) if isinstance(
                            x_end, (int, float)) else x_end)
                    duration = (x_end - x_start).total_seconds()
                else:
                    duration = float(x_end - x_start)

                # Skip negative or zero durations
                if duration <= 0:
                    continue

                # Get case_ids for context (events may be from different cases)
                case_from = event_a.get('case_id', 'N/A')
                case_to = event_b.get('case_id', 'N/A')

                gaps.append({
                    'resource': resource,
                    'group_key': resource,  # Uniform key for normality computation
                    'x_start': x_start,
                    'x_end': x_end,
                    'y_value_from': resource,
                    'y_value_to': resource,
                    'duration': duration,
                    'case_from': case_from,
                    'case_to': case_to,
                    # No transition info - this is NOT a process-flow gap
                })

        return gaps

    def _compute_normality_per_group(
        self,
        gaps: List[Dict[str, Any]],
        group_key: str = 'transition'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical normality thresholds per group.

        Works for both modes:
        - Transition mode: groups by 'transition' (Activity A → Activity B)
        - Resource inactivity mode: groups by 'resource'

        Only computes thresholds for groups with >= MIN_SAMPLES_FOR_NORMALITY.

        Parameters
        ----------
        gaps : list of dict
            List of gaps with group information
        group_key : str
            Key to group by ('transition' or 'resource')

        Returns
        -------
        dict
            Mapping from group to statistics:
            {
                'group_name': {
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
        # Group gaps by the specified key
        group_durations = {}
        for gap in gaps:
            group = gap.get(group_key, gap.get('group_key', 'unknown'))
            if group not in group_durations:
                group_durations[group] = []
            group_durations[group].append(gap['duration'])

        # Compute statistics per group
        group_stats = {}

        for group, durations in group_durations.items():
            durations_array = np.array(durations)
            count = len(durations)

            # Skip groups with insufficient samples
            if count < self.MIN_SAMPLES_FOR_NORMALITY:
                continue

            median = np.median(durations_array)
            q1 = np.percentile(durations_array, 25)
            q3 = np.percentile(durations_array, 75)
            iqr = q3 - q1
            p95 = np.percentile(durations_array, 95)

            # Compute threshold: max(P95, Q3 + 1.5*IQR)
            threshold = max(p95, q3 + 1.5 * iqr)

            group_stats[group] = {
                'count': count,
                'median': median,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'p95': p95,
                'threshold': threshold
            }

        return group_stats

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
        Detect abnormal gaps based on the configured mode.

        Modes:
        - transition: Case-internal gaps between activities (process flow)
        - resource_inactivity: Resource-timeline gaps (NOT process flow)

        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        """
        # Validate DataFrame is not empty
        if df is None or len(df) == 0:
            raise ValueError("Cannot detect gaps: DataFrame is empty")

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

            # Extract gaps based on mode
            if self.gap_mode == "resource_inactivity":
                # Resource inactivity mode: requires 'resource' column
                if 'resource' not in df.columns:
                    self.detected = None
                    return
                all_gaps = self._extract_resource_gaps(df, x_col)
                group_key = 'resource'
            else:
                # Transition mode (default): requires case_id and activity
                all_gaps = self._extract_transition_gaps(df, x_col, y_col)
                group_key = 'transition'

            if not all_gaps:
                self.detected = None
                return

            # Compute normality per group (transition or resource)
            self.transition_stats = self._compute_normality_per_group(
                all_gaps, group_key)

            if not self.transition_stats:
                # No groups with sufficient samples
                self.detected = None
                return

            # Identify abnormal gaps
            abnormal_gaps = []

            for gap in all_gaps:
                group = gap.get(group_key, gap.get('group_key', 'unknown'))

                # Skip groups without computed thresholds
                if group not in self.transition_stats:
                    continue

                duration = gap['duration']
                threshold = self.transition_stats[group]['threshold']

                if duration > threshold:
                    # Compute severity
                    severity = duration / threshold

                    # Compute Y position for visualization
                    y_low, y_high = self._compute_y_position(gap, df, y_col)

                    # Build abnormal gap structure (mode-specific fields)
                    if self.gap_mode == "resource_inactivity":
                        abnormal_gap = {
                            'resource': gap['resource'],
                            'group_key': group,
                            'x_start': gap['x_start'],
                            'x_end': gap['x_end'],
                            'duration': duration,
                            'threshold': threshold,
                            'severity': severity,
                            'y_low': y_low,
                            'y_high': y_high,
                            'y_value_from': gap['y_value_from'],
                            'y_value_to': gap['y_value_to'],
                            'case_from': gap.get('case_from', 'N/A'),
                            'case_to': gap.get('case_to', 'N/A'),
                        }
                    else:
                        abnormal_gap = {
                            'case_id': gap['case_id'],
                            'transition': gap['transition'],
                            'group_key': group,
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

            # Build result summary (mode-specific labels)
            total_gaps = len(all_gaps)
            total_abnormal = len(abnormal_gaps)
            total_groups = len(self.transition_stats)
            groups_with_anomalies = len(
                set(g['group_key'] for g in abnormal_gaps))

            self.detected = {
                'gap_mode': self.gap_mode,
                'total_gaps': total_gaps,
                'total_abnormal_gaps': total_abnormal,
                'total_groups': total_groups,
                'groups_with_anomalies': groups_with_anomalies,
                # Mode-specific aliases for backward compatibility
                'total_transitions': total_groups if self.gap_mode == 'transition' else 0,
                'transitions_with_anomalies': groups_with_anomalies if self.gap_mode == 'transition' else 0,
                'total_resources': total_groups if self.gap_mode == 'resource_inactivity' else 0,
                'resources_with_anomalies': groups_with_anomalies if self.gap_mode == 'resource_inactivity' else 0,
                'abnormal_gaps': abnormal_gaps,
                'transition_stats': self.transition_stats,  # Kept for backward compat
                'group_stats': self.transition_stats  # New generic name
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
        - Mode-aware: different hover text for transition vs resource_inactivity
        """
        if self.detected is None or not self.detected.get('abnormal_gaps'):
            return fig

        abnormal_gaps = self.detected['abnormal_gaps']
        gap_mode = self.detected.get('gap_mode', 'transition')

        # Filter by selected items based on mode
        if gap_mode == 'resource_inactivity':
            selected_resources = st.session_state.get('selected_gap_resources')
            if selected_resources is not None:
                abnormal_gaps = [g for g in abnormal_gaps if g['resource'] in selected_resources]
        else:
            selected_transitions = st.session_state.get('selected_gap_transitions')
            if selected_transitions is not None:
                abnormal_gaps = [g for g in abnormal_gaps if g.get('transition') in selected_transitions]

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

                # Mode-specific hover text
                if gap_mode == 'resource_inactivity':
                    hover = (f"<b>Resource: {gap['resource']}</b><br>"
                            f"Inactivity: {dur_str} (threshold: {thresh_str})<br>"
                            f"Severity: {gap['severity']:.1f}x<br>"
                            f"<i>Not a process-flow gap</i>")
                else:
                    hover = (f"<b>{gap.get('transition', 'N/A')}</b><br>"
                            f"Duration: {dur_str} (threshold: {thresh_str})<br>"
                            f"Severity: {gap['severity']:.1f}x<br>"
                            f"Case: {gap.get('case_id', 'N/A')}")
                hover_texts.extend([hover, hover, None])

            # Mode-specific legend group
            legend_group = 'resource_gaps' if gap_mode == 'resource_inactivity' else 'transition_gaps'

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
                legendgroup=legend_group
            ))

        return fig

    def get_gap_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected abnormal gaps.

        Returns mode-specific labels for UI display.

        Returns
        -------
        dict
            Summary dictionary with gap statistics and mode-specific info
        """
        if self.detected is None:
            return {
                'gap_mode': self.gap_mode,
                'total_gaps': 0,
                'total_abnormal_gaps': 0,
                'total_groups': 0,
                'groups_with_anomalies': 0,
                # Backward compat
                'total_transitions': 0,
                'transitions_with_anomalies': 0,
                'total_resources': 0,
                'resources_with_anomalies': 0,
                'total_magnitude': 0,
                'average_magnitude': 0,
                'abnormal_gaps': [],
                'gaps': [],
                'transition_stats': {},
                'group_stats': {}
            }

        # Calculate total and average duration for UI display
        abnormal_gaps = self.detected['abnormal_gaps']
        total_duration = sum(gap['duration']
                             for gap in abnormal_gaps) if abnormal_gaps else 0
        avg_duration = total_duration / \
            len(abnormal_gaps) if abnormal_gaps else 0

        gap_mode = self.detected.get('gap_mode', 'transition')

        return {
            'gap_mode': gap_mode,
            'total_gaps': self.detected['total_gaps'],
            'total_abnormal_gaps': self.detected['total_abnormal_gaps'],
            'total_groups': self.detected.get('total_groups', 0),
            'groups_with_anomalies': self.detected.get('groups_with_anomalies', 0),
            # Mode-specific aliases
            'total_transitions': self.detected.get('total_transitions', 0),
            'transitions_with_anomalies': self.detected.get('transitions_with_anomalies', 0),
            'total_resources': self.detected.get('total_resources', 0),
            'resources_with_anomalies': self.detected.get('resources_with_anomalies', 0),
            'total_magnitude': total_duration,
            'average_magnitude': avg_duration,
            'abnormal_gaps': abnormal_gaps,
            'gaps': abnormal_gaps,  # Backward compatibility
            'transition_stats': self.detected.get('transition_stats', {}),
            'group_stats': self.detected.get('group_stats', {})
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
