"""
Gap Detection for Dotted Charts.

Supports two modes:
- Transition: Case-internal gaps between activities (process flow)
- Resource Inactivity: Resource-timeline gaps (not process flow)
"""

from .pattern_base import Pattern
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List, Optional, Literal


class GapPattern(Pattern):
    """
    Gap detector supporting two modes:
    - TRANSITION (default): Process-aware gaps within cases
    - RESOURCE_INACTIVITY: Resource-timeline gaps (NOT process-flow)

    Both use: Threshold = max(P95, Q3 + 1.5*IQR), Severity = duration/threshold
    """

    MIN_SAMPLES_FOR_NORMALITY = 15
    MAX_GAPS_TO_DISPLAY = 50

    def __init__(
        self,
        view_config: Dict[str, str],
        y_is_categorical: bool = False,
        gap_mode: Literal["transition", "resource_inactivity"] = "transition",
        **kwargs
    ):
        mode_names = {"transition": "Transition Gap Detection", "resource_inactivity": "Resource Inactivity Detection"}
        super().__init__(mode_names.get(gap_mode, "Gap Detection"), view_config)
        self.y_is_categorical = y_is_categorical
        self.gap_mode = gap_mode
        self.detected = None
        self.transition_stats = None
        self.y_categories = None
        self.y_to_index = None

    def _is_time_like(self, x_series: pd.Series, x_col: str) -> bool:
        """Check if X-axis is time-like (required for gap detection)."""
        if pd.api.types.is_datetime64_any_dtype(x_series):
            return True
        return x_col in ["actual_time", "relative_time", "relative_ratio", "logical_time", "logical_relative"]

    def _compute_duration(self, x_start, x_end, x_is_datetime: bool) -> Optional[float]:
        """Compute duration between two X values. Returns None if invalid."""
        if x_is_datetime:
            if not isinstance(x_start, pd.Timestamp):
                x_start = pd.Timestamp(int(x_start) if isinstance(x_start, (int, float)) else x_start)
            if not isinstance(x_end, pd.Timestamp):
                x_end = pd.Timestamp(int(x_end) if isinstance(x_end, (int, float)) else x_end)
            duration = (x_end - x_start).total_seconds()
        else:
            duration = float(x_end - x_start)
        return duration if duration > 0 else None

    def _extract_gaps(self, df: pd.DataFrame, x_col: str, y_col: str, group_col: str) -> List[Dict[str, Any]]:
        """
        Extract gaps between consecutive events within groups.
        Works for both modes - transition (group by case_id) and resource (group by resource).
        """
        if group_col not in df.columns:
            return []

        is_transition_mode = (group_col == 'case_id')
        if is_transition_mode and 'activity' not in df.columns:
            return []

        df_sorted = df.sort_values([group_col, x_col]).copy()
        x_is_datetime = pd.api.types.is_datetime64_any_dtype(df_sorted[x_col])
        gaps = []

        for group_id, group_df in df_sorted.groupby(group_col):
            if len(group_df) < 2:
                continue
            group_df = group_df.reset_index(drop=True)

            for i in range(len(group_df) - 1):
                event_a, event_b = group_df.iloc[i], group_df.iloc[i + 1]
                duration = self._compute_duration(event_a[x_col], event_b[x_col], x_is_datetime)
                if duration is None:
                    continue

                gap = {
                    'x_start': event_a[x_col],
                    'x_end': event_b[x_col],
                    'duration': duration,
                    'y_value_from': event_a[y_col],
                    'y_value_to': event_b[y_col],
                }

                if is_transition_mode:
                    gap.update({
                        'case_id': group_id,
                        'activity_from': event_a['activity'],
                        'activity_to': event_b['activity'],
                        'transition': f"{event_a['activity']} → {event_b['activity']}",
                    })
                else:
                    gap.update({
                        'resource': group_id,
                        'group_key': group_id,
                        'y_value_from': group_id,
                        'y_value_to': group_id,
                        'case_from': event_a.get('case_id', 'N/A'),
                        'case_to': event_b.get('case_id', 'N/A'),
                    })
                gaps.append(gap)

        return gaps

    def _compute_normality_per_group(self, gaps: List[Dict[str, Any]], group_key: str = 'transition') -> Dict[str, Dict[str, float]]:
        """Compute statistical thresholds per group (transition or resource)."""
        group_durations = {}
        for gap in gaps:
            group = gap.get(group_key, gap.get('group_key', 'unknown'))
            if group not in group_durations:
                group_durations[group] = []
            group_durations[group].append(gap['duration'])

        group_stats = {}
        for group, durations in group_durations.items():
            if len(durations) < self.MIN_SAMPLES_FOR_NORMALITY:
                continue
            arr = np.array(durations)
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr = q3 - q1
            p95 = np.percentile(arr, 95)
            group_stats[group] = {
                'count': len(durations),
                'median': np.median(arr),
                'q1': q1, 'q3': q3, 'iqr': iqr, 'p95': p95,
                'threshold': max(p95, q3 + 1.5 * iqr)
            }
        return group_stats

    def _compute_y_position(self, gap: Dict[str, Any], df: pd.DataFrame, y_col: str) -> tuple:
        """Compute visual Y-position for gap visualization."""
        if self.y_is_categorical:
            y_value_from = gap['y_value_from']
            if y_value_from in self.y_to_index:
                idx = self.y_to_index[y_value_from]
                return idx - 0.4, idx + 0.4
            return 0, 1
        return df[y_col].min(), df[y_col].max()

    def detect(self, df: pd.DataFrame) -> None:
        """Detect abnormal gaps based on the configured mode."""
        if df is None or len(df) == 0:
            raise ValueError("Cannot detect gaps: DataFrame is empty")
        if df.empty:
            self.detected = None
            return

        try:
            x_col, y_col = self.view_config['x'], self.view_config['y']
            if x_col not in df.columns or y_col not in df.columns:
                self.detected = None
                return
            if not self._is_time_like(df[x_col], x_col):
                self.detected = None
                return

            if self.y_is_categorical:
                self.y_categories = list(pd.unique(df[y_col]))
                self.y_to_index = {cat: idx for idx, cat in enumerate(self.y_categories)}

            # Extract gaps based on mode
            if self.gap_mode == "resource_inactivity":
                if 'resource' not in df.columns:
                    self.detected = None
                    return
                all_gaps = self._extract_gaps(df, x_col, y_col, 'resource')
                group_key = 'resource'
            else:
                all_gaps = self._extract_gaps(df, x_col, y_col, 'case_id')
                group_key = 'transition'

            if not all_gaps:
                self.detected = None
                return

            self.transition_stats = self._compute_normality_per_group(all_gaps, group_key)
            if not self.transition_stats:
                self.detected = None
                return

            # Identify abnormal gaps
            abnormal_gaps = []
            for gap in all_gaps:
                group = gap.get(group_key, gap.get('group_key', 'unknown'))
                if group not in self.transition_stats:
                    continue
                threshold = self.transition_stats[group]['threshold']
                if gap['duration'] > threshold:
                    y_low, y_high = self._compute_y_position(gap, df, y_col)
                    abnormal_gap = {**gap, 'threshold': threshold, 'severity': gap['duration'] / threshold,
                                    'y_low': y_low, 'y_high': y_high, 'group_key': group}
                    abnormal_gaps.append(abnormal_gap)

            if not abnormal_gaps:
                self.detected = None
                return

            total_groups = len(self.transition_stats)
            groups_with_anomalies = len(set(g['group_key'] for g in abnormal_gaps))

            self.detected = {
                'gap_mode': self.gap_mode,
                'total_gaps': len(all_gaps),
                'total_abnormal_gaps': len(abnormal_gaps),
                'total_groups': total_groups,
                'groups_with_anomalies': groups_with_anomalies,
                'total_transitions': total_groups if self.gap_mode == 'transition' else 0,
                'transitions_with_anomalies': groups_with_anomalies if self.gap_mode == 'transition' else 0,
                'total_resources': total_groups if self.gap_mode == 'resource_inactivity' else 0,
                'resources_with_anomalies': groups_with_anomalies if self.gap_mode == 'resource_inactivity' else 0,
                'abnormal_gaps': abnormal_gaps,
                'transition_stats': self.transition_stats,
                'group_stats': self.transition_stats
            }
        except Exception:
            self.detected = None
            raise

    @staticmethod
    def _severity_to_color(severity: float) -> str:
        if severity < 2: return 'rgba(255, 193, 7, 0.7)'
        elif severity < 3: return 'rgba(255, 152, 0, 0.8)'
        elif severity < 5: return 'rgba(220, 53, 69, 0.85)'
        else: return 'rgba(139, 0, 0, 0.9)'

    @staticmethod
    def _severity_to_width(severity: float) -> float:
        return min(1.5 + severity * 0.4, 5)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        hours = seconds / 3600
        if hours < 1: return f"{hours * 60:.0f}min"
        elif hours < 24: return f"{hours:.1f}h"
        else: return f"{hours / 24:.1f}d"

    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """Overlay abnormal gaps with severity-based colors."""
        if self.detected is None or not self.detected.get('abnormal_gaps'):
            return fig

        abnormal_gaps = self.detected['abnormal_gaps']
        gap_mode = self.detected.get('gap_mode', 'transition')

        # Filter by selected items
        if gap_mode == 'resource_inactivity':
            selected = st.session_state.get('selected_gap_resources')
            if selected is not None:
                abnormal_gaps = [g for g in abnormal_gaps if g['resource'] in selected]
        else:
            selected = st.session_state.get('selected_gap_transitions')
            if selected is not None:
                abnormal_gaps = [g for g in abnormal_gaps if g.get('transition') in selected]

        if not abnormal_gaps:
            return fig

        gaps_sorted = sorted(abnormal_gaps, key=lambda g: g['severity'], reverse=True)
        gaps_to_show = gaps_sorted[:self.MAX_GAPS_TO_DISPLAY]

        severity_groups = {'Critical (>5x)': [], 'Severe (3-5x)': [], 'Moderate (2-3x)': [], 'Mild (1-2x)': []}
        for gap in gaps_to_show:
            sev = gap['severity']
            if sev >= 5: severity_groups['Critical (>5x)'].append(gap)
            elif sev >= 3: severity_groups['Severe (3-5x)'].append(gap)
            elif sev >= 2: severity_groups['Moderate (2-3x)'].append(gap)
            else: severity_groups['Mild (1-2x)'].append(gap)

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
                dur_str, thresh_str = self._format_duration(gap['duration']), self._format_duration(gap['threshold'])

                if gap_mode == 'resource_inactivity':
                    hover = f"<b>Resource: {gap['resource']}</b><br>Inactivity: {dur_str} (threshold: {thresh_str})<br>Severity: {gap['severity']:.1f}x<br><i>Not a process-flow gap</i>"
                else:
                    hover = f"<b>{gap.get('transition', 'N/A')}</b><br>Duration: {dur_str} (threshold: {thresh_str})<br>Severity: {gap['severity']:.1f}x<br>Case: {gap.get('case_id', 'N/A')}"
                hover_texts.extend([hover, hover, None])

            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords, mode='lines+markers',
                line=dict(color=group_colors[group_name], width=self._severity_to_width(avg_severity), dash='dot'),
                marker=dict(size=6 + avg_severity, color=group_colors[group_name], symbol='circle', line=dict(color='white', width=1)),
                hoverinfo='text', hovertext=hover_texts,
                name=f'{group_name} ({len(gaps)})', showlegend=True,
                legendgroup='resource_gaps' if gap_mode == 'resource_inactivity' else 'transition_gaps'
            ))
        return fig

    def get_gap_summary(self) -> Dict[str, Any]:
        """Get summary of detected abnormal gaps."""
        if self.detected is None:
            return {
                'gap_mode': self.gap_mode, 'total_gaps': 0, 'total_abnormal_gaps': 0,
                'total_groups': 0, 'groups_with_anomalies': 0,
                'total_transitions': 0, 'transitions_with_anomalies': 0,
                'total_resources': 0, 'resources_with_anomalies': 0,
                'total_magnitude': 0, 'average_magnitude': 0,
                'abnormal_gaps': [], 'gaps': [], 'transition_stats': {}, 'group_stats': {}
            }

        abnormal_gaps = self.detected['abnormal_gaps']
        total_duration = sum(g['duration'] for g in abnormal_gaps) if abnormal_gaps else 0
        avg_duration = total_duration / len(abnormal_gaps) if abnormal_gaps else 0

        return {
            'gap_mode': self.detected.get('gap_mode', 'transition'),
            'total_gaps': self.detected['total_gaps'],
            'total_abnormal_gaps': self.detected['total_abnormal_gaps'],
            'total_groups': self.detected.get('total_groups', 0),
            'groups_with_anomalies': self.detected.get('groups_with_anomalies', 0),
            'total_transitions': self.detected.get('total_transitions', 0),
            'transitions_with_anomalies': self.detected.get('transitions_with_anomalies', 0),
            'total_resources': self.detected.get('total_resources', 0),
            'resources_with_anomalies': self.detected.get('resources_with_anomalies', 0),
            'total_magnitude': total_duration, 'average_magnitude': avg_duration,
            'abnormal_gaps': abnormal_gaps, 'gaps': abnormal_gaps,
            'transition_stats': self.detected.get('transition_stats', {}),
            'group_stats': self.detected.get('group_stats', {})
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get standardized pattern summary."""
        gap_summary = self.get_gap_summary()
        return {
            'pattern_type': 'gap',
            'detected': self.detected is not None and gap_summary['total_abnormal_gaps'] > 0,
            'count': gap_summary['total_abnormal_gaps'],
            'details': gap_summary
        }
