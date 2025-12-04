"""
Outlier detection for event logs in dotted charts.
Detects various types of outliers in process mining data.
Handles missing columns gracefully.
"""

import pandas as pd
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from .pattern_base import Pattern
from config.extended_pattern_matrix import is_pattern_meaningful


class OutlierDetectionPattern(Pattern):
    """
    Detects outliers in event logs for dotted chart visualization.

    Automatically adapts to available columns in the event log.
    Works with minimal data: case_id, activity, and any time column.
    """

    def __init__(self, df: pd.DataFrame, view_config: Dict[str, str]):
        super().__init__("Outlier Detection", view_config)
        self.df = df
        self.outliers = {}
        self.outlier_scores = {}
        self.statistics = {}
        self.available_columns = set(df.columns.str.lower())

    def _has_column(self, *column_names: str) -> Optional[str]:
        """Check if any of the column names exist (case-insensitive)."""
        for col_name in column_names:
            if col_name.lower() in self.available_columns:
                # Find the actual column name with correct case
                for actual_col in self.df.columns:
                    if actual_col.lower() == col_name.lower():
                        return actual_col
        return None

    def detect(self) -> bool:
        """Detect various types of outliers in the event log."""
        # Check if outlier detection is meaningful for this view
        x_axis = self.view_config.get('x', '')
        y_axis = self.view_config.get('y', '')
        
        if not is_pattern_meaningful(x_axis, y_axis, 'outlier'):
            return False
        
        try:
            detection_count = 0

            # 1. Time-based outliers (if time data available)
            if self._detect_time_outliers():
                detection_count += 1

            # 2. Case duration outliers (if time data available)
            if self._detect_case_duration_outliers():
                detection_count += 1

            # 3. Activity frequency outliers (always possible)
            if self._detect_activity_frequency_outliers():
                detection_count += 1

            # 4. Resource behavior outliers (if resource column exists)
            if self._detect_resource_outliers():
                detection_count += 1

            # 5. Event sequence outliers (always possible with case_id + activity)
            if self._detect_sequence_outliers():
                detection_count += 1

            # 6. Case complexity outliers (always possible with case_id)
            if self._detect_case_complexity_outliers():
                detection_count += 1

            # Combine all outliers
            self._combine_outliers()

            # Safety check - if we have more than 10% outliers, something is wrong
            # Apply additional filtering to keep only the most extreme outliers
            total_outliers = len(self.outliers.get('combined', []))
            outlier_percentage = (
                total_outliers / len(self.df)) * 100 if len(self.df) > 0 else 0

            if outlier_percentage > 10:  # More than 10% is too much
                self._filter_extreme_outliers()

            self.detected = len(self.outliers.get('combined', [])) > 0
            self._calculate_statistics(detection_count)

            return self.detected

        except Exception as e:
            print(f"Error in outlier detection: {e}")
            self.detected = False
            return False

    def _detect_time_outliers(self) -> bool:
        """Detect events that occur at unusual times."""
        time_col = self._has_column(
            'actual_time', 'timestamp', 'time', 'start_time', 'complete_time')
        if not time_col:
            return False

        try:
            df_time = self.df.copy()

            # Try to convert to datetime
            df_time[time_col] = pd.to_datetime(
                df_time[time_col], errors='coerce')

            # Remove rows where time conversion failed
            df_time = df_time.dropna(subset=[time_col])
            if df_time.empty:
                return False

            df_time['hour'] = df_time[time_col].dt.hour
            df_time['day_of_week'] = df_time[time_col].dt.dayofweek

            time_outliers = []

            if df_time['hour'].nunique() > 10:  # Need more hour variety
                hour_counts = df_time['hour'].value_counts()
                if len(hour_counts) > 10:
                    # Only consider extremely rare hours (bottom 5%)
                    rare_threshold = max(1, hour_counts.quantile(0.05))
                    rare_hours = hour_counts[hour_counts <=
                                             rare_threshold].index
                    time_outliers.extend(
                        df_time[df_time['hour'].isin(rare_hours)].index.tolist())

            self.outliers['time'] = list(set(time_outliers))
            return len(time_outliers) > 0

        except Exception as e:
            print(f"Time outlier detection failed: {e}")
            return False

    def _detect_case_duration_outliers(self) -> bool:
        """Detect cases with extremely long or short durations."""
        time_col = self._has_column(
            'actual_time', 'timestamp', 'time', 'start_time', 'complete_time')
        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id')

        if not time_col or not case_col:
            return False

        try:
            df_duration = self.df.copy()
            df_duration[time_col] = pd.to_datetime(
                df_duration[time_col], errors='coerce')
            df_duration = df_duration.dropna(subset=[time_col])

            if df_duration.empty:
                return False

            case_stats = df_duration.groupby(case_col)[time_col].agg(
                ['min', 'max', 'count']).reset_index()
            case_stats.columns = [
                case_col, 'start_time', 'end_time', 'event_count']

            # Calculate duration
            case_stats['duration'] = case_stats['end_time'] - \
                case_stats['start_time']
            case_stats['duration_seconds'] = case_stats['duration'].dt.total_seconds()

            # Filter out cases with only one event (no duration)
            case_stats = case_stats[case_stats['event_count'] > 1]

            if case_stats.empty or case_stats['duration_seconds'].nunique() < 5:
                return False

            # Use much more strict IQR method - only extreme outliers (3 * IQR)
            Q1 = case_stats['duration_seconds'].quantile(0.25)
            Q3 = case_stats['duration_seconds'].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:  # All durations are the same
                return False

            # Much more strict - use 3 * IQR instead of 1.5 * IQR
            outlier_cases = case_stats[
                (case_stats['duration_seconds'] < Q1 - 3.0 * IQR) |
                (case_stats['duration_seconds'] > Q3 + 3.0 * IQR)
            ][case_col].tolist()

            if outlier_cases:
                duration_outliers = self.df[self.df[case_col].isin(
                    outlier_cases)].index.tolist()
                self.outliers['case_duration'] = duration_outliers
                return True

        except Exception as e:
            print(f"Duration outlier detection failed: {e}")

        return False

    def _detect_activity_frequency_outliers(self) -> bool:
        """Detect events with rare activities."""
        activity_col = self._has_column(
            'activity', 'event_name', 'activity_name')
        if not activity_col:
            return False

        try:
            activity_counts = self.df[activity_col].value_counts()

            # Much more strict - activities that occur less than 0.1% of total events
            total_events = len(self.df)
            rare_threshold = max(1, total_events * 0.01)

            rare_activities = activity_counts[activity_counts < rare_threshold].index.tolist(
            )

            if rare_activities:
                activity_outliers = self.df[self.df[activity_col].isin(
                    rare_activities)].index.tolist()
                self.outliers['activity_frequency'] = activity_outliers
                return True

        except Exception as e:
            print(f"Activity frequency outlier detection failed: {e}")

        return False

    def _detect_resource_outliers(self) -> bool:
        """Detect unusual resource behavior."""
        resource_col = self._has_column(
            'resource', 'org:resource', 'user', 'performer', 'resource_name')
        if not resource_col:
            return False

        try:
            # Resource workload outliers
            resource_counts = self.df[resource_col].value_counts()

            # Need at least 5 different resources for meaningful outlier detection
            if len(resource_counts) < 5:
                return False

            # Much more strict - use 3 * IQR for extreme outliers only
            Q1 = resource_counts.quantile(0.25)
            Q3 = resource_counts.quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:  # All resources have same workload
                return False

            outlier_resources = resource_counts[
                (resource_counts < Q1 - 3.0 * IQR) |
                (resource_counts > Q3 + 3.0 * IQR)
            ].index.tolist()

            if outlier_resources:
                resource_outliers = self.df[self.df[resource_col].isin(
                    outlier_resources)].index.tolist()
                self.outliers['resource'] = resource_outliers
                return True

        except Exception as e:
            print(f"Resource outlier detection failed: {e}")

        return False

    def _detect_sequence_outliers(self) -> bool:
        """Detect unusual activity sequences within cases."""
        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id')
        activity_col = self._has_column(
            'activity', 'concept:name', 'event_name', 'activity_name')
        time_col = self._has_column(
            'actual_time', 'timestamp', 'time', 'start_time', 'complete_time')

        if not case_col or not activity_col:
            return False

        try:
            # Sort by time if available, otherwise by index
            df_sorted = self.df.copy()
            if time_col:
                df_sorted[time_col] = pd.to_datetime(
                    df_sorted[time_col], errors='coerce')
                df_sorted = df_sorted.sort_values([case_col, time_col])
            else:
                df_sorted = df_sorted.sort_values(case_col)

            # Create activity sequences for each case
            case_sequences = df_sorted.groupby(
                case_col)[activity_col].apply(list).to_dict()

            # Find rare transitions
            transitions = {}
            for case_id, activities in case_sequences.items():
                for i in range(len(activities) - 1):
                    transition = (activities[i], activities[i + 1])
                    transitions[transition] = transitions.get(
                        transition, 0) + 1

            if not transitions or len(transitions) < 10:  # Need more transitions
                return False

            # Much more strict - bottom 1% of all transitions only
            transition_counts = pd.Series(transitions)
            rare_threshold = max(1, transition_counts.quantile(
                0.01))  # Bottom 1% instead of 5%
            rare_transitions = transition_counts[transition_counts <= rare_threshold].index.tolist(
            )

            if not rare_transitions:
                return False

            # Find events involved in rare transitions
            sequence_outliers = []
            for case_id, activities in case_sequences.items():
                case_events = df_sorted[df_sorted[case_col] == case_id]
                for i in range(len(activities) - 1):
                    transition = (activities[i], activities[i + 1])
                    if transition in rare_transitions:
                        # Add both events in the rare transition
                        if i < len(case_events) - 1:
                            sequence_outliers.extend(
                                case_events.iloc[i:i+2].index.tolist())

            if sequence_outliers:
                self.outliers['sequence'] = list(set(sequence_outliers))
                return True

        except Exception as e:
            print(f"Sequence outlier detection failed: {e}")

        return False

    def _detect_case_complexity_outliers(self) -> bool:
        """Detect cases with unusual complexity patterns."""
        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id')
        activity_col = self._has_column(
            'activity', 'concept:name', 'event_name', 'activity_name')
        time_col = self._has_column(
            'actual_time', 'timestamp', 'time', 'start_time', 'complete_time')

        if not case_col:
            return False

        try:
            # Calculate case complexity metrics
            case_metrics = {}

            for case_id in self.df[case_col].unique():
                case_events = self.df[self.df[case_col] == case_id]

                # Metric 1: Number of events per case
                event_count = len(case_events)

                # Metric 2: Number of unique activities (if activity column exists)
                unique_activities = len(
                    case_events[activity_col].unique()) if activity_col else event_count

                # Metric 3: Time span of case (if time column exists)
                time_span = 0
                if time_col:
                    case_times = pd.to_datetime(
                        case_events[time_col], errors='coerce').dropna()
                    if len(case_times) > 1:
                        time_span = (case_times.max() - case_times.min()
                                     ).total_seconds() / 3600  # hours

                case_metrics[case_id] = {
                    'event_count': event_count,
                    'unique_activities': unique_activities,
                    'time_span_hours': time_span
                }

            if len(case_metrics) < 10:  # Need enough cases for meaningful outlier detection
                return False

            complexity_outliers = []

            # Find outliers based on event count
            event_counts = [metrics['event_count']
                            for metrics in case_metrics.values()]
            if len(set(event_counts)) > 1:  # Not all cases have same event count
                event_counts_series = pd.Series(event_counts)
                Q1 = event_counts_series.quantile(0.25)
                Q3 = event_counts_series.quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:  # Avoid division by zero
                    # Cases with too many or too few events (3×IQR method)
                    outlier_threshold_low = Q1 - 3.0 * IQR
                    outlier_threshold_high = Q3 + 3.0 * IQR

                    for case_id, metrics in case_metrics.items():
                        if (metrics['event_count'] < outlier_threshold_low or
                                metrics['event_count'] > outlier_threshold_high):
                            # Add all events from this complex case
                            case_events = self.df[self.df[case_col] == case_id]
                            complexity_outliers.extend(
                                case_events.index.tolist())

            # Find outliers based on activity diversity (if activity data available)
            if activity_col:
                activity_diversity = [metrics['unique_activities']
                                      for metrics in case_metrics.values()]
                if len(set(activity_diversity)) > 1:
                    diversity_series = pd.Series(activity_diversity)
                    Q1_div = diversity_series.quantile(0.25)
                    Q3_div = diversity_series.quantile(0.75)
                    IQR_div = Q3_div - Q1_div

                    if IQR_div > 0:
                        # Cases with unusually high/low activity diversity
                        outlier_threshold_low_div = Q1_div - 2.0 * IQR_div
                        outlier_threshold_high_div = Q3_div + 2.0 * IQR_div

                        for case_id, metrics in case_metrics.items():
                            if (metrics['unique_activities'] < outlier_threshold_low_div or
                                    metrics['unique_activities'] > outlier_threshold_high_div):
                                case_events = self.df[self.df[case_col]
                                                      == case_id]
                                complexity_outliers.extend(
                                    case_events.index.tolist())

            # Find outliers based on time span (if time data available)
            if time_col:
                time_spans = [metrics['time_span_hours'] for metrics in case_metrics.values()
                              if metrics['time_span_hours'] > 0]
                if len(time_spans) > 5 and len(set(time_spans)) > 1:
                    time_spans_series = pd.Series(time_spans)
                    Q1_time = time_spans_series.quantile(0.25)
                    Q3_time = time_spans_series.quantile(0.75)
                    IQR_time = Q3_time - Q1_time

                    if IQR_time > 0:
                        # Cases with unusually long/short duration
                        outlier_threshold_low_time = Q1_time - 2.5 * IQR_time
                        outlier_threshold_high_time = Q3_time + 2.5 * IQR_time

                        for case_id, metrics in case_metrics.items():
                            if (metrics['time_span_hours'] > 0 and
                                (metrics['time_span_hours'] < outlier_threshold_low_time or
                                 metrics['time_span_hours'] > outlier_threshold_high_time)):
                                case_events = self.df[self.df[case_col]
                                                      == case_id]
                                complexity_outliers.extend(
                                    case_events.index.tolist())

            # Remove duplicates and store results
            if complexity_outliers:
                self.outliers['case_complexity'] = list(
                    set(complexity_outliers))
                return True

        except Exception as e:
            print(f"Case complexity outlier detection failed: {e}")

        return False

    def _combine_outliers(self):
        """Combine all types of outliers and calculate composite scores."""
        all_outliers = set()
        outlier_types = {}

        for outlier_type, indices in self.outliers.items():
            all_outliers.update(indices)
            for idx in indices:
                if idx not in outlier_types:
                    outlier_types[idx] = []
                outlier_types[idx].append(outlier_type)

        self.outliers['combined'] = list(all_outliers)
        self.outlier_types = outlier_types

        # Calculate outlier scores (number of different outlier types)
        self.outlier_scores = {idx: len(types)
                               for idx, types in outlier_types.items()}

    def _filter_extreme_outliers(self):
        """Keep only the most extreme outliers when we have too many."""
        if not self.outlier_scores:
            return

        # Sort by outlier score (number of detection methods that flagged this event)
        sorted_outliers = sorted(
            self.outlier_scores.items(), key=lambda x: x[1], reverse=True)

        # Keep only top 5% of events or those detected by multiple methods
        # At most 5% or minimum 10 events
        max_outliers = max(10, len(self.df) * 0.05)

        # Filter to keep only high-confidence outliers
        filtered_outliers = []
        for idx, score in sorted_outliers:
            if len(filtered_outliers) >= max_outliers:
                break
            # Keep if detected by multiple methods or in top outliers
            if score > 1 or len(filtered_outliers) < max_outliers / 2:
                filtered_outliers.append(idx)

        # Update all outlier collections
        self.outliers['combined'] = filtered_outliers

        # Update individual outlier type collections to only include filtered outliers
        filtered_set = set(filtered_outliers)
        for outlier_type in list(self.outliers.keys()):
            if outlier_type != 'combined':
                self.outliers[outlier_type] = [
                    idx for idx in self.outliers[outlier_type] if idx in filtered_set]

        # Update outlier types and scores
        self.outlier_types = {
            idx: types for idx, types in self.outlier_types.items() if idx in filtered_set}
        self.outlier_scores = {
            idx: score for idx, score in self.outlier_scores.items() if idx in filtered_set}

    def _calculate_statistics(self, detection_methods_used: int):
        """Calculate outlier detection statistics."""
        total_events = len(self.df)

        # Get available columns info
        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id')
        total_cases = self.df[case_col].nunique() if case_col else 0

        outlier_cases = 0
        if case_col and self.outliers.get('combined'):
            outlier_cases = len(
                set(self.df.loc[self.outliers['combined'], case_col].tolist()))

        self.statistics = {
            'total_events': total_events,
            'total_outliers': len(self.outliers.get('combined', [])),
            'outlier_percentage': (len(self.outliers.get('combined', [])) / total_events * 100) if total_events > 0 else 0,
            'outlier_types': {
                outlier_type: len(indices)
                for outlier_type, indices in self.outliers.items()
                if outlier_type != 'combined'
            },
            'max_outlier_score': max(self.outlier_scores.values()) if self.outlier_scores else 0,
            'cases_with_outliers': outlier_cases,
            'total_cases': total_cases,
            'detection_methods_used': detection_methods_used,
            'available_features': list(self.available_columns)
        }

    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Add outlier visualization to the existing figure.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe (for consistency with Pattern API)
        fig : go.Figure
            Plotly figure to annotate
            
        Returns
        -------
        go.Figure
            Figure with outlier visualization added
        """
        if not self.detected:
            return fig

        # Get outlier data - filter to only maximum score outliers
        all_outlier_indices = self.outliers.get('combined', [])
        if not all_outlier_indices:
            return fig

        # Find maximum score and filter to only those outliers
        max_score = max(self.outlier_scores.values()
                        ) if self.outlier_scores else 0
        if max_score <= 1:
            return fig

        # Only show outliers with maximum score (detected by most methods)
        max_score_indices = [
            idx for idx in all_outlier_indices if self.outlier_scores.get(idx, 0) == max_score]

        if not max_score_indices:
            return fig

        outlier_data = self.df.loc[max_score_indices]

        # Get column names for display
        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id') or 'case_id'
        activity_col = self._has_column(
            'activity', 'concept:name', 'event_name', 'activity_name') or 'activity'

        # Create hover texts for maximum score outliers only
        hover_texts = []
        for idx in max_score_indices:
            score = self.outlier_scores.get(idx, 1)
            detection_reasons = self.outlier_types.get(idx, ['unknown'])
            case_id = self.df.loc[idx,
                                  case_col] if case_col in self.df.columns else 'N/A'
            activity = self.df.loc[idx,
                                   activity_col] if activity_col in self.df.columns else 'N/A'

            # Convert detection method codes to user-friendly explanations
            reason_explanations = []
            for reason in detection_reasons:
                if reason == 'time':
                    reason_explanations.append(
                        "⏰ Unusual timing (off-hours/weekend)")
                elif reason == 'case_duration':
                    reason_explanations.append(
                        "📅 Case duration anomaly (too long/short)")
                elif reason == 'activity_frequency':
                    reason_explanations.append(
                        "📊 Rare activity (< 1% frequency)")
                elif reason == 'resource':
                    reason_explanations.append(
                        "👩‍⚕️ Resource workload anomaly")
                elif reason == 'sequence':
                    reason_explanations.append("Unusual workflow sequence")
                elif reason == 'case_complexity':
                    reason_explanations.append("Case complexity anomaly")
                else:
                    reason_explanations.append(f"{reason}")

            reasons_text = '<br>'.join(reason_explanations)
            hover_texts.append(
                f'HIGH-CONFIDENCE OUTLIER<br><br><b>Case:</b> {case_id}<br><b>Activity:</b> {activity}<br><b>Score:</b> {score}/{max_score}<br><br><b>Detection Reasons:</b><br>{reasons_text}')

        # Add maximum score outlier points as a separate trace with enhanced highlighting
        fig.add_trace(go.Scatter(
            x=outlier_data[self.view_config['x_axis']],
            y=outlier_data[self.view_config['y_axis']],
            mode='markers',
            marker=dict(
                size=10,
                # Slightly more visible for max score outliers
                color='rgba(255, 0, 0, 0.2)',
                symbol='circle',
                # Thicker, darker border for emphasis
                line=dict(width=4, color='darkred')
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name=f'Max Score Outliers ({max_score})',
            showlegend=True
        ))

        # Add outlier statistics as annotation
        methods_used = self.statistics.get('detection_methods_used', 0)
        total_outliers = self.statistics['total_outliers']
        max_score_count = len(max_score_indices)
        stats_text = f"Max Score Outliers: {max_score_count}/{total_outliers} (Score: {max_score})<br>Detection Methods: {methods_used}/6"
        fig.add_annotation(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text=stats_text,
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='darkred',
            borderwidth=1,
            font=dict(color='darkred', size=10)
        )

        return fig

    def get_outlier_summary(self) -> Dict[str, Any]:
        """Get detailed summary of detected outliers."""
        if not self.detected:
            return {
                "message": "No outliers detected",
                "available_columns": list(self.available_columns),
                "detection_methods_used": self.statistics.get('detection_methods_used', 0)
            }

        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id')
        activity_col = self._has_column(
            'activity', 'concept:name', 'event_name', 'activity_name')

        summary = {
            "statistics": self.statistics,
            "outlier_details": {},
            "top_outlier_cases": [],
            "outlier_activities": [],
            "available_columns": list(self.available_columns)
        }

        # Detailed breakdown by type
        for outlier_type, indices in self.outliers.items():
            if outlier_type != 'combined' and indices:
                sample_cases = []
                if case_col and case_col in self.df.columns:
                    sample_cases = list(
                        set(self.df.loc[indices[:10], case_col].tolist()))

                summary["outlier_details"][outlier_type] = {
                    "count": len(indices),
                    "percentage": len(indices) / len(self.df) * 100,
                    "sample_cases": sample_cases
                }

        # Top outlier cases (cases with most outlier events)
        if self.outliers.get('combined') and case_col and case_col in self.df.columns:
            outlier_cases = self.df.loc[self.outliers['combined']][case_col].value_counts(
            )
            summary["top_outlier_cases"] = [
                {"case_id": case_id, "outlier_events": count}
                for case_id, count in outlier_cases.head(10).items()
            ]

        # Most common outlier activities
        if self.outliers.get('combined') and activity_col and activity_col in self.df.columns:
            outlier_activities = self.df.loc[self.outliers['combined']][activity_col].value_counts(
            )
            summary["outlier_activities"] = [
                {"activity": activity, "outlier_count": count}
                for activity, count in outlier_activities.head(10).items()
            ]

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """
        Get standardized pattern summary.
        
        Returns
        -------
        Dict[str, Any]
            Standardized summary with pattern_type, detected, count, and details
        """
        outlier_summary = self.get_outlier_summary()
        
        return {
            'pattern_type': 'outlier',
            'detected': self.detected,
            'count': outlier_summary.get('statistics', {}).get('total_outliers', 0),
            'details': outlier_summary
        }