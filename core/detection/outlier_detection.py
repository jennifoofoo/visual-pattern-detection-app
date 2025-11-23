"""
Outlier detection for event logs in dotted charts.
Detects various types of outliers in process mining data.
Handles missing columns gracefully.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from .pattern_base import Pattern


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
        try:
            detection_count = 0

            # 1. Time-based outliers (VIEW DEPENDENT - adapts to X-axis)
            if self._detect_time_outliers():
                detection_count += 1

            # 2. Case duration outliers (if time data available)
            # VIEW INDEPENDENT
            if self._detect_case_duration_outliers():
                detection_count += 1

            # 3. Activity frequency outliers (always possible)
            # VIEW INDEPENDENT
            if self._detect_activity_frequency_outliers():
                detection_count += 1

            # 4. Resource behavior outliers (if resource column exists)
            # VIEW INDEPENDENT
            if self._detect_resource_outliers():
                detection_count += 1

            # 5. Event sequence outliers (VIEW DEPENDENT - only runs for Y=activity)
            if self._detect_sequence_outliers():
                detection_count += 1

            # 6. Case complexity outliers (always possible with case_id)
            # VIEW INDEPENDENT
            if self._detect_case_complexity_outliers():
                detection_count += 1
            # 7. Position/order outliers (VIEW DEPENDENT - adapts to Y-axis)
            if self._detect_position_outliers():
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
        """Detect events that occur at unusual times (view-dependent)."""
        x_axis = self.view_config.get('x_axis', '')

        # Different detection logic based on X-axis type
        if x_axis == 'actual_time':
            return self._detect_calendar_time_outliers()
        elif x_axis in ['relative_time', 'relative_ratio']:
            return self._detect_relative_time_outliers()
        elif x_axis in ['logical_time', 'logical_relative']:
            return self._detect_logical_time_outliers()
        else:
            return False

    def _detect_calendar_time_outliers(self) -> bool:
        """Detect events at unusual calendar times (weekends, off-hours)."""
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

            # Calendar-based anomalies: off-hours and weekends
            if df_time['hour'].nunique() > 10:  # Need hour variety
                hour_counts = df_time['hour'].value_counts()
                if len(hour_counts) > 10:
                    # Detect extremely rare hours (bottom 5%)
                    rare_threshold = max(1, hour_counts.quantile(0.05))
                    rare_hours = hour_counts[hour_counts <=
                                             rare_threshold].index
                    time_outliers.extend(
                        df_time[df_time['hour'].isin(rare_hours)].index.tolist())

            # Weekend work detection
            if df_time['day_of_week'].nunique() > 3:
                weekend_mask = df_time['day_of_week'].isin(
                    [5, 6])  # Saturday, Sunday
                weekend_events = df_time[weekend_mask]
                if len(weekend_events) < len(df_time) * 0.1:  # Less than 10% weekend work
                    time_outliers.extend(weekend_events.index.tolist())

            # Weekend work detection
            if df_time['day_of_week'].nunique() > 3:
                weekend_mask = df_time['day_of_week'].isin(
                    [5, 6])  # Saturday, Sunday
                weekend_events = df_time[weekend_mask]
                if len(weekend_events) < len(df_time) * 0.1:  # Less than 10% weekend work
                    time_outliers.extend(weekend_events.index.tolist())

            self.outliers['time'] = list(set(time_outliers))
            return len(time_outliers) > 0

        except Exception as e:
            print(f"Calendar time outlier detection failed: {e}")
            return False

    def _detect_relative_time_outliers(self) -> bool:
        """Detect events at unusual positions within cases."""
        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id')
        x_axis = self.view_config.get('x_axis', '')

        if not case_col or x_axis not in self.df.columns:
            return False

        try:
            df_rel = self.df.copy()
            time_outliers = []

            # For each case, find events at unusual relative positions
            for case_id in df_rel[case_col].unique():
                case_events = df_rel[df_rel[case_col] == case_id]
                if len(case_events) < 3:  # Need enough events for meaningful analysis
                    continue

                case_times = case_events[x_axis].values

                if x_axis == 'relative_ratio':
                    # Find events at very beginning (< 5%) or very end (> 95%)
                    extreme_early = case_events[case_times < 0.05]
                    extreme_late = case_events[case_times > 0.95]
                    time_outliers.extend(extreme_early.index.tolist())
                    time_outliers.extend(extreme_late.index.tolist())

                elif x_axis == 'relative_time':
                    # Find events with unusual relative timing using z-score
                    if len(case_times) > 2:
                        mean_time = case_times.mean()
                        std_time = case_times.std()
                        if std_time > 0:
                            z_scores = abs((case_times - mean_time) / std_time)
                            outlier_mask = z_scores > 2.5
                            time_outliers.extend(
                                case_events[outlier_mask].index.tolist())

            self.outliers['time'] = list(set(time_outliers))
            return len(time_outliers) > 0

        except Exception as e:
            print(f"Relative time outlier detection failed: {e}")
            return False

    def _detect_logical_time_outliers(self) -> bool:
        """Detect events at unusual logical positions in the global sequence."""
        x_axis = self.view_config.get('x_axis', '')

        if x_axis not in self.df.columns:
            return False

        try:
            df_logical = self.df.copy()
            logical_times = df_logical[x_axis].values

            # Find gaps or jumps in logical sequence
            if x_axis == 'logical_time':
                # Find large gaps in global event sequence
                sorted_times = sorted(logical_times)
                gaps = [sorted_times[i+1] - sorted_times[i]
                        for i in range(len(sorted_times)-1)]

                if gaps:
                    gap_threshold = np.percentile(gaps, 95)  # Top 5% of gaps
                    time_outliers = []

                    for i, gap in enumerate(gaps):
                        if gap > gap_threshold:
                            # Events after large gaps are outliers
                            gap_start_time = sorted_times[i+1]
                            outlier_events = df_logical[df_logical[x_axis]
                                                        == gap_start_time]
                            time_outliers.extend(outlier_events.index.tolist())

            elif x_axis == 'logical_relative':
                # Find events at unusual positions within their cases
                case_col = self._has_column(
                    'case_id', 'case:concept:name', 'caseid', 'trace_id')
                if case_col:
                    time_outliers = []
                    case_max_positions = df_logical.groupby(case_col)[
                        x_axis].max()

                    for case_id in df_logical[case_col].unique():
                        case_events = df_logical[df_logical[case_col] == case_id]
                        max_pos = case_max_positions[case_id]

                        if max_pos > 5:  # Need enough events in case
                            # Find events at very beginning (pos 0) or very end (max pos)
                            extreme_positions = case_events[
                                (case_events[x_axis] == 0) |
                                (case_events[x_axis] == max_pos)
                            ]
                            # Only flag if these positions are rare across all cases
                            if len(extreme_positions) < len(case_events) * 0.2:
                                time_outliers.extend(
                                    extreme_positions.index.tolist())

            self.outliers['time'] = list(set(time_outliers))
            return len(time_outliers) > 0

        except Exception as e:
            print(f"Logical time outlier detection failed: {e}")
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
        """Detect unusual resource behavior. 
        for example, resources with extremely high or low workload."""
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
    # TODO: Odo i need it or we can use the varians other way around?

    def _detect_sequence_outliers(self) -> bool:
        """Detect unusual activity sequences within cases (view-dependent)."""
        y_axis = self.view_config.get('y_axis', '')

        # Only meaningful when Y-axis shows activities
        if y_axis not in ['activity', 'concept:name', 'event_name', 'activity_name']:
            return False

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

    def _detect_position_outliers(self) -> bool:
        """Detect position outliers based on view configuration."""
        y_axis = self.view_config.get('y_axis', '')

        # Different position detection logic based on Y-axis
        if y_axis in ['activity', 'concept:name', 'event_name', 'activity_name']:
            return self._detect_activity_position_outliers()
        elif y_axis in ['case_id', 'case:concept:name', 'caseid', 'trace_id']:
            return self._detect_case_timing_outliers()
        elif y_axis in ['resource', 'org:resource', 'user', 'performer', 'resource_name']:
            return self._detect_resource_timing_outliers()
        else:
            return False

    def _detect_activity_position_outliers(self) -> bool:
        """Detect activities appearing at unusual positions within cases."""
        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id')
        activity_col = self._has_column(
            'activity', 'concept:name', 'event_name', 'activity_name')
        time_col = self._has_column(
            'actual_time', 'timestamp', 'time', 'start_time', 'complete_time')
        resource_col = self._has_column(
            'resource', 'org:resource', 'user', 'performer', 'resource_name')

        if not case_col or not activity_col:
            return False

        try:
            df = self.df.copy()

            # Calculate position outliers
            df['position'] = df.groupby(case_col).cumcount()
            pos_stats = df.groupby(activity_col)['position'].agg(
                ['mean', 'std']).reset_index()
            pos_stats = pos_stats.rename(
                columns={'mean': 'pos_mean', 'std': 'pos_std'})
            df = df.merge(pos_stats, on=activity_col, how='left')
            df['pos_std'] = df['pos_std'].replace(0, 1e-9)
            df['position_z'] = (
                df['position'] - df['pos_mean']) / df['pos_std']

            position_outliers = df[df['position_z'].abs() > 3].index.tolist()

            # Calculate workload metrics if we have time and resource data
            if time_col and resource_col:
                self._calculate_workload_metrics(
                    df, time_col, resource_col, activity_col)

            if position_outliers:
                self.outliers['position'] = position_outliers
                return True

            return False

        except Exception as e:
            print(f"Activity position outlier detection failed: {e}")
            return False

    def _detect_case_timing_outliers(self) -> bool:
        """Detect cases with unusual event timing patterns."""
        case_col = self._has_column(
            'case_id', 'case:concept:name', 'caseid', 'trace_id')
        x_axis = self.view_config.get('x_axis', '')

        if not case_col or x_axis not in self.df.columns:
            return False

        try:
            df_case = self.df.copy()
            position_outliers = []

            # Find cases with unusual event timing distributions
            for case_id in df_case[case_col].unique():
                case_events = df_case[df_case[case_col] == case_id]
                if len(case_events) < 3:
                    continue

                case_times = case_events[x_axis].values

                # Detect cases with irregular timing patterns
                if len(case_times) > 2:
                    time_diffs = np.diff(sorted(case_times))
                    if len(time_diffs) > 0:
                        # Find cases with extremely irregular timing
                        cv = np.std(
                            time_diffs) / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
                        if cv > 2.0:  # High coefficient of variation = irregular timing
                            position_outliers.extend(
                                case_events.index.tolist())

            if position_outliers:
                self.outliers['position'] = position_outliers
                return True

            return False

        except Exception as e:
            print(f"Case timing outlier detection failed: {e}")
            return False

    def _detect_resource_timing_outliers(self) -> bool:
        """Detect resources working at unusual times or with irregular patterns."""
        resource_col = self._has_column(
            'resource', 'org:resource', 'user', 'performer', 'resource_name')
        x_axis = self.view_config.get('x_axis', '')

        if not resource_col or x_axis not in self.df.columns:
            return False

        try:
            df_resource = self.df.copy()
            position_outliers = []

            # Calculate workload metrics for timing analysis
            time_col = self._has_column(
                'actual_time', 'timestamp', 'time', 'start_time', 'complete_time')
            activity_col = self._has_column(
                'activity', 'concept:name', 'event_name', 'activity_name')

            if time_col and activity_col:
                self._calculate_workload_metrics(
                    df_resource, time_col, resource_col, activity_col)

            # Find resources with unusual working patterns
            for resource_id in df_resource[resource_col].unique():
                resource_events = df_resource[df_resource[resource_col]
                                              == resource_id]
                if len(resource_events) < 5:
                    continue

                resource_times = resource_events[x_axis].values

                # Detect resources with very clustered or very spread work patterns
                if len(resource_times) > 2:
                    time_span = max(resource_times) - min(resource_times)
                    avg_gap = time_span / \
                        len(resource_times) if len(resource_times) > 1 else 0

                    # Find resources with either too clustered (< 10% of time span) or too spread work
                    if time_span > 0:
                        density = len(resource_times) / time_span
                        # Get global density statistics for comparison
                        all_densities = []
                        for rid in df_resource[resource_col].unique():
                            r_events = df_resource[df_resource[resource_col] == rid]
                            if len(r_events) > 2:
                                r_times = r_events[x_axis].values
                                r_span = max(r_times) - min(r_times)
                                if r_span > 0:
                                    all_densities.append(len(r_times) / r_span)

                        if all_densities:
                            density_threshold = np.percentile(
                                all_densities, 95)
                            if density > density_threshold:  # Extremely high density = clustered work
                                position_outliers.extend(
                                    resource_events.index.tolist())

            if position_outliers:
                self.outliers['position'] = position_outliers
                return True

            return False

        except Exception as e:
            print(f"Resource timing outlier detection failed: {e}")
            return False

    def _calculate_workload_metrics(self, df: pd.DataFrame, time_col: str, resource_col: str, activity_col: str):
        """Calculate workload metrics for time x resource x activity."""
        try:
            # Convert time to datetime
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col])

            if df.empty:
                return

            # Extract time components
            df['hour'] = df[time_col].dt.hour
            df['day'] = df[time_col].dt.date
            df['week'] = df[time_col].dt.isocalendar().week
            df['month'] = df[time_col].dt.month

            # Calculate workload by different time granularities
            workload_metrics = {}

            # Hourly workload per resource per activity
            hourly_workload = df.groupby(
                [resource_col, activity_col, 'hour']).size().reset_index(name='workload')
            workload_metrics['hourly'] = hourly_workload

            # Daily workload per resource per activity
            daily_workload = df.groupby(
                [resource_col, activity_col, 'day']).size().reset_index(name='workload')
            workload_metrics['daily'] = daily_workload

            # Weekly workload per resource per activity
            weekly_workload = df.groupby(
                [resource_col, activity_col, 'week']).size().reset_index(name='workload')
            workload_metrics['weekly'] = weekly_workload

            # Store workload data for visualization
            self.workload_data = workload_metrics

            # Detect workload outliers
            self._detect_workload_outliers(
                workload_metrics, df, resource_col, activity_col)

        except Exception as e:
            print(f"Workload calculation failed: {e}")

    def _detect_workload_outliers(self, workload_metrics: dict, df: pd.DataFrame, resource_col: str, activity_col: str):
        """Detect outliers in workload patterns."""
        try:
            workload_outlier_indices = []

            for granularity, workload_df in workload_metrics.items():
                if workload_df.empty:
                    continue

                # Use IQR method to find workload outliers
                Q1 = workload_df['workload'].quantile(0.25)
                Q3 = workload_df['workload'].quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:
                    # Find extreme workload cases (3 * IQR)
                    outlier_threshold_low = Q1 - 3.0 * IQR
                    outlier_threshold_high = Q3 + 3.0 * IQR

                    outlier_workloads = workload_df[
                        (workload_df['workload'] < outlier_threshold_low) |
                        (workload_df['workload'] > outlier_threshold_high)
                    ]

                    # Map back to original dataframe indices
                    for _, row in outlier_workloads.iterrows():
                        if granularity == 'hourly':
                            mask = (df[resource_col] == row[resource_col]) & \
                                   (df[activity_col] == row[activity_col]) & \
                                   (df['hour'] == row['hour'])
                        elif granularity == 'daily':
                            mask = (df[resource_col] == row[resource_col]) & \
                                   (df[activity_col] == row[activity_col]) & \
                                   (df['day'] == row['day'])
                        else:  # weekly
                            mask = (df[resource_col] == row[resource_col]) & \
                                   (df[activity_col] == row[activity_col]) & \
                                   (df['week'] == row['week'])

                        workload_outlier_indices.extend(
                            df[mask].index.tolist())

            if workload_outlier_indices:
                self.outliers['workload'] = list(set(workload_outlier_indices))

        except Exception as e:
            print(f"Workload outlier detection failed: {e}")

    def create_workload_heatmap(self) -> go.Figure:
        """Create a dedicated workload heatmap visualization."""
        if not hasattr(self, 'workload_data') or not self.workload_data:
            return go.Figure().add_annotation(
                text="No workload data available. Ensure your data has time, resource, and activity columns.",
                x=0.5, y=0.5, showarrow=False, font=dict(size=14)
            )

        try:
            resource_col = self._has_column(
                'resource', 'org:resource', 'user', 'performer', 'resource_name')
            activity_col = self._has_column(
                'activity', 'concept:name', 'event_name', 'activity_name')

            daily_workload = self.workload_data.get('daily')
            if daily_workload is None or daily_workload.empty:
                return go.Figure().add_annotation(
                    text="No daily workload data available",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=14)
                )

            # Create pivot table for heatmap
            pivot_data = daily_workload.pivot_table(
                index=resource_col,
                columns='day',
                values='workload',
                aggfunc='sum',
                fill_value=0
            )

            # Filter to top 5 most loaded resources
            resource_totals = pivot_data.sum(
                axis=1).sort_values(ascending=False)
            top_5_resources = resource_totals.head(5).index
            pivot_data_filtered = pivot_data.loc[top_5_resources]

            total_resources = len(pivot_data.index)

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data_filtered.values,
                x=[str(d) for d in pivot_data_filtered.columns],
                y=pivot_data_filtered.index,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Events per Day"),
                hovertemplate='<b>Resource:</b> %{y}<br><b>Date:</b> %{x}<br><b>Workload:</b> %{z} events<extra></extra>'
            ))

            fig.update_layout(
                title=f"Top 5 Most Loaded Resources Heatmap ({len(top_5_resources)} of {total_resources} resources)",
                xaxis_title="Date",
                yaxis_title="Resource",
                height=max(400, len(pivot_data_filtered.index) * 30),
                width=max(800, len(pivot_data_filtered.columns) * 50)
            )

            return fig

        except Exception as e:
            print(f"Workload heatmap creation failed: {e}")
            return go.Figure().add_annotation(
                text=f"Error creating workload heatmap: {e}",
                x=0.5, y=0.5, showarrow=False, font=dict(size=14)
            )

    def create_workload_summary_chart(self) -> go.Figure:
        """Create workload summary charts with multiple views."""
        if not hasattr(self, 'workload_data') or not self.workload_data:
            return go.Figure().add_annotation(
                text="No workload data available",
                x=0.5, y=0.5, showarrow=False, font=dict(size=14)
            )

        try:
            from plotly.subplots import make_subplots

            resource_col = self._has_column(
                'resource', 'org:resource', 'user', 'performer', 'resource_name')
            activity_col = self._has_column(
                'activity', 'concept:name', 'event_name', 'activity_name')

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Resource Workload Distribution', 'Activity Frequency',
                                'Hourly Activity Pattern', 'Resource vs Activity Workload'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "heatmap"}]]
            )

            daily_workload = self.workload_data.get('daily', pd.DataFrame())
            hourly_workload = self.workload_data.get('hourly', pd.DataFrame())

            if not daily_workload.empty:
                # 1. Resource workload distribution
                resource_total = daily_workload.groupby(
                    resource_col)['workload'].sum().sort_values(ascending=False)
                fig.add_trace(
                    go.Bar(x=resource_total.index,
                           y=resource_total.values, name='Resource Workload'),
                    row=1, col=1
                )

                # 2. Activity frequency
                activity_total = daily_workload.groupby(
                    activity_col)['workload'].sum().sort_values(ascending=False)
                fig.add_trace(
                    go.Bar(x=activity_total.index,
                           y=activity_total.values, name='Activity Frequency'),
                    row=1, col=2
                )

            if not hourly_workload.empty:
                # 3. Hourly pattern
                hourly_pattern = hourly_workload.groupby('hour')[
                    'workload'].sum()
                fig.add_trace(
                    go.Bar(x=hourly_pattern.index,
                           y=hourly_pattern.values, name='Hourly Pattern'),
                    row=2, col=1
                )

                # 4. Resource vs Activity heatmap
                if not daily_workload.empty:
                    resource_activity = daily_workload.groupby([resource_col, activity_col])[
                        'workload'].sum().reset_index()
                    pivot_ra = resource_activity.pivot(
                        index=resource_col, columns=activity_col, values='workload').fillna(0)

                    fig.add_trace(
                        go.Heatmap(
                            z=pivot_ra.values,
                            x=pivot_ra.columns,
                            y=pivot_ra.index,
                            colorscale='Blues',
                            showscale=False
                        ),
                        row=2, col=2
                    )

            fig.update_layout(
                height=800,
                title_text="Workload Analysis Dashboard",
                showlegend=False
            )

            return fig

        except Exception as e:
            print(f"Workload summary chart creation failed: {e}")
            return go.Figure().add_annotation(
                text=f"Error: {e}", x=0.5, y=0.5, showarrow=False, font=dict(size=14)
            )

    def visualize_workload(self, fig: go.Figure) -> go.Figure:
        """Add workload visualization to the existing figure."""
        if not hasattr(self, 'workload_data') or not self.workload_data:
            return fig

        try:
            resource_col = self._has_column(
                'resource', 'org:resource', 'user', 'performer', 'resource_name')
            activity_col = self._has_column(
                'activity', 'concept:name', 'event_name', 'activity_name')

            if not resource_col or not activity_col:
                return fig

            # Create workload heatmap data
            daily_workload = self.workload_data.get('daily')
            if daily_workload is not None and not daily_workload.empty:

                # Add workload information as annotation
                max_workload = daily_workload['workload'].max()
                min_workload = daily_workload['workload'].min()
                avg_workload = daily_workload['workload'].mean()

                workload_stats = f"Workload Stats:<br>Range: {min_workload:.0f} - {max_workload:.0f}<br>Average: {avg_workload:.1f} events/day<br>Resources: {daily_workload[resource_col].nunique()}<br>Activities: {daily_workload[activity_col].nunique()}"

                fig.add_annotation(
                    x=0.98, y=0.02,
                    xref='paper', yref='paper',
                    text=workload_stats,
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='blue',
                    borderwidth=1,
                    font=dict(color='blue', size=9),
                    align='left'
                )

                # Highlight high workload periods on the main plot
                if 'workload' in self.outliers:
                    workload_outliers = self.df.loc[self.outliers['workload']]

                    if not workload_outliers.empty:
                        hover_texts = []
                        for idx in self.outliers['workload']:
                            resource = self.df.loc[idx,
                                                   resource_col] if resource_col in self.df.columns else 'N/A'
                            activity = self.df.loc[idx,
                                                   activity_col] if activity_col in self.df.columns else 'N/A'
                            hover_texts.append(
                                f'WORKLOAD OUTLIER<br><b>Resource:</b> {resource}<br><b>Activity:</b> {activity}')

                        fig.add_trace(go.Scatter(
                            x=workload_outliers[self.view_config['x_axis']],
                            y=workload_outliers[self.view_config['y_axis']],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color='rgba(0, 0, 255, 0.3)',
                                symbol='diamond',
                                line=dict(width=2, color='blue')
                            ),
                            text=hover_texts,
                            hovertemplate='%{text}<extra></extra>',
                            name='Workload Outliers',
                            showlegend=True
                        ))

            return fig

        except Exception as e:
            print(f"Workload visualization failed: {e}")
            return fig

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

    def visualize(self, fig: go.Figure) -> go.Figure:
        """Add outlier visualization to the existing figure."""
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
                elif reason == 'workload':
                    reason_explanations.append(
                        "💼 Workload anomaly (resource overload/underload)")
                elif reason == 'position':
                    reason_explanations.append(
                        "📍 Position anomaly (unusual activity order)")
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
        stats_text = f"Max Score Outliers: {max_score_count}/{total_outliers} (Score: {max_score})<br>Detection Methods: {methods_used}/8"
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

    def _compute_workload(self) -> bool:
        """Public helper to compute workload_data from the current dataframe.
        Returns True if workload_data was created, False otherwise.
        Call this from the app before rendering heatmap/dashboard.
        """
        time_col = self._has_column(
            'actual_time', 'timestamp', 'time', 'start_time', 'complete_time')
        resource_col = self._has_column(
            'resource', 'org:resource', 'user', 'performer', 'resource_name')
        activity_col = self._has_column(
            'activity', 'concept:name', 'event_name', 'activity_name')

        if not (time_col and resource_col and activity_col):
            return False

        try:
            # pass a copy so internal df isn't mutated unexpectedly
            self._calculate_workload_metrics(
                self.df.copy(), time_col, resource_col, activity_col)
            return hasattr(self, 'workload_data') and bool(self.workload_data)
        except Exception:
            return False

    def create_activity_cluster_plot(self, x_col, y_col, color_col):
        """
        Create a dedicated plot showing only activity clusters.

        Args:
            x_col: Column for X-axis
            y_col: Column for Y-axis  
            color_col: Column for color mapping

        Returns:
            plotly.graph_objects.Figure: The activity cluster plot
        """
        import plotly.graph_objects as go
        from ..detection.temporal_cluster import TemporalClusterPattern

        # Create temporal cluster detector to get activity clusters
        cluster_detector = TemporalClusterPattern()
        cluster_detector.df = self.df.copy()

        # Detect patterns to get activity clusters
        cluster_detector.detect_patterns(x_col, y_col, color_col)

        # Create base plot
        fig = go.Figure()

        # Add original data points as background
        fig.add_trace(go.Scatter(
            x=self.df[x_col],
            y=self.df[y_col],
            mode='markers',
            marker=dict(
                color='lightgray',
                size=4,
                opacity=0.3
            ),
            name='All Events',
            hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>'
        ))

        # Add activity clusters if they exist
        if hasattr(cluster_detector, 'activity_clusters') and cluster_detector.activity_clusters:
            try:
                # Get unique activities from clusters
                clustered_activities = set()
                for cluster_info in cluster_detector.activity_clusters:
                    if isinstance(cluster_info, dict) and 'activities' in cluster_info:
                        clustered_activities.update(cluster_info['activities'])

                if clustered_activities:
                    # Filter data for clustered activities
                    cluster_mask = self.df[color_col].isin(
                        clustered_activities)
                    cluster_data = self.df[cluster_mask]

                    if not cluster_data.empty:
                        # Add clustered points with highlighting
                        fig.add_trace(go.Scatter(
                            x=cluster_data[x_col],
                            y=cluster_data[y_col],
                            mode='markers',
                            marker=dict(
                                color=cluster_data[color_col].astype(
                                    'category').cat.codes,
                                colorscale='Set3',
                                size=8,
                                opacity=0.8,
                                line=dict(width=1, color='black')
                            ),
                            name='Activity Clusters',
                            text=cluster_data[color_col],
                            hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>Activity: %{{text}}<extra></extra>'
                        ))

                        # Add cluster annotations
                        for i, cluster_info in enumerate(cluster_detector.activity_clusters):
                            if isinstance(cluster_info, dict) and 'activities' in cluster_info:
                                activities = cluster_info['activities']
                                # Show first 3 activities
                                activity_str = ', '.join(activities[:3])
                                if len(activities) > 3:
                                    activity_str += f' (+{len(activities)-3} more)'

                                # Find center of cluster for annotation
                                cluster_points = cluster_data[cluster_data[color_col].isin(
                                    activities)]
                                if not cluster_points.empty:
                                    center_x = cluster_points[x_col].mean()
                                    center_y = cluster_points[y_col].mean()

                                    fig.add_annotation(
                                        x=center_x,
                                        y=center_y,
                                        text=f"Cluster {i+1}<br>{activity_str}",
                                        showarrow=True,
                                        arrowhead=2,
                                        arrowsize=1,
                                        arrowwidth=2,
                                        arrowcolor="red",
                                        bgcolor="white",
                                        bordercolor="red",
                                        borderwidth=1,
                                        font=dict(size=10)
                                    )

            except Exception as e:
                # Add error message if cluster visualization fails
                fig.add_annotation(
                    x=0.5, y=0.95,
                    xref="paper", yref="paper",
                    text=f"Activity cluster visualization error: {str(e)}",
                    showarrow=False,
                    font=dict(color="red", size=12),
                    bgcolor="white",
                    bordercolor="red",
                    borderwidth=1
                )
        else:
            # Add message if no clusters found
            fig.add_annotation(
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                text="No activity clusters detected in current data",
                showarrow=False,
                font=dict(size=14, color="gray"),
                bgcolor="lightyellow",
                bordercolor="orange",
                borderwidth=1
            )

        # Update layout
        fig.update_layout(
            title="Activity Cluster Analysis",
            xaxis_title=x_col,
            yaxis_title=y_col,
            showlegend=True,
            hovermode='closest'
        )

        return fig
