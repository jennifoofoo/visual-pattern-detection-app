"""
Simplified outlier detection using only Isolation Forest.
Single robust ML-based method for detecting various outlier types.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .pattern_base import Pattern
from config.extended_pattern_matrix import is_pattern_meaningful


class OutlierDetectionPattern(Pattern):
    """
    Outlier detector using Isolation Forest algorithm.
    Automatically builds features from available columns and detects anomalies.
    Provides human-readable explanations for each outlier.
    """

    def __init__(self, df: pd.DataFrame, view_config: Dict[str, str]):
        super().__init__("Outlier Detection", view_config)
        self.df = df
        self.outlier_indices: List[int] = []
        self.outlier_scores: Dict[int, float] = {}
        # Why each event is an outlier
        self.outlier_explanations: Dict[int, str] = {}
        self.statistics = {}
        self.outliers = {}  # For backward compatibility
        # Store feature names for explanations
        self.feature_names: List[str] = []
        # Store scaled features
        self.feature_matrix: Optional[np.ndarray] = None
        # Store unscaled features
        self.raw_feature_matrix: Optional[np.ndarray] = None

    def _find_column(self, *names: str) -> Optional[str]:
        """Find first matching column (case-insensitive)."""
        df_cols_lower = {col.lower(): col for col in self.df.columns}
        for name in names:
            if name.lower() in df_cols_lower:
                return df_cols_lower[name.lower()]
        return None

    def detect(self) -> bool:
        """Detect outliers using Isolation Forest."""
        if self.df is None or len(self.df) == 0:
            return False

        # Check if meaningful for this view
        x_col = self.view_config.get('x', '')
        y_col = self.view_config.get('y', '')
        color_col = self.view_config.get('color', '')

        if not is_pattern_meaningful(x_col, y_col, color_col, 'outlier'):
            print("Outlier detection skipped - not meaningful for this view")
            return False

        try:
            features, feature_names = self._build_features()

            if len(features) < 2:  # Need at least 2 features
                print(f"Not enough features: {len(features)}")
                return False

            # Stack and normalize features
            X = np.column_stack(features)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Store for explanations
            self.feature_names = feature_names
            self.feature_matrix = X_scaled
            self.raw_feature_matrix = X  # Keep unscaled for readable explanations

            # Run Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.05,  # Expect 5% outliers
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )

            predictions = iso_forest.fit_predict(X_scaled)
            anomaly_scores = iso_forest.score_samples(X_scaled)

            # Get outlier indices
            outlier_mask = predictions == -1
            self.outlier_indices = self.df.index[outlier_mask].tolist()

            # Store anomaly scores and generate explanations
            for idx, score in zip(self.df.index[outlier_mask], anomaly_scores[outlier_mask]):
                self.outlier_scores[idx] = abs(score)
                row_position = self.df.index.get_loc(idx)
                self.outlier_explanations[idx] = self._explain_outlier(
                    row_position, X_scaled[row_position], X[row_position])

            # For backward compatibility
            self.outliers['combined'] = self.outlier_indices

            # Add outlier_reason column to dataframe for hover display
            self.df['outlier_reason'] = ''
            for idx, explanation in self.outlier_explanations.items():
                self.df.at[idx, 'outlier_reason'] = explanation

            # Calculate statistics
            self._calculate_statistics(feature_names)

            self.detected = len(self.outlier_indices) > 0
            print(
                f"Isolation Forest detected {len(self.outlier_indices)} outliers using {len(feature_names)} features")
            return self.detected

        except Exception as e:
            print(f"Outlier detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _build_features(self) -> tuple:
        """Build feature matrix from available columns."""
        features = []
        feature_names = []

        case_col = self._find_column('case_id', 'case:concept:name', 'caseid')
        time_col = self._find_column(
            'actual_time', 'timestamp', 'time', 'start_time')
        activity_col = self._find_column(
            'activity', 'concept:name', 'event_name')
        resource_col = self._find_column('resource', 'org:resource')

        # Feature 1: Case-level metrics
        if case_col:
            case_counts = self.df.groupby(case_col).size()
            features.append(self.df[case_col].map(case_counts).values)
            feature_names.append('events_per_case')

            case_position = self.df.groupby(case_col).cumcount()
            features.append(case_position.values)
            feature_names.append('position_in_case')

        # Feature 2: Time-based features
        if time_col:
            df_time = self.df.copy()
            df_time[time_col] = pd.to_datetime(
                df_time[time_col], errors='coerce')

            valid_time_mask = df_time[time_col].notna()
            if valid_time_mask.sum() > 0:
                # Extract temporal features
                hours = df_time[time_col].dt.hour.fillna(12).values
                day_of_week = df_time[time_col].dt.dayofweek.fillna(2).values

                min_time = df_time.loc[valid_time_mask, time_col].min()
                hours_from_start = (
                    df_time[time_col] - min_time).dt.total_seconds() / 3600
                hours_from_start = hours_from_start.fillna(0).values

                features.append(hours)
                features.append(day_of_week)
                features.append(hours_from_start)
                feature_names.extend(['hour', 'day_of_week', 'time_offset'])

        # Feature 3: Activity frequency
        if activity_col:
            activity_freq = self.df[activity_col].value_counts()
            features.append(self.df[activity_col].map(activity_freq).values)
            feature_names.append('activity_frequency')

        # Feature 4: Resource workload
        if resource_col:
            resource_freq = self.df[resource_col].value_counts()
            features.append(self.df[resource_col].map(resource_freq).values)
            feature_names.append('resource_workload')

        return features, feature_names

    def _explain_outlier(self, row_idx: int, scaled_features: np.ndarray, raw_features: np.ndarray) -> str:
        """Generate human-readable explanation for why an event is an outlier."""
        reasons = []

        # Check each feature - if it's >2 std devs from mean, it's extreme
        for i, (feat_name, scaled_val, raw_val) in enumerate(zip(self.feature_names, scaled_features, raw_features)):
            if abs(scaled_val) > 2.0:  # More than 2 standard deviations
                if feat_name == 'events_per_case':
                    if scaled_val > 2.0:
                        reasons.append(
                            f"Case has unusually many events ({int(raw_val)})")
                    else:
                        reasons.append(
                            f"Case has very few events ({int(raw_val)})")

                elif feat_name == 'position_in_case':
                    if scaled_val > 2.0:
                        reasons.append(
                            f"Very late position in case (#{int(raw_val)+1})")
                    else:
                        reasons.append(
                            f"Very early in case (#{int(raw_val)+1})")

                elif feat_name == 'hour':
                    hour = int(raw_val)
                    if hour < 7 or hour > 18:
                        reasons.append(f"Off-hours timing ({hour}:00)")
                    else:
                        reasons.append(f"Unusual hour ({hour}:00)")

                elif feat_name == 'day_of_week':
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    try:
                        day_index = int(raw_val)
                    except (TypeError, ValueError):
                        # Skip invalid day values gracefully
                        continue
                    if 0 <= day_index < len(days):
                        day = days[day_index]
                        if day_index >= 5:
                            reasons.append(f"Weekend activity ({day})")
                        else:
                            reasons.append(f"Unusual weekday ({day})")
                    else:
                        # Skip out-of-range day values gracefully
                        continue

                elif feat_name == 'time_offset':
                    hours = raw_val
                    if scaled_val > 2.0:
                        reasons.append(
                            f"Very late in timeline ({hours:.0f}h from start)")
                    else:
                        reasons.append(f"Very early in timeline")

                elif feat_name == 'activity_frequency':
                    if scaled_val < -2.0:
                        reasons.append(
                            f"Rare activity (only {int(raw_val)} occurrences)")
                    else:
                        reasons.append(
                            f"Very common activity ({int(raw_val)} times)")

                elif feat_name == 'resource_workload':
                    if scaled_val > 2.0:
                        reasons.append(
                            f"Resource handles many events ({int(raw_val)})")
                    else:
                        reasons.append(
                            f"Resource handles few events ({int(raw_val)})")

        # If we found extreme features, return them
        if reasons:
            return " • ".join(reasons)

        # If no extreme features found, show top contributors
        if not reasons:
            # Get top 3 features by absolute scaled value
            feature_contributions = sorted(
                zip(self.feature_names, scaled_features, raw_features),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]

            for feat_name, scaled_val, raw_val in feature_contributions:
                if feat_name == 'events_per_case':
                    reasons.append(f"{int(raw_val)} events in case")
                elif feat_name == 'position_in_case':
                    reasons.append(f"Position #{int(raw_val)+1} in case")
                elif feat_name == 'hour':
                    reasons.append(f"At {int(raw_val)}:00")
                elif feat_name == 'day_of_week':
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    if raw_val >= 0 and raw_val < len(days):
                        reasons.append(f"On {days[int(raw_val)]}")
                elif feat_name == 'time_offset':
                    reasons.append(f"{raw_val:.0f}h from start")
                elif feat_name == 'activity_frequency':
                    reasons.append(f"Activity occurs {int(raw_val)} times")
                elif feat_name == 'resource_workload':
                    reasons.append(f"Resource has {int(raw_val)} events")

            return "Combined pattern: " + " • ".join(reasons)

    def _calculate_statistics(self, feature_names: List[str]):
        """Calculate detection statistics."""
        case_col = self._find_column('case_id', 'case:concept:name', 'caseid')

        total_events = len(self.df)
        total_outliers = len(self.outlier_indices)

        self.statistics = {
            'total_events': total_events,
            'total_outliers': total_outliers,
            'outlier_percentage': (total_outliers / total_events * 100) if total_events > 0 else 0,
            'features_used': feature_names,
            'detection_method': 'Isolation Forest',
            'detection_methods_used': 1
        }

        if case_col:
            self.statistics['total_cases'] = self.df[case_col].nunique()
            if self.outlier_indices:
                self.statistics['cases_with_outliers'] = len(
                    set(self.df.loc[self.outlier_indices, case_col].tolist())
                )

    def get_summary(self) -> Dict[str, Any]:
        """Get standardized pattern summary."""
        if not self.detected:
            return {
                'pattern_type': 'outlier',
                'detected': False,
                'count': 0,
                'details': {'message': 'No outliers detected'}
            }

        stats = self.statistics
        details = {
            'statistics': stats,
            'message': f"Detected {stats['total_outliers']} outlier events ({stats['outlier_percentage']:.1f}%) using {stats['detection_method']}"
        }

        if 'cases_with_outliers' in stats:
            details['cases_affected'] = f"{stats['cases_with_outliers']}/{stats['total_cases']} cases"

        return {
            'pattern_type': 'outlier',
            'detected': True,
            'count': stats['total_outliers'],
            'details': details
        }

    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """Add outlier markers to the figure."""
        if not self.detected or not self.outlier_indices:
            return fig

        # Get view configuration
        x_col = self.view_config.get('x')
        y_col = self.view_config.get('y')

        if not x_col or not y_col:
            return fig

        # Filter outliers based on session state selection if available
        import streamlit as st
        selected_outliers = st.session_state.get(
            'selected_outlier_indices', [])

        if selected_outliers:
            display_indices = [
                idx for idx in self.outlier_indices if idx in selected_outliers]
        else:
            display_indices = self.outlier_indices

        if not display_indices:
            return fig

        # Limit for performance (max 500 outliers shown)
        if len(display_indices) > 500:
            # Show top 500 by anomaly score
            display_indices = sorted(display_indices,
                                     key=lambda idx: self.outlier_scores.get(
                                         idx, 0),
                                     reverse=True)[:500]

        # Get outlier data
        outlier_df = self.df.loc[display_indices]

        # Create hover text with anomaly scores and explanations
        hover_texts = []
        case_col = self._find_column(
            'case_id', 'case:concept:name', 'caseid') or 'case_id'
        activity_col = self._find_column(
            'activity', 'concept:name', 'event_name') or 'activity'

        for idx in display_indices:
            score = self.outlier_scores.get(idx, 0)
            case_id = outlier_df.loc[idx,
                                     case_col] if case_col in outlier_df.columns else 'N/A'
            activity = outlier_df.loc[idx,
                                      activity_col] if activity_col in outlier_df.columns else 'N/A'
            explanation = self.outlier_explanations.get(
                idx, "Anomalous pattern")

            hover_texts.append(
                f"<b>⚠️ OUTLIER</b><br>"
                f"Case: {case_id}<br>"
                f"Activity: {activity}<br>"
                f"Anomaly Score: {score:.3f}<br>"
                f"<br><b>Why:</b> {explanation}"
            )

        # Add outlier scatter trace
        fig.add_trace(go.Scatter(
            x=outlier_df[x_col],
            y=outlier_df[y_col],
            mode='markers',
            marker=dict(
                symbol='x',
                size=12,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            name=f'Outliers ({len(display_indices)})',
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))

        return fig

    def get_outlier_data(self) -> pd.DataFrame:
        """Get DataFrame of outlier events with scores and explanations."""
        if not self.detected:
            return pd.DataFrame()

        outlier_df = self.df.loc[self.outlier_indices].copy()
        outlier_df['anomaly_score'] = outlier_df.index.map(self.outlier_scores)
        outlier_df['reason'] = outlier_df.index.map(self.outlier_explanations)
        outlier_df = outlier_df.sort_values('anomaly_score', ascending=False)

        return outlier_df

    def get_outlier_summary(self) -> Dict[str, Any]:
        """Get detailed summary for backward compatibility and overview display."""
        if not self.detected:
            return {'message': 'No outliers detected'}

        case_col = self._find_column('case_id', 'case:concept:name', 'caseid')
        activity_col = self._find_column(
            'activity', 'concept:name', 'event_name')

        summary = {'statistics': self.statistics}

        # Analyze explanation patterns - what are the most common reasons?
        reason_categories = {
            'Timing issues': 0,
            'Case complexity': 0,
            'Rare activities': 0,
            'Resource workload': 0,
            'Weekend/unusual days': 0,
            'Position anomalies': 0,
            'Timeline anomalies': 0,
            'Combined patterns': 0
        }

        for idx, explanation in self.outlier_explanations.items():
            if not explanation:  # Skip None or empty explanations
                continue
            if 'Off-hours' in explanation or 'Unusual hour' in explanation:
                reason_categories['Timing issues'] += 1
            if 'Case has' in explanation:
                reason_categories['Case complexity'] += 1
            if 'Rare activity' in explanation:
                reason_categories['Rare activities'] += 1
            if 'Resource' in explanation:
                reason_categories['Resource workload'] += 1
            if 'Weekend' in explanation or 'Unusual weekday' in explanation:
                reason_categories['Weekend/unusual days'] += 1
            if 'position in case' in explanation:
                reason_categories['Position anomalies'] += 1
            if 'timeline' in explanation:
                reason_categories['Timeline anomalies'] += 1
            if 'Combined pattern' in explanation:
                reason_categories['Combined patterns'] += 1

        # Filter out zero counts and sort by frequency
        top_reasons = sorted(
            [(reason, count)
             for reason, count in reason_categories.items() if count > 0],
            key=lambda x: x[1],
            reverse=True
        )
        outlier_count = len(self.outlier_indices)
        if outlier_count == 0:
            summary['top_reasons'] = []
        else:
            summary['top_reasons'] = [
                {
                    'reason': reason,
                    'count': count,
                    'percentage': (count / outlier_count * 100),
                }
                for reason, count in top_reasons[:5]
            ]
        # Most common specific explanations
        from collections import Counter
        # Filter out None values before counting
        valid_explanations = [
            exp for exp in self.outlier_explanations.values() if exp]
        explanation_counts = Counter(valid_explanations)
        if outlier_count == 0:
            summary['most_common_explanations'] = []
        else:
            summary['most_common_explanations'] = [
                {
                    'explanation': explanation,
                    'count': count,
                    'percentage': (count / outlier_count * 100),
                }
                for explanation, count in explanation_counts.most_common(5)
                if explanation  # Additional safety check
            ]
        # Top outlier cases
        if case_col and case_col in self.df.columns:
            outlier_cases = self.df.loc[self.outlier_indices,
                                        case_col].value_counts()
            summary['top_outlier_cases'] = [
                {'case_id': case_id, 'outlier_events': count}
                for case_id, count in outlier_cases.head(10).items()
            ]

        # Most common outlier activities
        if activity_col and activity_col in self.df.columns:
            outlier_activities = self.df.loc[self.outlier_indices, activity_col].value_counts(
            )
            summary['outlier_activities'] = [
                {'activity': activity, 'outlier_count': count}
                for activity, count in outlier_activities.head(10).items()
            ]

        return summary
