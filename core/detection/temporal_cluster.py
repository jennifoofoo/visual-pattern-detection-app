"""
Temporal and spatial cluster detection for event log dotted charts.

This module detects meaningful patterns based on the specific X and Y axis combinations
available in the visualization system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.cluster import DBSCAN
from datetime import datetime
from .pattern_base import Pattern


class TemporalClusterPattern(Pattern):
    """
    Detect temporal and spatial clusters in dotted charts.

    Clusters are detected based on the specific X and Y axis being used,
    ensuring that only meaningful patterns are identified.
    """

    def __init__(self, df: pd.DataFrame, x_axis: str, y_axis: str,
                 min_cluster_size: int = 5,
                 temporal_eps: float = None,
                 spatial_eps: float = None):
        """
        Initialize temporal cluster detector.

        Args:
            df: Event log dataframe
            x_axis: X-axis column name ('actual_time', 'relative_time', etc.)
            y_axis: Y-axis column name ('case_id', 'activity', etc.)
            min_cluster_size: Minimum events to form a cluster
            temporal_eps: Optional custom epsilon for temporal clustering (auto-calculated if None)
            spatial_eps: Optional custom epsilon for spatial clustering (auto-calculated if None)
        """
        # Initialize parent with name and view_config
        view_config = {'x': x_axis, 'y': y_axis}
        super().__init__(name="Temporal Cluster", view_config=view_config)

        self.df = df
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.min_cluster_size = min_cluster_size
        self.temporal_eps = temporal_eps
        self.spatial_eps = spatial_eps

        self.clusters = {}
        self.cluster_metadata = {}

    def _has_column(self, *column_names: str):
        """Check if any of the column names exist (case-insensitive)."""
        for col_name in column_names:
            for actual_col in self.df.columns:
                if actual_col.lower() == col_name.lower():
                    return actual_col
        return None

    def detect(self, df: pd.DataFrame = None) -> bool:
        """
        Main detection method - routes to appropriate pattern detection based on axes.

        Args:
            df: Optional dataframe (uses self.df if not provided)

        Returns:
            True if any meaningful patterns are detected
        """
        # Use provided df or fall back to self.df
        if df is not None:
            self.df = df

        detected_patterns = []

        # Determine which patterns are meaningful for this axis combination
        if self._should_detect_temporal_bursts():
            if self._detect_temporal_bursts():
                detected_patterns.append('temporal_bursts')

        if self._should_detect_activity_clusters():
            if self._detect_activity_time_clusters():
                detected_patterns.append('activity_clusters')

        if self._should_detect_case_parallelism():
            if self._detect_case_parallelism():
                detected_patterns.append('case_parallelism')

        if self._should_detect_resource_patterns():
            if self._detect_resource_time_patterns():
                detected_patterns.append('resource_patterns')

        if self._should_detect_variant_patterns():
            if self._detect_variant_timing_patterns():
                detected_patterns.append('variant_patterns')

        return len(detected_patterns) > 0

    # ==================== Pattern Applicability Logic ====================

    def _should_detect_temporal_bursts(self) -> bool:
        """Check if temporal burst detection is meaningful for current axes."""
        # Meaningful when X-axis is actual time and Y-axis groups events
        return (self.x_axis == 'actual_time' and
                self.y_axis in ['activity', 'resource', 'case_id'])

    def _should_detect_activity_clusters(self) -> bool:
        """Check if activity-time clustering is meaningful."""
        # When Y-axis is activity and X-axis is any time representation
        return (self.y_axis == 'activity' and
                self.x_axis in ['actual_time', 'relative_time', 'relative_ratio'])

    def _should_detect_case_parallelism(self) -> bool:
        """Check if case parallelism detection is meaningful."""
        # When Y-axis is case_id and X-axis is actual or relative time
        return (self.y_axis == 'case_id' and
                self.x_axis in ['actual_time', 'relative_time'])

    def _should_detect_resource_patterns(self) -> bool:
        """Check if resource pattern detection is meaningful."""
        # When Y-axis is resource and X-axis is time-based
        return (self.y_axis == 'resource' and
                self.x_axis in ['actual_time', 'relative_time'])

    def _should_detect_variant_patterns(self) -> bool:
        """Check if variant pattern detection is meaningful."""
        # When Y-axis is variant and X-axis is relative time/ratio
        return (self.y_axis == 'variant' and
                self.x_axis in ['relative_time', 'relative_ratio'])

    # ==================== Temporal Burst Detection ====================

    def _detect_temporal_bursts(self) -> bool:
        """
        Detect temporal bursts - periods of high event concentration.

        Meaningful for: actual_time × {activity, resource, case_id}

        Example: Many events happening in a short time period (e.g., batch processing,
        shift changes, system issues)
        """
        if self.x_axis != 'actual_time':
            return False

        # Convert timestamps to numeric values (seconds since epoch)
        df_work = self.df.copy()
        df_work['time_numeric'] = pd.to_datetime(
            df_work[self.x_axis]).astype(np.int64) / 1e9

        # Auto-calculate epsilon if not provided (5% of time range or max 1 hour)
        if self.temporal_eps is None:
            time_range = df_work['time_numeric'].max() - \
                df_work['time_numeric'].min()
            self.temporal_eps = min(time_range * 0.05, 3600)  # Max 1 hour

        # Perform DBSCAN clustering on time dimension
        X = df_work[['time_numeric']].values
        clustering = DBSCAN(eps=self.temporal_eps,
                            min_samples=self.min_cluster_size)
        df_work['time_cluster'] = clustering.fit_predict(X)

        # Filter out noise (-1 label)
        bursts = df_work[df_work['time_cluster'] >= 0].groupby('time_cluster').agg({
            'time_numeric': ['min', 'max', 'count'],
            self.y_axis: 'nunique'
        })

        if len(bursts) == 0:
            return False

        # Store burst information
        self.clusters['temporal_bursts'] = []
        for cluster_id, row in bursts.iterrows():
            start_time = datetime.fromtimestamp(row[('time_numeric', 'min')])
            end_time = datetime.fromtimestamp(row[('time_numeric', 'max')])
            duration = (end_time - start_time).total_seconds()

            self.clusters['temporal_bursts'].append({
                'cluster_id': int(cluster_id),
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration,
                'event_count': int(row[('time_numeric', 'count')]),
                f'unique_{self.y_axis}': int(row[(self.y_axis, 'nunique')])
            })

        return True

    # ==================== Activity-Time Clustering ====================

    def _detect_activity_time_clusters(self) -> bool:
        """
        Detect when specific activities cluster at certain times.

        Meaningful for: {actual_time, relative_time, relative_ratio} × activity

        Example: 'Lab Test' activities clustered at morning hours,
        or 'Approval' activities clustered near case end
        """
        if self.y_axis != 'activity':
            return False

        df_work = self.df.copy()

        # Convert time axis to numeric
        if self.x_axis == 'actual_time':
            df_work['time_numeric'] = pd.to_datetime(
                df_work[self.x_axis]).astype(np.int64) / 1e9
            if self.temporal_eps is None:
                time_range = df_work['time_numeric'].max(
                ) - df_work['time_numeric'].min()
                self.temporal_eps = min(time_range * 0.05, 3600)
        else:
            # relative_time or relative_ratio are already numeric
            df_work['time_numeric'] = df_work[self.x_axis]
            if self.temporal_eps is None:
                self.temporal_eps = df_work['time_numeric'].std() * 0.5

        # For each activity, find time clusters
        self.clusters['activity_time'] = {}

        for activity, activity_df in df_work.groupby('activity'):
            if len(activity_df) < self.min_cluster_size:
                continue

            X = activity_df[['time_numeric']].values
            clustering = DBSCAN(eps=self.temporal_eps, min_samples=max(
                3, self.min_cluster_size // 2))
            labels = clustering.fit_predict(X)

            # Count clusters (excluding noise)
            clusters_found = len(set(labels) - {-1})

            if clusters_found > 0:
                cluster_info = []
                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue

                    cluster_events = activity_df[labels == cluster_id]
                    cluster_info.append({
                        'cluster_id': int(cluster_id),
                        'event_count': len(cluster_events),
                        'time_mean': float(cluster_events['time_numeric'].mean()),
                        'time_std': float(cluster_events['time_numeric'].std())
                    })

                self.clusters['activity_time'][activity] = cluster_info

        return len(self.clusters['activity_time']) > 0

    # ==================== Case Parallelism Detection ====================

    def _detect_case_parallelism(self) -> bool:
        """
        Detect parallel case execution - multiple cases running simultaneously.

        Meaningful for: {actual_time, relative_time} × case_id

        Example: High degree of case overlap indicating concurrent processing
        """
        if self.y_axis != 'case_id' or self.x_axis not in ['actual_time', 'relative_time']:
            return False

        # Calculate case start and end times
        df_work = self.df.copy()

        if self.x_axis == 'actual_time':
            df_work['time_numeric'] = pd.to_datetime(
                df_work[self.x_axis]).astype(np.int64) / 1e9
        else:
            df_work['time_numeric'] = df_work[self.x_axis]

        case_ranges = df_work.groupby(
            'case_id')['time_numeric'].agg(['min', 'max'])

        # Find maximum overlapping cases at any point in time
        # Create events for case start (+1) and end (-1)
        events = []
        for case_id, row in case_ranges.iterrows():
            events.append((row['min'], 1))  # Case start
            events.append((row['max'], -1))  # Case end

        events.sort()

        max_parallel = 0
        current_parallel = 0
        parallel_over_time = []

        for time, delta in events:
            current_parallel += delta
            max_parallel = max(max_parallel, current_parallel)
            parallel_over_time.append((time, current_parallel))

        # Store parallelism statistics
        self.clusters['case_parallelism'] = {
            'max_parallel_cases': max_parallel,
            'avg_parallel_cases': np.mean([count for _, count in parallel_over_time]),
            'timeline': parallel_over_time[:100]  # Limit to avoid huge data
        }

        # Return True if significant parallelism detected (more than 3 cases)
        return max_parallel > 3

    # ==================== Resource Pattern Detection ====================

    def _detect_resource_time_patterns(self) -> bool:
        """
        Detect resource utilization patterns over time.

        Meaningful for: {actual_time, relative_time} × resource

        Example: Certain resources only working at specific times,
        or resource shift patterns
        """
        if self.y_axis != 'resource':
            return False

        resource_col = self._has_column(
            'resource', 'org:resource', 'org:group')
        if not resource_col:
            return False

        df_work = self.df.copy()

        if self.x_axis == 'actual_time':
            df_work['time_numeric'] = pd.to_datetime(
                df_work[self.x_axis]).astype(np.int64) / 1e9
            if self.temporal_eps is None:
                time_range = df_work['time_numeric'].max(
                ) - df_work['time_numeric'].min()
                self.temporal_eps = min(time_range * 0.05, 3600)
        else:
            df_work['time_numeric'] = df_work[self.x_axis]
            if self.temporal_eps is None:
                self.temporal_eps = df_work['time_numeric'].std() * 0.5

        # For each resource, detect time-based work patterns
        self.clusters['resource_time'] = {}

        for resource, resource_df in df_work.groupby(resource_col):
            if len(resource_df) < self.min_cluster_size:
                continue

            X = resource_df[['time_numeric']].values
            clustering = DBSCAN(eps=self.temporal_eps, min_samples=max(
                3, self.min_cluster_size // 2))
            labels = clustering.fit_predict(X)

            clusters_found = len(set(labels) - {-1})

            if clusters_found > 1:  # Resource works in distinct time periods
                cluster_info = []
                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue

                    cluster_events = resource_df[labels == cluster_id]
                    cluster_info.append({
                        'cluster_id': int(cluster_id),
                        'event_count': len(cluster_events),
                        'time_min': float(cluster_events['time_numeric'].min()),
                        'time_max': float(cluster_events['time_numeric'].max())
                    })

                self.clusters['resource_time'][resource] = cluster_info

        return len(self.clusters['resource_time']) > 0

    # ==================== Variant Timing Pattern Detection ====================

    def _detect_variant_timing_patterns(self) -> bool:
        """
        Detect if different process variants have different timing patterns.

        Meaningful for: {relative_time, relative_ratio} × variant

        Example: Fast-track variant completes in 0.2 normalized time,
        while complex variant takes full duration
        """
        if self.y_axis != 'variant':
            return False

        if 'variant' not in self.df.columns:
            return False

        df_work = self.df.copy()
        df_work['time_numeric'] = df_work[self.x_axis]

        # For each variant, calculate timing statistics
        variant_stats = df_work.groupby('variant')['time_numeric'].agg([
            'min', 'max', 'mean', 'std', 'count'
        ])

        # Only consider variants with enough events
        variant_stats = variant_stats[variant_stats['count']
                                      >= self.min_cluster_size]

        if len(variant_stats) < 2:
            return False

        # Detect if variants have statistically different timing patterns
        # Use coefficient of variation to identify distinct patterns
        variant_stats['cv'] = variant_stats['std'] / variant_stats['mean']

        self.clusters['variant_timing'] = variant_stats.to_dict('index')

        return True

    # ==================== Visualization Support ====================

    def visualize(self, df: pd.DataFrame = None, fig=None):
        #### FOR NOW ONLY ACTIVITY BURSTS IS VISUALISED
        """
        Add cluster visualizations to the figure.

        Args:
            df: DataFrame (uses self.df if not provided)
            fig: Plotly figure to annotate (creates metadata dict if None)

        Returns:
            Plotly Figure with cluster overlays, or Dict with cluster metadata
        """
        if df is None:
            df = self.df

        # If no figure provided, return metadata for visualization
        if fig is None:
            visualization_data = {
                'clusters': self.clusters,
                'x_axis': self.x_axis,
                'y_axis': self.y_axis
            }
            return visualization_data

        # Add visual overlays to the figure
        import plotly.graph_objects as go

        # Visualize temporal bursts
        if 'temporal_bursts' in self.clusters:
            self._add_burst_visualization(fig)

        # Visualize case parallelism
        if 'case_parallelism' in self.clusters:
            self._add_parallelism_visualization(fig)

        # Visualize activity-time clusters
        if 'activity_time' in self.clusters:
            self._add_activity_cluster_visualization(fig)

        # Visualize resource patterns
        if 'resource_time' in self.clusters:
            self._add_resource_pattern_visualization(fig)

        return fig

    def _add_burst_visualization(self, fig):
        """Add temporal burst overlays to figure."""
        import plotly.graph_objects as go

        bursts = self.clusters['temporal_bursts']

        # Show only significant bursts (top 20 by event count)
        sorted_bursts = sorted(
            bursts, key=lambda x: x['event_count'], reverse=True)[:20]

        # Color palette for different clusters
        colors = ['red', 'orange', 'purple', 'magenta', 'brown',
                  'pink', 'crimson', 'darkred', 'coral', 'salmon',
                  'tomato', 'indianred', 'firebrick', 'maroon', 'darkmagenta',
                  'orchid', 'mediumvioletred', 'deeppink', 'hotpink', 'palevioletred']

        # Get cluster assignments for all events
        df_work = self.df.copy()
        df_work['time_numeric'] = pd.to_datetime(
            df_work[self.x_axis]).astype(np.int64) / 1e9

        X = df_work[['time_numeric']].values
        clustering = DBSCAN(eps=self.temporal_eps,
                            min_samples=self.min_cluster_size)
        cluster_labels = clustering.fit_predict(X)
        df_work['cluster_label'] = cluster_labels

        # For each significant burst, add highlighted points
        for i, burst in enumerate(sorted_bursts):
            cluster_id = burst['cluster_id']
            color = colors[i % len(colors)]

            # Get events in this cluster
            cluster_events = df_work[df_work['cluster_label'] == cluster_id]

            if len(cluster_events) > 0:
                # Add scatter trace for this cluster
                fig.add_trace(go.Scatter(
                    x=cluster_events[self.x_axis],
                    y=cluster_events[self.y_axis],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='circle-open',
                        line=dict(width=2, color=color),
                        opacity=0.7
                    ),
                    name=f'Burst {i+1} ({burst["event_count"]} events)',
                    showlegend=True,
                    hovertemplate=f"<b>Burst Cluster {i+1}</b><br>" +
                                  f"Events: {burst['event_count']}<br>" +
                                  f"{self.x_axis}: %{{x}}<br>" +
                                  f"{self.y_axis}: %{{y}}<extra></extra>"
                ))

        # Add summary annotation
        fig.add_annotation(
            text=f"🔴 {len(bursts)} Temporal Bursts<br>Showing top {len(sorted_bursts)}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 200, 200, 0.8)",
            bordercolor="red",
            borderwidth=1,
            font=dict(size=10)
        )

    def _add_parallelism_visualization(self, fig):
        """Add case parallelism visualization to figure."""
        para = self.clusters['case_parallelism']

        # Add annotation with parallelism statistics
        fig.add_annotation(
            text=f"⏱️ Case Parallelism<br>Max: {para['max_parallel_cases']} cases<br>Avg: {para['avg_parallel_cases']:.1f} cases",
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            showarrow=False,
            bgcolor="rgba(173, 216, 230, 0.9)",
            bordercolor="blue",
            borderwidth=1,
            font=dict(size=10)
        )

    def _add_activity_cluster_visualization(self, fig):
        """Add activity-time cluster visualization."""
        activity_clusters = self.clusters['activity_time']

        # Count total clusters
        total_clusters = sum(len(clusters)
                             for clusters in activity_clusters.values())

        fig.add_annotation(
            text=f"🎯 Activity Clusters<br>{len(activity_clusters)} activities<br>{total_clusters} time clusters",
            xref="paper", yref="paper",
            x=0.02, y=0.90,
            xanchor="left", yanchor="top",
            showarrow=False,
            bgcolor="rgba(144, 238, 144, 0.8)",
            bordercolor="green",
            borderwidth=1,
            font=dict(size=10)
        )

    def _add_resource_pattern_visualization(self, fig):
        """Add resource pattern visualization."""
        resource_patterns = self.clusters['resource_time']

        total_clusters = sum(len(clusters)
                             for clusters in resource_patterns.values())

        fig.add_annotation(
            text=f"👥 Resource Patterns<br>{len(resource_patterns)} resources<br>{total_clusters} shift periods",
            xref="paper", yref="paper",
            x=0.98, y=0.90,
            xanchor="right", yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 215, 0, 0.8)",
            bordercolor="orange",
            borderwidth=1,
            font=dict(size=10)
        )

    def get_summary_text(self) -> str:
        """
        Generate human-readable summary text of detected clusters.

        Returns:
            Formatted string describing all detected patterns
        """
        if not self.clusters:
            return "No temporal clusters detected."

        summary = []
        summary.append(
            f"=== Temporal Cluster Analysis ({self.x_axis} × {self.y_axis}) ===\n")

        # Temporal bursts
        if 'temporal_bursts' in self.clusters:
            bursts = self.clusters['temporal_bursts']
            summary.append(
                f"📊 **Temporal Bursts Detected:** {len(bursts)} burst periods")
            for i, burst in enumerate(bursts[:5], 1):  # Show top 5
                summary.append(
                    f"   Burst {i}: {burst['event_count']} events in "
                    f"{burst['duration_seconds']:.1f}s "
                    f"({burst['start_time'].strftime('%Y-%m-%d %H:%M:%S')})"
                )

        # Activity-time clusters
        if 'activity_time' in self.clusters:
            summary.append(
                f"\n🎯 **Activity-Time Patterns:** {len(self.clusters['activity_time'])} activities with temporal clustering")
            for activity, clusters in list(self.clusters['activity_time'].items())[:5]:
                summary.append(
                    f"   '{activity}': {len(clusters)} distinct time clusters")

        # Case parallelism
        if 'case_parallelism' in self.clusters:
            para = self.clusters['case_parallelism']
            summary.append(
                f"\n⏱️ **Case Parallelism:** Max {para['max_parallel_cases']} concurrent cases, "
                f"Avg {para['avg_parallel_cases']:.1f} cases"
            )

        # Resource patterns
        if 'resource_time' in self.clusters:
            summary.append(
                f"\n👥 **Resource Time Patterns:** {len(self.clusters['resource_time'])} resources with shift-like behavior")

        # Variant patterns
        if 'variant_timing' in self.clusters:
            summary.append(
                f"\n🔄 **Variant Timing Differences:** {len(self.clusters['variant_timing'])} variants with distinct timing")

        return '\n'.join(summary)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get standardized pattern summary.
        
        Returns
        -------
        Dict[str, Any]
            Standardized summary with pattern_type, detected, count, and details
        """
        # Count total clusters across all types
        total_count = sum(
            len(v) if isinstance(v, (list, dict)) else 1
            for v in self.clusters.values()
        ) if self.clusters else 0

        return {
            'pattern_type': 'temporal_cluster',
            'detected': bool(self.clusters),
            'count': total_count,
            'details': {
                'x_axis': self.x_axis,
                'y_axis': self.y_axis,
                'clusters': self.clusters,
                'metadata': self.cluster_metadata,
                'summary_text': self.get_summary_text()
            }
        }
