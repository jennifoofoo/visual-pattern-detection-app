"""
Simple clustering pattern detection for Dotted Charts.
Uses DataPreprocessor for data handling and focuses on clarity over complexity.
"""

from ..data_processing.preprocessor import DataPreprocessor
from .pattern_base import Pattern
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ClusterPattern(Pattern):
    """
    Simple cluster detection for dotted charts.

    Supports OPTICS and DBSCAN clustering algorithms.
    Uses DataPreprocessor for consistent data handling.
    Dynamic hyperparameters adapt to dataset characteristics.
    """

    def __init__(self, view_config: Dict[str, str], algorithm: str = 'optics', **kwargs):
        """
        Initialize simple cluster detector.

        Parameters
        ----------
        view_config : dict
            Configuration with "x" and "y" keys for chart dimensions
        algorithm : str, default 'optics'
            Clustering algorithm: 'optics' or 'dbscan'
        **kwargs : dict
            Algorithm-specific parameters
        """
        super().__init__(f"Cluster ({algorithm.upper()})", view_config)
        self.algorithm = algorithm.lower()
        self.algorithm_params = kwargs
        self.preprocessor = DataPreprocessor()

        # Set optimal default parameters
        self._set_default_params()

        # Results storage
        self.detected = None
        self.original_indices = None
        self.total_input_points = 0
        self.df = None

    def _set_default_params(self):
        """Set optimal default parameters for each algorithm."""
        defaults = {
            'optics': {
                'min_samples': 3,
                'max_eps': 0.5,
                'xi': 0.01,
                'min_cluster_size': 5
            },
            'dbscan': {
                'eps': 0.3,
                'min_samples': 3
            },
        }

        for param, value in defaults[self.algorithm].items():
            if param not in self.algorithm_params:
                self.algorithm_params[param] = value

    def _calculate_dynamic_params(self, X: np.ndarray, df: pd.DataFrame) -> None:
        """
        Calculate dynamic hyperparameters based on data characteristics.

        Similar to temporal clustering, adapts parameters to dataset size and distribution.

        Parameters
        ----------
        X : np.ndarray
            Coordinate matrix for clustering
        df : pd.DataFrame
            Original dataframe for context
        """
        n_points = len(X)

        # Adaptive min_samples: scales with dataset size
        # Floor: 3, Ceiling: 1% of dataset (max 100 for small datasets, no upper limit for large ones)
        base_size = int(np.sqrt(n_points) / 3)
        max_size = max(100, int(n_points * 0.01))  # 1% of dataset, minimum 100
        dynamic_min_samples = max(3, min(base_size, max_size))

        if self.algorithm == 'optics':
            # Update min_samples if not explicitly provided by user
            if 'min_samples' not in self.algorithm_params or self.algorithm_params.get('min_samples') == 3:
                self.algorithm_params['min_samples'] = dynamic_min_samples

            # Update min_cluster_size if not explicitly provided
            if 'min_cluster_size' not in self.algorithm_params or self.algorithm_params.get('min_cluster_size') == 5:
                self.algorithm_params['min_cluster_size'] = dynamic_min_samples

            # Auto-calculate max_eps based on data spread if not provided
            if 'max_eps' not in self.algorithm_params or self.algorithm_params.get('max_eps') == 0.5:
                # Calculate pairwise distances for a sample of points
                sample_size = min(1000, n_points)
                sample_indices = np.random.choice(
                    n_points, sample_size, replace=False)
                sample_X = X[sample_indices]

                # Calculate standard deviation of coordinates as proxy for spread
                std_x = np.std(sample_X[:, 0])
                std_y = np.std(sample_X[:, 1])
                avg_std = (std_x + std_y) / 2

                # Set max_eps as a fraction of the average std, capped at 3x std to prevent over-clustering
                # Floor at 0.1 to handle very tight distributions
                self.algorithm_params['max_eps'] = min(
                    avg_std * 3.0, max(0.1, avg_std * 0.5))

        elif self.algorithm == 'dbscan':
            # Update min_samples if not explicitly provided by user
            if 'min_samples' not in self.algorithm_params or self.algorithm_params.get('min_samples') == 3:
                self.algorithm_params['min_samples'] = dynamic_min_samples

            # Auto-calculate eps based on data distribution if not provided
            if 'eps' not in self.algorithm_params or self.algorithm_params.get('eps') == 0.3:
                # Calculate pairwise distances using std-based approach
                # Calculate standard deviation of coordinates as proxy for spread
                std_x = np.std(X[:, 0])
                std_y = np.std(X[:, 1])
                avg_std = (std_x + std_y) / 2

                # Set eps as a fraction of the average std, capped at 2x std for conservative clustering
                # Floor at 0.05 to handle very tight distributions
                self.algorithm_params['eps'] = min(
                    avg_std * 2.0, max(0.05, avg_std * 0.2))

    def detect(self, df: pd.DataFrame) -> None:
        """
        Detect clusters using the preprocessor for data handling.

        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe with x and y coordinates
        """
        if df.empty:
            self.detected = None
            self.total_input_points = 0
            return

        self.total_input_points = len(df)
        self.df = df

        try:
            # Create view_config for preprocessor
            # Use view from view_config if available, otherwise let preprocessor determine it
            preprocessor_config = {
                'x': self.view_config['x'],
                'y': self.view_config['y'],
                'scaler': 'standard'  # Better for clustering
            }
            # Include view if it's in view_config (should be part of config)
            if 'view' in self.view_config:
                preprocessor_config['view'] = self.view_config['view']

            # Use preprocessor to prepare data (automatically determines view type)
            processed_df = self.preprocessor.process(df, preprocessor_config)

            if processed_df.empty:
                self.detected = None
                return

            # Extract coordinates from processed data
            x_col = self.view_config['x']
            y_col = self.view_config['y']

            # Use processed columns (scaled/encoded) if available, otherwise fall back to original
            # Check for scaled/normalized versions first
            x_processed_col = f"{x_col}_scaled" if f"{x_col}_scaled" in processed_df.columns else x_col
            y_processed_col = f"{y_col}_code" if f"{y_col}_code" in processed_df.columns else (
                f"{y_col}_scaled" if f"{y_col}_scaled" in processed_df.columns else y_col)

            # Get the processed coordinates
            x_data = processed_df[x_processed_col].copy()
            y_data = processed_df[y_processed_col].copy()

            # Ensure they are numeric
            x_data = pd.to_numeric(x_data, errors='coerce')
            y_data = pd.to_numeric(y_data, errors='coerce')

            # Remove any NaN values
            valid_mask = pd.notna(x_data) & pd.notna(y_data)
            if not valid_mask.any():
                self.detected = None
                return

            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]
            clean_indices = processed_df.index[valid_mask]

            # Create coordinate matrix
            X = np.column_stack([x_clean, y_clean])
            self.original_indices = clean_indices.values

            if len(X) < 2:
                self.detected = None
                return

            # Calculate dynamic hyperparameters based on data characteristics
            self._calculate_dynamic_params(X, processed_df)

            # Apply clustering
            labels = self._apply_clustering(X)

            # Store results
            self.detected = {
                'labels': labels,
                'coordinates': X,
                'original_indices': self.original_indices,
                'processed_df': processed_df,
                'n_clusters': len(np.unique(labels[labels >= 0])),
                'algorithm': self.algorithm,
                'params': self.algorithm_params.copy(),
                'x_processed_col': x_processed_col,
                'y_processed_col': y_processed_col
            }

        except Exception as e:
            print(f"Error during clustering: {e}")
            import traceback
            traceback.print_exc()
            self.detected = None

    def _apply_clustering(self, X: np.ndarray) -> np.ndarray:
        """Apply the specified clustering algorithm."""
        try:
            if self.algorithm == 'optics':
                clusterer = OPTICS(**self.algorithm_params)
            elif self.algorithm == 'dbscan':
                clusterer = DBSCAN(**self.algorithm_params)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            labels = clusterer.fit_predict(X)
            return labels

        except Exception as e:
            print(f"Clustering failed: {e}")
            return np.full(len(X), -1)

    def visualize(self, df: pd.DataFrame, fig: go.Figure, selected_clusters: list = None, show_noise: bool = False) -> go.Figure:
        """
        Add simple cluster visualization with different colors.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe
        fig : go.Figure
            Plotly figure to annotate
        selected_clusters : list, optional
            List of cluster IDs to display. If None, uses st.session_state.
        show_noise : bool, default False
            Whether to visualize noise points (label -1).
        """
        if self.detected is None:
            return fig

        labels = self.detected['labels']
        original_indices = self.detected['original_indices']
        unique_labels = np.unique(labels[labels >= 0])

        # Check for selected clusters filter
        import streamlit as st
        if selected_clusters is None:
            selected_clusters = st.session_state.get('selected_OPTICS_clusters', None)

        # Simple color palette for clusters
        colors = [
            'red', 'blue', 'green', 'orange', 'purple',
            'brown', 'pink', 'gray', 'olive', 'cyan',
            'magenta', 'yellow', 'navy', 'lime', 'maroon'
        ]

        x_col = self.view_config['x']
        y_col = self.view_config['y']

        # Helper to build hover data
        def build_hover_text(data, cluster_label, cluster_size=None):
            texts = []
            for idx in data.index:
                row = data.loc[idx]
                parts = [
                    f"<b>Cluster {cluster_label}</b>" if cluster_label != 'Noise' else "<b>Noise Point</b>"]
                if cluster_size:
                    parts[0] += f" ({cluster_size} points)"
                if 'case_id' in data.columns:
                    parts.append(f"Case: {row['case_id']}")
                if 'activity' in data.columns:
                    parts.append(f"Activity: {row['activity']}")
                if 'resource' in data.columns:
                    parts.append(f"Resource: {row['resource']}")
                texts.append("<br>".join(parts))
            return texts

        # Add cluster points with different colors
        for i, label in enumerate(unique_labels):
            if selected_clusters is not None and int(label) not in selected_clusters:
                continue
                
            mask = labels == label
            if not np.any(mask):
                continue

            cluster_indices = original_indices[mask]
            cluster_data = df.loc[cluster_indices]
            cluster_size = len(cluster_data)
            color = colors[i % len(colors)]

            hover_texts = build_hover_text(cluster_data, label, cluster_size)

            fig.add_trace(go.Scatter(
                x=cluster_data[x_col],
                y=cluster_data[y_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='circle',
                    line=dict(color='black', width=1),
                    opacity=0.8
                ),
                name=f"Cluster {label} ({cluster_size})",
                showlegend=True,
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))

        # Add noise points if any AND show_noise is True
        if show_noise:
            noise_mask = labels == -1
            if np.any(noise_mask):
                noise_indices = original_indices[noise_mask]
                noise_data = df.loc[noise_indices]

                hover_texts = build_hover_text(noise_data, 'Noise')

                fig.add_trace(go.Scatter(
                    x=noise_data[x_col],
                    y=noise_data[y_col],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='lightgray',
                        symbol='x',
                        opacity=0.5
                    ),
                    name=f"Noise ({len(noise_data)})",
                    showlegend=True,
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ))

        return fig

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of detected clusters."""
        if self.detected is None:
            return {}

        labels = self.detected['labels']
        unique_labels = np.unique(labels[labels >= 0])

        summary = {
            'algorithm': self.algorithm,
            'parameters': self.algorithm_params,
            'total_clusters': len(unique_labels),
            'total_points': self.total_input_points,
            'clustered_points': int(np.sum(labels >= 0)),
            'noise_count': int(self.total_input_points - np.sum(labels >= 0)),
            'clusters': {}
        }

        # Per-cluster statistics
        for label in unique_labels:
            mask = labels == label
            summary['clusters'][int(label)] = {
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100)
            }

        # Calculate silhouette score if possible
        if len(unique_labels) > 1 and np.sum(labels >= 0) > 1:
            try:
                clustered_mask = labels >= 0
                if np.sum(clustered_mask) > 1:
                    X = self.detected['coordinates']
                    score = silhouette_score(
                        X[clustered_mask], labels[clustered_mask])
                    summary['silhouette_score'] = float(score)
            except Exception:
                summary['silhouette_score'] = None

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """
        Get standardized pattern summary.

        Returns
        -------
        Dict[str, Any]
            Standardized summary with pattern_type, detected, count, and details
        """
        cluster_summary = self.get_cluster_summary()

        return {
            'pattern_type': 'cluster',
            'detected': self.detected is not None and cluster_summary.get('total_clusters', 0) > 0,
            'count': cluster_summary.get('total_clusters', 0),
            'details': cluster_summary
        }
