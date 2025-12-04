"""
Simple clustering pattern detection for Dotted Charts.
Uses DataPreprocessor for data handling and focuses on clarity over complexity.
"""

from ..data_processing.preprocessor import DataPreprocessor
from .pattern_base import Pattern
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import OPTICS, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ClusterPattern(Pattern):
    """
    Simple cluster detection for dotted charts.

    Supports OPTICS, DBSCAN, and K-means clustering algorithms.
    Uses DataPreprocessor for consistent data handling.
    """

    def __init__(self, view_config: Dict[str, str], algorithm: str = 'optics', **kwargs):
        """
        Initialize simple cluster detector.

        Parameters
        ----------
        view_config : dict
            Configuration with "x" and "y" keys for chart dimensions
        algorithm : str, default 'optics'
            Clustering algorithm: 'optics', 'dbscan', or 'kmeans'
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
            'kmeans': {
                'n_clusters': 8,
                'random_state': 42,
                'n_init': 10
            }
        }

        for param, value in defaults[self.algorithm].items():
            if param not in self.algorithm_params:
                self.algorithm_params[param] = value

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
            return

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
            y_processed_col = f"{y_col}_code" if f"{y_col}_code" in processed_df.columns else (f"{y_col}_scaled" if f"{y_col}_scaled" in processed_df.columns else y_col)
            
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

            # Debug: Check data types and values
            print(f"Data shape: {X.shape}, X dtype: {X.dtype}")
            print(f"X column: {x_processed_col}, Y column: {y_processed_col}")
            print(
                f"X range: [{np.min(X[:, 0])}, {np.max(X[:, 0])}], Y range: [{np.min(X[:, 1])}, {np.max(X[:, 1])}]")

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

    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Add simple cluster visualization with different colors.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe
        fig : go.Figure
            Plotly figure to annotate

        Returns
        -------
        go.Figure
            Figure with cluster overlays
        """
        if self.detected is None:
            return fig

        labels = self.detected['labels']
        original_indices = self.detected['original_indices']
        unique_labels = np.unique(labels[labels >= 0])

        # Simple color palette for clusters
        colors = [
            'red', 'blue', 'green', 'orange', 'purple',
            'brown', 'pink', 'gray', 'olive', 'cyan',
            'magenta', 'yellow', 'navy', 'lime', 'maroon'
        ]

        x_col = self.view_config['x']
        y_col = self.view_config['y']

        # Add cluster points with different colors
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if not np.any(mask):
                continue

            # Get indices of points in this cluster
            cluster_indices = original_indices[mask]

            # Get original data for these points
            cluster_data = df.iloc[cluster_indices]

            color = colors[i % len(colors)]

            # Add cluster points
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
                name=f"Cluster {label}",
                showlegend=True,
                hovertemplate=f"<b>Cluster {label}</b><br>" +
                f"{x_col}: %{{x}}<br>" +
                f"{y_col}: %{{y}}<extra></extra>"
            ))

        # Add noise points if any
        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_indices = original_indices[noise_mask]
            noise_data = df.iloc[noise_indices]

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
                name="Noise",
                showlegend=True,
                hovertemplate="<b>Noise Point</b><br>" +
                f"{x_col}: %{{x}}<br>" +
                f"{y_col}: %{{y}}<extra></extra>"
            ))

        # Add algorithm info
        n_clusters = self.detected['n_clusters']
        n_noise = np.sum(labels == -1)

        fig.add_annotation(
            text=f"Algorithm: {self.algorithm.upper()}<br>" +
                 f"Clusters: {n_clusters}<br>" +
                 f"Noise Points: {n_noise}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )

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
            'total_points': len(labels),
            'clustered_points': np.sum(labels >= 0),
            'noise_points': np.sum(labels == -1),
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