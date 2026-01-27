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
                parts = [f"<b>Cluster {cluster_label}</b>" if cluster_label != 'Noise' else "<b>Noise Point</b>"]
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

            # Calculate bounding box for cluster rectangle
            x_min = cluster_data[x_col].min()
            x_max = cluster_data[x_col].max()

            # X-Axis padding
            if pd.api.types.is_datetime64_any_dtype(cluster_data[x_col]):
                time_range = pd.to_datetime(x_max) - pd.to_datetime(x_min)
                padding = time_range * 0.1 if time_range.total_seconds() > 0 else pd.Timedelta(hours=1)
                x_min_padded = pd.to_datetime(x_min) - padding
                x_max_padded = pd.to_datetime(x_max) + padding
            else:
                x_range = x_max - x_min if x_max != x_min else 1
                padding = x_range * 0.1
                x_min_padded = x_min - padding
                x_max_padded = x_max + padding

            # Y-Axis padding (handle categorical y-axis)
            if pd.api.types.is_object_dtype(cluster_data[y_col]):
                cluster_y_categories = cluster_data[y_col].unique().tolist()
                y_min_padded = cluster_y_categories[0]
                y_max_padded = cluster_y_categories[-1] if len(cluster_y_categories) > 1 else cluster_y_categories[0]
            else:
                y_range = cluster_data[y_col].max() - cluster_data[y_col].min()
                padding = y_range * 0.1 if y_range > 0 else 0.5
                y_min_padded = cluster_data[y_col].min() - padding
                y_max_padded = cluster_data[y_col].max() + padding

            # Add rectangle shape for cluster boundary
            fig.add_shape(
                type="rect",
                x0=x_min_padded, x1=x_max_padded,
                y0=y_min_padded, y1=y_max_padded,
                line=dict(color=color, width=2),
                fillcolor=color,
                opacity=0.15,
                layer="below"
            )

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