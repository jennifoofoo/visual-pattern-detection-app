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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ClusterPattern(Pattern):
    """
    Simple cluster detection for dotted charts.

    Supports OPTICS, DBSCAN clustering algorithms.
    Uses DataPreprocessor for consistent data handling.
    """

    def __init__(self, view_config: Dict[str, str], algorithm: str = 'optics', **kwargs):
        """
        Initialize simple cluster detector.

        Parameters
        ----------
        view_config : dict
            Configuration with "x", "y", and optionally "color" keys for chart dimensions
            If "color" is provided, hierarchical clustering groups by color first
        algorithm : str, default 'optics'
            Clustering algorithm: 'optics', 'dbscan', or 'kmeans'
        **kwargs : dict
            Algorithm-specific parameters
        """
        super().__init__(f"Cluster ({algorithm.upper()})", view_config)
        self.algorithm = algorithm.lower()
        self.algorithm_params = kwargs
        self.preprocessor = DataPreprocessor()

        # Results storage
        self.detected = None
        self.original_indices = None

    def _calculate_dynamic_embedding_params(self, df: pd.DataFrame, column: str) -> tuple:
        """
        Calculate optimal embedding parameters based on data characteristics.

        Small dataset: 50 rows, 8 activities → embed_dim=5, max_features=5
        Medium dataset: 1000 rows, 25 activities  → embed_dim=10, max_features=7
        Large dataset: 50000 rows, 200 activities → embed_dim=25, max_features=60

        Parameters
        df : pd.DataFrame
            Input dataframe
        column : str
            Column to analyze

        Returns
        tuple
            (embed_dim, max_features)
        """
        unique_values = df[column].nunique()
        total_rows = len(df)
        # Calculate vocabulary size (unique categories)
        vocab_size = unique_values

        # Dynamic max_features calculation
        #  It defines the proportion of vocab_size to use as max_features
        vocab_ratio = self.algorithm_params.get('vocab_ratio', 0.3)
        max_features = max(5, min(100, int(vocab_size * vocab_ratio)))

        # Dynamic embed_dim calculation based on data size and complexity
        min_dim = self.algorithm_params.get('min_embed_dim', 5)
        max_dim = self.algorithm_params.get('max_embed_dim', 50)

        # Base embedding dimension on vocabulary size and data complexity
        if vocab_size <= 10:
            embed_dim = min_dim  # Small vocabulary, small embedding
        elif vocab_size <= 50:
            embed_dim = min(max_dim, max(
                min_dim, int(np.log2(vocab_size) * 2)))
        else:
            # Large vocabulary - use logarithmic scaling
            embed_dim = min(max_dim, max(
                min_dim, int(np.log10(vocab_size) * 8)))

        # Adjust based on data size
        if total_rows < 100:
            # Smaller for small datasets
            embed_dim = max(min_dim, embed_dim // 2)
        elif total_rows > 10000:
            embed_dim = min(max_dim, embed_dim + 5)  # Larger for big datasets
        return embed_dim, max_features

    def _calculate_dynamic_clustering_params(self, X: np.ndarray) -> dict:
        """
        Calculate optimal clustering parameters based on simple data characteristics.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for clustering

        Returns
        -------
        dict
            Dynamic parameters for the clustering algorithm
        """
        n_samples, n_features = X.shape

        # Simple statistical approach for parameter estimation
        # Calculate data spread using percentiles (more robust than std)
        data_range = np.percentile(X, 75, axis=0) - \
            np.percentile(X, 25, axis=0)
        mean_range = np.mean(data_range)

        # Estimate eps based on data spread
        # Adjust based on dimensionality
        if n_features <= 2:
            # Low dimensional (visual clustering) - looser for meaningful patterns
            # For dotted charts, we want visible, actionable clusters
            eps = mean_range * 1.8  # Increased from 1.1 for larger clusters
            min_samples = max(5, int(np.sqrt(n_samples) / 8)
                              )  # Adaptive min_samples, higher baseline
        else:
            # High dimensional (TF-IDF) - tighter
            eps = mean_range * 0.3
            min_samples = 2

        # Ensure minimum values for visual clustering
        if eps < 0.1:
            eps = 0.5  # Increased from 0.3
        if min_samples < 5:
            min_samples = 5  # Increased from 3        # Algorithm-specific parameter calculation
        dynamic_params = {}

        if self.algorithm == 'optics':
            # Adaptive min_cluster_size based on dataset size
            # Aim for meaningful, visible clusters (15-25 points minimum)
            if n_samples < 30:
                # Very small groups - require at least 70% to form a cluster
                adaptive_min_cluster_size = max(15, int(n_samples * 0.7))
            elif n_samples < 100:
                # Small to medium groups - fixed minimum for visibility
                adaptive_min_cluster_size = max(min_samples, 20)
            else:
                # Large groups - scale with data but keep substantial
                adaptive_min_cluster_size = max(
                    min_samples, int(n_samples * 0.15))

            dynamic_params = {
                'min_samples': min_samples,
                # Larger max_eps for looser clustering
                'max_eps': eps * 2.5,
                'xi': 0.08,  # Slightly more selective extraction
                'min_cluster_size': adaptive_min_cluster_size
            }
        elif self.algorithm == 'dbscan':
            dynamic_params = {
                'eps': eps,
                'min_samples': min_samples
            }
        elif self.algorithm == 'kmeans':
            # Simple k estimation based on sample size
            estimated_k = max(3, min(15, int(np.sqrt(n_samples / 50))))
            dynamic_params = {
                'n_clusters': estimated_k,
                'random_state': 42,
                'n_init': 10
            }

        print(
            f"Dynamic clustering params for {self.algorithm}: {dynamic_params}")
        print(f"  Data shape: {n_samples} samples, {n_features} features")
        if self.algorithm == 'optics':
            print(
                f"  Calculated eps: {eps:.4f}, min_samples: {min_samples}, min_cluster_size: {dynamic_params['min_cluster_size']}")
        else:
            print(f"  Calculated eps: {eps:.4f}, min_samples: {min_samples}")

        return dynamic_params

    def _encode_categorical_tfidf(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Create TF-IDF embeddings for categorical data.

        Parameters
        df : pd.DataFrame
            Input dataframe
        column : str
            Column to encode with TF-IDF embeddings

        Returns
        pd.DataFrame
            Dataframe with embedding columns
        """
        # Calculate dynamic parameters based on data characteristics
        embed_dim, max_features = self._calculate_dynamic_embedding_params(
            df, column)

        # Create TF-IDF vectors treating each category as a document
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r'\b\w+\b',
            lowercase=True
        )

        try:
            categories = df[column].astype(str).fillna('unknown')
            tfidf_matrix = vectorizer.fit_transform(categories)

            # Reduce dimensionality if needed
            if tfidf_matrix.shape[1] > embed_dim:
                # Disable parallel processing to avoid CPU core detection issues
                import os
                os.environ['LOKY_MAX_CPU_COUNT'] = '1'
                pca = PCA(n_components=embed_dim)
                embeddings = pca.fit_transform(tfidf_matrix.toarray())
            else:
                embeddings = tfidf_matrix.toarray()
                # Pad with zeros if we have fewer features than desired dimensions
                if embeddings.shape[1] < embed_dim:
                    padding = np.zeros(
                        (embeddings.shape[0], embed_dim - embeddings.shape[1]))
                    embeddings = np.hstack([embeddings, padding])

        except Exception as e:
            print(
                f"TF-IDF encoding failed for {column}, using frequency encoding: {e}")
            # Fallback to frequency encoding
            freq_map = df[column].value_counts().to_dict()
            encoded = df[column].map(freq_map).fillna(0)
            # Convert to embedding format
            embeddings = encoded.values.reshape(-1, 1)
            # Pad to desired dimension
            if embeddings.shape[1] < embed_dim:
                padding = np.zeros(
                    (embeddings.shape[0], embed_dim - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])

        # Create embedding columns
        embed_df = pd.DataFrame(
            embeddings,
            columns=[f"{column}_embed_{i}" for i in range(
                embeddings.shape[1])],
            index=df.index
        )

        return embed_df

    def detect(self, df: pd.DataFrame) -> bool:
        """
        Detect clusters using label encoding or hierarchical clustering.
        If color column is specified, performs hierarchical clustering by color groups.

        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe with x and y coordinates

        Returns
        -------
        bool
            True if clusters were detected, False otherwise
        """
        if df.empty:
            self.detected = None
            return False

        # Check if hierarchical clustering by color is requested
        color_col = self.view_config.get('color')
        if color_col and color_col in df.columns:
            # Use hierarchical clustering grouped by color
            return self._detect_hierarchical(df, color_col)

        try:
            # Use label encoding for visual patterns (faster and better for dotted charts)
            X, valid_indices, column_info = self._prepare_clustering_data_with_labels(
                df)

            if X is None or len(X) < 2:
                self.detected = None
                return False

            self.original_indices = valid_indices.values

            # CRITICAL: Scale features to comparable ranges
            # This prevents timestamp columns from dominating the clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            print(f"Feature scaling applied: original range [{X.min():.2e}, {X.max():.2e}] "
                  f"-> scaled range [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

            dynamic_params = self._calculate_dynamic_clustering_params(
                X_scaled)
            for param, value in dynamic_params.items():
                if param not in self.algorithm_params or self.algorithm_params[param] is None:
                    self.algorithm_params[param] = value

            # Apply clustering on scaled data
            labels = self._apply_clustering(X_scaled)

            # Debug output
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            n_noise = np.sum(labels == -1)
            print(
                f"Clustering result: {n_clusters} clusters, {n_noise} noise points out of {len(labels)} total")
            if n_clusters > 0:
                for label in unique_labels[unique_labels >= 0]:
                    count = np.sum(labels == label)
                    print(
                        f"  Cluster {label}: {count} points ({100*count/len(labels):.1f}%)")

            # Store results
            self.detected = {
                'labels': labels,
                'coordinates': X,
                'original_indices': self.original_indices,
                'n_clusters': len(np.unique(labels[labels >= 0])),
                'algorithm': self.algorithm,
                'params': self.algorithm_params.copy(),
                'column_info': column_info,
                'feature_columns': column_info.get('feature_columns', [])
            }

            # Return True if we found at least one cluster
            return self.detected['n_clusters'] > 0

        except Exception as e:
            print(f"Error during clustering: {e}")
            import traceback
            traceback.print_exc()
            self.detected = None
            return False

    def _detect_hierarchical(self, df: pd.DataFrame, color_col: str) -> None:
        """
        Hierarchical clustering: first group by color, then spatial clustering within each group.

        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        color_col : str
            Column to use for hierarchical grouping (e.g., 'resource', 'case_id')
        """
        try:
            x_col = self.view_config['x']
            y_col = self.view_config['y']

            # Get unique color groups
            color_groups = df[color_col].unique()
            print(
                f"Hierarchical clustering: {len(color_groups)} color groups in '{color_col}'")
            all_labels = np.full(len(df), -1)  # Initialize all as noise
            cluster_counter = 0
            hierarchical_info = {'groups': {}}

            # Process each color group separately
            for group_value in color_groups:
                group_mask = df[color_col] == group_value
                group_df = df[group_mask]

                if len(group_df) < 3:  # Skip tiny groups
                    print(
                        f"  Skipping '{group_value}': only {len(group_df)} points")
                    continue

                # Prepare data for this group
                X, valid_indices, column_info = self._prepare_clustering_data_with_labels(
                    group_df)

                if X is None or len(X) < 3:
                    continue

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Calculate group-specific parameters
                dynamic_params = self._calculate_dynamic_clustering_params(
                    X_scaled)
                group_params = self.algorithm_params.copy()
                for param, value in dynamic_params.items():
                    if param not in group_params or group_params[param] is None:
                        group_params[param] = value

                # Apply clustering to this group
                # Temporarily store params and restore after
                old_params = self.algorithm_params
                self.algorithm_params = group_params
                group_labels = self._apply_clustering(X_scaled)
                self.algorithm_params = old_params

                # Count clusters in this group
                unique_group_labels = np.unique(
                    group_labels[group_labels >= 0])
                n_group_clusters = len(unique_group_labels)
                n_group_noise = np.sum(group_labels == -1)

                print(
                    f"  Group '{group_value}': {n_group_clusters} clusters, {n_group_noise} noise ({len(group_df)} total)")

                # Map group labels to global labels
                for local_label in unique_group_labels:
                    # Find points with this label in the group
                    local_mask = group_labels == local_label
                    # Map to original dataframe indices
                    global_indices = valid_indices[local_mask].values
                    # Find positions in full dataframe
                    for idx in global_indices:
                        df_position = df.index.get_loc(idx)
                        all_labels[df_position] = cluster_counter

                    cluster_counter += 1

                # Store group info
                hierarchical_info['groups'][str(group_value)] = {
                    'n_clusters': n_group_clusters,
                    'n_noise': n_group_noise,
                    'total_points': len(group_df)
                }

            # Store results
            total_clusters = len(np.unique(all_labels[all_labels >= 0]))
            total_noise = np.sum(all_labels == -1)

            print(
                f"Hierarchical clustering complete: {total_clusters} total clusters, {total_noise} noise points")

            self.detected = {
                'labels': all_labels,
                'coordinates': None,  # Not applicable for hierarchical
                'original_indices': df.index.values,
                'n_clusters': total_clusters,
                'algorithm': f'{self.algorithm}_hierarchical',
                'params': self.algorithm_params.copy(),
                'column_info': {'feature_columns': [x_col, y_col, color_col]},
                'hierarchical': True,
                'color_column': color_col,
                'hierarchical_info': hierarchical_info
            }
            self.original_indices = df.index.values

            # Return True if we found at least one cluster
            return total_clusters > 0

        except Exception as e:
            print(f"Error during hierarchical clustering: {e}")
            import traceback
            traceback.print_exc()
            self.detected = None
            return False

    def _prepare_clustering_data_with_labels(self, df: pd.DataFrame) -> tuple:
        """Fast clustering prep - uses label encoding instead of TF-IDF."""
        x_col = self.view_config['x']
        y_col = self.view_config['y']

        from sklearn.preprocessing import LabelEncoder

        all_features = []
        feature_columns = []

        # X axis
        if df[x_col].dtype == 'object':
            le_x = LabelEncoder()
            x_encoded = le_x.fit_transform(df[x_col].astype(str))
            all_features.append(x_encoded.reshape(-1, 1))
            feature_columns.append(f"{x_col}_label")
        else:
            x_data = pd.to_numeric(df[x_col], errors='coerce').values
            all_features.append(x_data.reshape(-1, 1))
            feature_columns.append(x_col)

        # Y axis
        if df[y_col].dtype == 'object':
            le_y = LabelEncoder()
            y_encoded = le_y.fit_transform(df[y_col].astype(str))
            all_features.append(y_encoded.reshape(-1, 1))
            feature_columns.append(f"{y_col}_label")
        else:
            y_data = pd.to_numeric(df[y_col], errors='coerce').values
            all_features.append(y_data.reshape(-1, 1))
            feature_columns.append(y_col)

        # Combine
        X = np.hstack(all_features)
        valid_mask = ~np.isnan(X).any(axis=1)

        return X[valid_mask], df.index[valid_mask], {'feature_columns': feature_columns}

    def _apply_clustering(self, X: np.ndarray) -> np.ndarray:
        """Apply the specified clustering algorithm."""
        try:
            # Fix CPU core detection issue
            import os
            os.environ['LOKY_MAX_CPU_COUNT'] = '1'

            # Filter out embedding-specific parameters that sklearn doesn't recognize
            embedding_params = {
                'use_embeddings', 'embed_dim', 'max_features', 'min_embed_dim',
                'max_embed_dim', 'vocab_ratio', 'adaptive_params'
            }

            # Create clean parameters for sklearn algorithms
            clean_params = {k: v for k, v in self.algorithm_params.items()
                            if k not in embedding_params}

            # Disable parallel processing for all algorithms
            if self.algorithm == 'optics':
                clean_params['n_jobs'] = 1  # Force single-threaded
                clusterer = OPTICS(**clean_params)
            elif self.algorithm == 'dbscan':
                clean_params['n_jobs'] = 1  # Force single-threaded
                clusterer = DBSCAN(**clean_params)
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
        color_col = self.view_config.get('color')

        # Get the coordinates used for clustering
        coordinates = self.detected.get('coordinates')

        # Add cluster boundaries with rectangles
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if not np.any(mask):
                continue

            # Get indices of points in this cluster
            cluster_indices = original_indices[mask]

            # Get original data for these points (for display)
            # Use .loc since cluster_indices are index LABELS, not positions
            cluster_data = df.loc[cluster_indices]

            color = colors[i % len(colors)]

            # Use the clustering coordinates to calculate boundaries
            if coordinates is not None:
                # Get coordinates for this cluster
                cluster_coords = coordinates[mask]

                # Calculate boundaries in coordinate space
                coord_x_min = cluster_coords[:, 0].min()
                coord_x_max = cluster_coords[:, 0].max()

                # Map back to original data space for visualization
                # Get the actual min/max values from original data
                x_min = cluster_data[x_col].min()
                x_max = cluster_data[x_col].max()

                # Add padding based on coordinate space
                coord_x_range = coord_x_max - coord_x_min
                padding_factor = 0.1  # 10% padding

                # For x-axis
                if pd.api.types.is_datetime64_any_dtype(cluster_data[x_col]):
                    # For datetime, add proportional time padding
                    time_range = pd.to_datetime(x_max) - pd.to_datetime(x_min)
                    padding = time_range * \
                        padding_factor if time_range.total_seconds() > 0 else pd.Timedelta(hours=1)
                    x_min_padded = pd.to_datetime(x_min) - padding
                    x_max_padded = pd.to_datetime(x_max) + padding
                elif pd.api.types.is_object_dtype(cluster_data[x_col]):
                    # For categorical x-axis, use category names
                    all_x_categories = df[x_col].unique().tolist()
                    cluster_x_categories = cluster_data[x_col].unique(
                    ).tolist()
                    cluster_x_indices = [all_x_categories.index(
                        cat) for cat in cluster_x_categories if cat in all_x_categories]

                    if cluster_x_indices:
                        min_idx = min(cluster_x_indices)
                        max_idx = max(cluster_x_indices)
                        x_min_padded = all_x_categories[min_idx]
                        x_max_padded = all_x_categories[max_idx]
                    else:
                        x_min_padded = x_min
                        x_max_padded = x_max
                else:
                    # For numeric axes
                    x_range = x_max - x_min if x_max != x_min else 1
                    padding = x_range * padding_factor
                    x_min_padded = x_min - padding
                    x_max_padded = x_max + padding

                # For y-axis: handle categorical properly
                if pd.api.types.is_object_dtype(cluster_data[y_col]):
                    # For categorical y-axis, use category names
                    all_y_categories = df[y_col].unique().tolist()
                    cluster_y_categories = cluster_data[y_col].unique(
                    ).tolist()
                    cluster_y_indices = [all_y_categories.index(
                        cat) for cat in cluster_y_categories if cat in all_y_categories]

                    if cluster_y_indices:
                        min_idx = min(cluster_y_indices)
                        max_idx = max(cluster_y_indices)
                        y_min_padded = all_y_categories[min_idx]
                        y_max_padded = all_y_categories[max_idx]
                    else:
                        y_min_padded = cluster_y_categories[0]
                        y_max_padded = cluster_y_categories[-1] if len(
                            cluster_y_categories) > 1 else cluster_y_categories[0]
                else:
                    # Numeric y-axis
                    y_min = cluster_data[y_col].min()
                    y_max = cluster_data[y_col].max()
                    y_range = y_max - y_min if y_max != y_min else 1
                    padding = y_range * padding_factor
                    y_min_padded = y_min - padding
                    y_max_padded = y_max + padding
            else:
                # Fallback for hierarchical clustering (no coordinates)
                # For categorical axes, we need to get all categories from the figure/dataframe
                # to properly calculate indices for rectangles

                # For x-axis
                if pd.api.types.is_datetime64_any_dtype(cluster_data[x_col]):
                    time_range = pd.to_datetime(
                        cluster_data[x_col].max()) - pd.to_datetime(cluster_data[x_col].min())
                    padding = time_range * 0.1 if time_range.total_seconds() > 0 else pd.Timedelta(hours=1)
                    x_min_padded = pd.to_datetime(
                        cluster_data[x_col].min()) - padding
                    x_max_padded = pd.to_datetime(
                        cluster_data[x_col].max()) + padding
                elif pd.api.types.is_object_dtype(cluster_data[x_col]):
                    # For categorical x-axis (e.g., activity), get all categories and their indices
                    all_x_categories = df[x_col].unique().tolist()
                    cluster_x_categories = cluster_data[x_col].unique(
                    ).tolist()

                    # Find the indices of cluster categories in the full category list
                    cluster_x_indices = [all_x_categories.index(
                        cat) for cat in cluster_x_categories if cat in all_x_categories]

                    if cluster_x_indices:
                        min_idx = min(cluster_x_indices)
                        max_idx = max(cluster_x_indices)
                        # Use category names with half-category padding for rectangle width
                        x_min_padded = all_x_categories[min_idx]
                        x_max_padded = all_x_categories[max_idx]
                    else:
                        x_min_padded = cluster_x_categories[0]
                        x_max_padded = cluster_x_categories[-1] if len(
                            cluster_x_categories) > 1 else cluster_x_categories[0]
                else:
                    # Numeric x-axis
                    x_range = cluster_data[x_col].max(
                    ) - cluster_data[x_col].min()
                    padding = x_range * 0.1 if x_range > 0 else 0.5
                    x_min_padded = cluster_data[x_col].min() - padding
                    x_max_padded = cluster_data[x_col].max() + padding

                # For y-axis: get actual categorical values
                if pd.api.types.is_object_dtype(cluster_data[y_col]):
                    # For categorical y-axis, get all categories and their indices
                    all_y_categories = df[y_col].unique().tolist()
                    cluster_y_categories = cluster_data[y_col].unique(
                    ).tolist()

                    # Find the indices of cluster categories in the full category list
                    cluster_y_indices = [all_y_categories.index(
                        cat) for cat in cluster_y_categories if cat in all_y_categories]

                    if cluster_y_indices:
                        min_idx = min(cluster_y_indices)
                        max_idx = max(cluster_y_indices)
                        # Use category names
                        y_min_padded = all_y_categories[min_idx]
                        y_max_padded = all_y_categories[max_idx]
                    else:
                        y_min_padded = cluster_y_categories[0]
                        y_max_padded = cluster_y_categories[-1] if len(
                            cluster_y_categories) > 1 else cluster_y_categories[0]
                else:
                    y_range = cluster_data[y_col].max(
                    ) - cluster_data[y_col].min()
                    padding = y_range * 0.1 if y_range > 0 else 0.5
                    y_min_padded = cluster_data[y_col].min() - padding
                    y_max_padded = cluster_data[y_col].max() + padding

            # Add rectangle shape for cluster boundary
            # Use category names directly for categorical axes - Plotly accepts them!
            fig.add_shape(
                type="rect",
                x0=x_min_padded, x1=x_max_padded,
                y0=y_min_padded, y1=y_max_padded,
                line=dict(color=color, width=3),
                fillcolor=color,
                opacity=0.15,
                layer="below"
            )

            # Calculate center point for hover marker
            x_is_object = pd.api.types.is_object_dtype(cluster_data[x_col])
            y_is_object = pd.api.types.is_object_dtype(cluster_data[y_col])

            if x_is_object:
                center_x = cluster_data[x_col].iloc[len(cluster_data)//2]
            else:
                center_x = cluster_data[x_col].mean()

            if y_is_object:
                center_y = cluster_data[y_col].iloc[len(cluster_data)//2]
            else:
                center_y = cluster_data[y_col].mean()

            fig.add_trace(go.Scatter(
                x=[center_x],
                y=[center_y],
                mode='markers',
                marker=dict(size=20, color=color, symbol='square',
                            opacity=0.01),  # Nearly invisible
                name=f"Cluster {label} ({len(cluster_data)} pts)",
                showlegend=True,
                hovertemplate=(
                    f"<b>Cluster {label}</b><br>" +
                    f"Points: {len(cluster_data)}<br>" +
                    f"Algorithm: {self.algorithm.upper()}<br>" +
                    "<extra></extra>"
                )
            ))

        # Add noise points if any
        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_indices = original_indices[noise_mask]
            noise_data = df.loc[noise_indices]

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

        # Show hierarchical info if available
        if self.detected.get('hierarchical'):
            color_col = self.detected.get('color_column', 'color')
            n_groups = len(self.detected.get(
                'hierarchical_info', {}).get('groups', {}))
            info_text = f"Algorithm: {self.algorithm.upper()} (Hierarchical)<br>" + \
                f"Grouped by: {color_col}<br>" + \
                f"Color Groups: {n_groups}<br>" + \
                f"Total Clusters: {n_clusters}<br>" + \
                f"Noise Points: {n_noise}"
        else:
            info_text = f"Algorithm: {self.algorithm.upper()}<br>" + \
                f"Clusters: {n_clusters}<br>" + \
                f"Noise Points: {n_noise}"

        fig.add_annotation(
            text=info_text,
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
