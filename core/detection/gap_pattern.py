"""
Hybrid Visual Gap pattern detection for Dotted Charts.

Detects gaps using:
- 1D interval detection for time/ratio-based X-axes or categorical Y-axes
- 2D occupancy grid detection for numeric X + numeric Y
"""

from .pattern_base import Pattern
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
from collections import deque


class GapPattern(Pattern):
    """
    Detects gaps in the visual dotted chart space using hybrid 1D/2D detection.
    
    - 1D gaps: For time/ratio-based X-axes or categorical Y-axes (interval gaps)
    - 2D gaps: For numeric X + numeric Y (true 2D empty regions using occupancy grid)
    
    Works on normalized visual coordinates (0-1 range).
    """
    
    def __init__(
        self, 
        view_config: Dict[str, str],
        min_gap_area: Optional[float] = None,
        min_gap_x_width: Optional[float] = None,
        y_is_categorical: bool = False,
        **kwargs
    ):
        """
        Initialize 2D gap detector.
        
        Parameters
        ----------
        view_config : dict
            Configuration with "x" and "y" keys for chart dimensions
        min_gap_area : float, optional
            Minimum gap area as fraction of visual space (0-1). Used for numeric Y views.
            Default: 0.01
        min_gap_x_width : float, optional
            Minimum horizontal gap width as fraction of X-axis (0-1). Used for categorical Y views.
            Default: 0.02
        y_is_categorical : bool, default False
            Whether Y-axis is categorical (determined by cardinality in app.py)
        """
        super().__init__("2D Gap Detection", view_config)
        self.min_gap_area = min_gap_area
        self.min_gap_x_width = min_gap_x_width
        self.y_is_categorical = y_is_categorical
        self.detected = None
        self.x_bins = None
        self.y_bins = None
        self.x_original = None
        self.y_original = None
        self.x_normalized = None
        self.y_normalized = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.y_category_order = None
        self.x_is_datetime = False
        self.threshold_used = None
    
    def _is_time_like(self, x_series: pd.Series, x_col: str) -> bool:
        """
        Check if X-axis is time-like (datetime or time-based numeric).
        
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
    
    def _detect_gaps_1d_x(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ) -> List[Dict[str, Any]]:
        """
        Detect 1D interval gaps along X-axis (full Y-range).
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        x_col : str
            X-axis column name
        y_col : str
            Y-axis column name
            
        Returns
        -------
        list of dict
            List of gaps in normalized 0-1 coordinates
        """
        # Get X values and remove NaN
        x_vals = df[x_col].dropna()
        if len(x_vals) < 2:
            return []
        
        # Convert datetime to numeric if needed
        if pd.api.types.is_datetime64_any_dtype(x_vals):
            x_vals = x_vals.astype('int64').astype(float)
        
        # Sort and get unique values
        x_unique = sorted(x_vals.unique())
        if len(x_unique) < 2:
            return []
        
        # Calculate span and threshold
        span = x_unique[-1] - x_unique[0]
        if span < 1e-10:
            return []
        
        # Use appropriate threshold
        # For time-based 1D detection, use smaller default threshold
        # because time spans can be very large (years)
        if self.y_is_categorical:
            threshold = self.min_gap_x_width if self.min_gap_x_width is not None else 0.05
        else:
            # For numeric Y with time-based X, use area threshold but interpret as X-width
            # Default: 0.001 (0.1%) for time-based, 0.01 (1%) for other numeric
            if self._is_time_like(df[x_col], x_col):
                threshold = self.min_gap_area if self.min_gap_area is not None else 0.001
            else:
                threshold = self.min_gap_area if self.min_gap_area is not None else 0.01
        
        self.threshold_used = threshold
        threshold_absolute = threshold * span
        
        # Normalize coordinates for Y-range
        x_norm, y_norm, x_orig, y_orig = self._normalize_coordinates(df)
        
        # Get Y range in normalized coordinates
        if self.y_is_categorical:
            y_start_norm = 0.0
            y_end_norm = 1.0
        else:
            y_start_norm = y_norm.min() if len(y_norm) > 0 else 0.0
            y_end_norm = y_norm.max() if len(y_norm) > 0 else 1.0
        
        # Find gaps
        gaps = []
        for i in range(len(x_unique) - 1):
            delta = x_unique[i + 1] - x_unique[i]
            if delta > threshold_absolute:
                # Normalize gap boundaries
                x_start_norm = (x_unique[i] - x_unique[0]) / span
                x_end_norm = (x_unique[i + 1] - x_unique[0]) / span
                
                # Calculate area (normalized)
                area = (x_end_norm - x_start_norm) * (y_end_norm - y_start_norm)
                
                gaps.append({
                    'x_start': x_start_norm,
                    'x_end': x_end_norm,
                    'y_start': y_start_norm,
                    'y_end': y_end_norm,
                    'area': area,
                    'label': i + 1,  # Simple label for 1D gaps
                    'gap_type': 'time_x_interval',
                    'raw_x_start': x_unique[i],
                    'raw_x_end': x_unique[i + 1]
                })
        
        return gaps
    
    def _detect_gaps_1d_per_category(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ) -> List[Dict[str, Any]]:
        """
        Detect 1D interval gaps per category (horizontal gaps in X within each category).
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        x_col : str
            X-axis column name
        y_col : str
            Y-axis column name (categorical)
            
        Returns
        -------
        list of dict
            List of gaps in normalized 0-1 coordinates
        """
        if not self.y_category_order:
            # Normalize to get category order
            _, _, _, y_orig = self._normalize_coordinates(df)
            self.y_category_order = list(pd.unique(y_orig))
        
        gaps = []
        cat_band_width = 1.0 / len(self.y_category_order) if len(self.y_category_order) > 0 else 1.0
        
        # Detect gaps for each category
        for cat_idx, category in enumerate(self.y_category_order):
            cat_df = df[df[y_col] == category]
            if len(cat_df) < 2:
                continue
            
            # Get X values for this category
            x_vals = cat_df[x_col].dropna()
            if len(x_vals) < 2:
                continue
            
            # Convert datetime to numeric if needed
            if pd.api.types.is_datetime64_any_dtype(x_vals):
                x_vals = x_vals.astype('int64').astype(float)
            
            # Sort and get unique values
            x_unique = sorted(x_vals.unique())
            if len(x_unique) < 2:
                continue
            
            # Calculate span and threshold
            span = x_unique[-1] - x_unique[0]
            if span < 1e-10:
                continue
            
            threshold = self.min_gap_x_width if self.min_gap_x_width is not None else 0.05
            self.threshold_used = threshold
            # For 1D detection, threshold is relative to span
            threshold_absolute = threshold * span
            
            # Category band in normalized coordinates
            y_start_norm = cat_idx * cat_band_width
            y_end_norm = (cat_idx + 1) * cat_band_width
            
            # Normalize X coordinates
            x_norm, _, _, _ = self._normalize_coordinates(df)
            x_min = x_norm.min()
            x_max = x_norm.max()
            x_span = x_max - x_min if x_max > x_min else 1.0
            
            # Find gaps in this category
            for i in range(len(x_unique) - 1):
                delta = x_unique[i + 1] - x_unique[i]
                if delta > threshold_absolute:
                    # Normalize gap boundaries
                    x_start_norm = (x_unique[i] - x_min) / x_span if x_span > 0 else 0.0
                    x_end_norm = (x_unique[i + 1] - x_min) / x_span if x_span > 0 else 1.0
                    
                    # Clamp to [0, 1]
                    x_start_norm = max(0.0, min(1.0, x_start_norm))
                    x_end_norm = max(0.0, min(1.0, x_end_norm))
                    
                    # Calculate area (normalized)
                    area = (x_end_norm - x_start_norm) * (y_end_norm - y_start_norm)
                    
                    gaps.append({
                        'x_start': x_start_norm,
                        'x_end': x_end_norm,
                        'y_start': y_start_norm,
                        'y_end': y_end_norm,
                        'area': area,
                        'label': len(gaps) + 1,
                        'category_idx': cat_idx,
                        'gap_type': 'category_x_interval',
                        'raw_x_start': x_unique[i],
                        'raw_x_end': x_unique[i + 1],
                        'y_category': category
                    })
        
        return gaps
    
    def _detect_gaps_2d(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ) -> List[Dict[str, Any]]:
        """
        Detect 2D gaps using occupancy grid and connected components.
        Only used for numeric X + numeric Y.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        x_col : str
            X-axis column name
        y_col : str
            Y-axis column name
            
        Returns
        -------
        list of dict
            List of gaps in normalized 0-1 coordinates
        """
        # Normalize coordinates
        x_norm, y_norm, x_orig, y_orig = self._normalize_coordinates(df)
        
        # Store for mapping
        self.x_original = x_orig
        self.y_original = y_orig
        self.x_normalized = x_norm
        self.y_normalized = y_norm
        
        # Remove NaN values
        valid_mask = pd.notna(x_norm) & pd.notna(y_norm)
        if not valid_mask.any():
            return []
        
        x_norm_clean = x_norm[valid_mask]
        y_norm_clean = y_norm[valid_mask]
        
        # Build occupancy grid
        grid, x_bins, y_bins = self._build_grid(x_norm_clean, y_norm_clean)
        self.x_bins = x_bins
        self.y_bins = y_bins
        
        # Label connected empty components
        labels, component_stats = self._label_components(grid)
        
        # Identify background label (largest border-touching region)
        total_empty_cells = (grid == 0).sum()
        background_label = None
        max_border_size = 0
        
        for label, stats in component_stats.items():
            if not stats["touches_border"]:
                continue
            
            touches_all_borders = stats.get("touches_all_borders", False)
            size_ratio = stats["size"] / total_empty_cells if total_empty_cells > 0 else 0
            is_very_large = size_ratio > 0.8
            is_large_and_touches_all = touches_all_borders and size_ratio > 0.5
            
            if is_very_large or is_large_and_touches_all:
                if stats["size"] > max_border_size:
                    max_border_size = stats["size"]
                    background_label = label
        
        # Extract gaps
        effective_min = self.min_gap_area if self.min_gap_area is not None else 0.01
        self.threshold_used = effective_min
        gaps_visual = self._extract_gaps_numeric_y(labels, x_bins, y_bins, effective_min, background_label)
        
        return gaps_visual
    
    def _normalize_coordinates(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Normalize X and Y coordinates to 0-1 range.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
            
        Returns
        -------
        tuple
            (x_normalized, y_normalized, x_original, y_original)
        """
        x_col = self.view_config['x']
        y_col = self.view_config['y']
        
        # Get original values
        x_original = df[x_col].copy()
        y_original = df[y_col].copy()
        
        # Normalize X: numeric/datetime → float → MinMax to 0-1
        if pd.api.types.is_datetime64_any_dtype(x_original):
            self.x_is_datetime = True
            x_numeric = x_original.astype('int64').astype(float)
        else:
            self.x_is_datetime = False
            x_numeric = pd.to_numeric(x_original, errors='coerce')
        
        x_min, x_max = x_numeric.min(), x_numeric.max()
        self.x_min = x_min
        self.x_max = x_max
        
        if x_max - x_min < 1e-10:
            x_normalized = pd.Series([0.5] * len(x_numeric), index=x_numeric.index)
        else:
            x_normalized = (x_numeric - x_min) / (x_max - x_min)
        
        # Normalize Y: use y_is_categorical from constructor parameter
        if self.y_is_categorical:
            # Categorical: map to evenly spaced bands in 0-1
            # Use pd.unique() to match Plotly's rendering order
            category_order = list(pd.unique(y_original))
            
            self.y_category_order = category_order
            
            cat_to_index = {cat: idx for idx, cat in enumerate(category_order)}
            y_indices = y_original.map(cat_to_index)
            n_cats = len(category_order)
            
            if n_cats == 1:
                y_normalized = pd.Series([0.5] * len(y_original), index=y_original.index)
                self.y_min = 0.5
                self.y_max = 0.5
            else:
                # Distribute categories across bands: each category gets a band
                # Category i maps to band [i/n_cats, (i+1)/n_cats]
                # For gap detection, we use the center of each band
                cat_band_width = 1.0 / n_cats
                y_normalized = (y_indices + 0.5) * cat_band_width
                self.y_min = 0.0
                self.y_max = 1.0
        else:
            # Numeric: MinMax to 0-1
            self.y_category_order = None
            
            y_numeric = pd.to_numeric(y_original, errors='coerce')
            y_min, y_max = y_numeric.min(), y_numeric.max()
            self.y_min = y_min
            self.y_max = y_max
            
            if y_max - y_min < 1e-10:
                y_normalized = pd.Series([0.5] * len(y_numeric), index=y_numeric.index)
            else:
                y_normalized = (y_numeric - y_min) / (y_max - y_min)
        
        return x_normalized, y_normalized, x_original, y_original
    
    def _build_grid(
        self, 
        x_norm: pd.Series,
        y_norm: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build 2D occupancy grid from normalized coordinates.
        
        Parameters
        ----------
        x_norm : pd.Series
            Normalized X coordinates (0-1)
        y_norm : pd.Series
            Normalized Y coordinates (0-1)
            
        Returns
        -------
        tuple
            (grid, x_bins, y_bins) where grid[y][x] = 1 if occupied, 0 if empty
        """
        # Fixed grid resolution: 150x150 for numeric, 150x 50 for categorical (enough Y resolution)
        if self.y_is_categorical and self.y_category_order is not None:
            n_cats = len(self.y_category_order)
            n_x_bins = 150
            n_y_bins = 50  # Fixed Y resolution for categorical (not just n_cats)
        else:
            n_x_bins = 150
            n_y_bins = 150
        
        # Create uniform bins (0-1 range)
        x_bins = np.linspace(0, 1, n_x_bins + 1)
        y_bins = np.linspace(0, 1, n_y_bins + 1)
        
        # Initialize grid (0 = empty, 1 = occupied)
        grid = np.zeros((n_y_bins, n_x_bins), dtype=np.int8)
        
        # Mark occupied cells using explicit integer binning
        for i in range(len(x_norm)):
            x_val = x_norm.iloc[i]
            y_val = y_norm.iloc[i]
            
            if pd.isna(x_val) or pd.isna(y_val):
                continue
            
            # Explicit integer binning with proper clamping
            x_idx = min(int(x_val * n_x_bins), n_x_bins - 1)
            y_idx = min(int(y_val * n_y_bins), n_y_bins - 1)
            
            # Ensure non-negative
            x_idx = max(x_idx, 0)
            y_idx = max(y_idx, 0)
            
            grid[y_idx, x_idx] = 1
        
        return grid, x_bins, y_bins
    
    def _label_components(self, grid: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
        """
        Label connected empty regions using BFS flood fill.
        Records statistics for each component.
        
        Parameters
        ----------
        grid : np.ndarray
            2D grid where 0 = empty, 1 = occupied
            
        Returns
        -------
        tuple
            (labels, component_stats) where:
            - labels: 2D array with component labels (0 = occupied or unlabeled)
            - component_stats: dict mapping label -> {"size": int, "touches_border": bool}
        """
        height, width = grid.shape
        labels = np.zeros_like(grid, dtype=np.int32)
        current_label = 1
        component_stats = {}
        
        # 4-connected directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 1 or labels[y, x] != 0:
                    continue
                
                # BFS from this empty cell
                queue = deque([(y, x)])
                labels[y, x] = current_label
                touches_top = (y == 0)
                touches_bottom = (y == height - 1)
                touches_left = (x == 0)
                touches_right = (x == width - 1)
                touches_border = touches_top or touches_bottom or touches_left or touches_right
                cell_count = 1
                
                while queue:
                    cy, cx = queue.popleft()
                    
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        
                        if 0 <= ny < height and 0 <= nx < width:
                            if grid[ny, nx] == 0 and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                queue.append((ny, nx))
                                cell_count += 1
                                # Check which borders this cell touches
                                if ny == 0:
                                    touches_top = True
                                if ny == height - 1:
                                    touches_bottom = True
                                if nx == 0:
                                    touches_left = True
                                if nx == width - 1:
                                    touches_right = True
                                touches_border = True
                
                # Record statistics for this component
                touches_all_borders = touches_top and touches_bottom and touches_left and touches_right
                component_stats[current_label] = {
                    "size": cell_count,
                    "touches_border": touches_border,
                    "touches_all_borders": touches_all_borders
                }
                
                current_label += 1
        
        return labels, component_stats
    
    def _extract_gaps_categorical_y(
        self,
        labels: np.ndarray,
        x_bins: np.ndarray,
        y_bins: np.ndarray,
        min_gap_x_width: float,
        background_label: int
    ) -> List[Dict[str, Any]]:
        """
        Extract gaps for categorical Y views: horizontal gaps within category bands.
        Excludes border-touching regions.
        
        Parameters
        ----------
        labels : np.ndarray
            Labeled grid with connected components
        x_bins : np.ndarray
            X-axis bin edges
        y_bins : np.ndarray
            Y-axis bin edges
        min_gap_x_width : float
            Minimum horizontal gap width (0-1)
        background_label : int or None
            Label ID of the background region (largest border-touching component)
            
        Returns
        -------
        list of dict
            List of gaps with 'x_start', 'x_end', 'y_start', 'y_end', 'area'
        """
        gaps = []
        unique_labels = np.unique(labels[labels > 0])
        n_cats = len(self.y_category_order)
        
        if n_cats == 1:
            return []
        
        cat_band_width = 1.0 / n_cats
        
        # For each category band, find gaps that intersect with it
        for cat_idx in range(n_cats):
            cat_band_start = cat_idx * cat_band_width
            cat_band_end = (cat_idx + 1) * cat_band_width
            
            # Find which y_bins fall within this category band
            y_bin_indices_in_band = []
            for i in range(len(y_bins) - 1):
                bin_start = y_bins[i]
                bin_end = y_bins[i + 1]
                # Check if bin overlaps with category band
                if not (bin_end <= cat_band_start or bin_start >= cat_band_end):
                    y_bin_indices_in_band.append(i)
            
            if not y_bin_indices_in_band:
                continue
            
            # For each empty region, check if it has a gap within this category band
            for label in unique_labels:
                # Skip the background region (largest border-touching component)
                if label == background_label:
                    continue
                
                mask = labels == label
                y_indices, x_indices = np.where(mask)
                
                if len(y_indices) == 0:
                    continue
                
                # Check if this gap intersects with the current category band
                gap_y_min_idx = y_indices.min()
                gap_y_max_idx = y_indices.max()
                gap_y_start = y_bins[gap_y_min_idx]
                gap_y_end = y_bins[gap_y_max_idx + 1] if gap_y_max_idx + 1 < len(y_bins) else 1.0
                
                # Check if gap overlaps with category band
                if gap_y_end <= cat_band_start or gap_y_start >= cat_band_end:
                    continue
                
                # Get X range of gap within this category band (strictly from component indices)
                x_min_idx = x_indices.min()
                x_max_idx = x_indices.max()
                x_start = x_bins[x_min_idx]
                x_end = x_bins[x_max_idx + 1] if x_max_idx + 1 < len(x_bins) else 1.0
                
                # Check horizontal width threshold
                x_width = x_end - x_start
                
                if x_width >= min_gap_x_width:
                    # Clip Y to category band
                    y_start = max(gap_y_start, cat_band_start)
                    y_end = min(gap_y_end, cat_band_end)
                    area = x_width * (y_end - y_start)
                    
                    gaps.append({
                        'x_start': x_start,
                        'x_end': x_end,
                        'y_start': y_start,
                        'y_end': y_end,
                        'area': area,
                        'label': int(label),
                        'category_idx': cat_idx,
                        'gap_type': 'category_x_interval'
                    })
        
        return gaps
    
    def _extract_gaps_numeric_y(
        self,
        labels: np.ndarray,
        x_bins: np.ndarray,
        y_bins: np.ndarray,
        min_gap_area: float,
        background_label: int
    ) -> List[Dict[str, Any]]:
        """
        Extract gaps for numeric Y views: true 2D empty regions.
        Excludes border-touching regions.
        
        Parameters
        ----------
        labels : np.ndarray
            Labeled grid with connected components
        x_bins : np.ndarray
            X-axis bin edges
        y_bins : np.ndarray
            Y-axis bin edges
        min_gap_area : float
            Minimum gap area (0-1)
        background_label : int or None
            Label ID of the background region (largest border-touching component)
            
        Returns
        -------
        list of dict
            List of gaps with 'x_start', 'x_end', 'y_start', 'y_end', 'area'
        """
        gaps = []
        unique_labels = np.unique(labels[labels > 0])
        
        for label in unique_labels:
            # Skip the background region (largest border-touching component)
            if background_label is not None and label == background_label:
                continue
            
            mask = labels == label
            y_indices, x_indices = np.where(mask)
            
            if len(y_indices) == 0:
                continue
            
            # Bounding box in bin indices (strictly from component indices)
            y_min_idx = y_indices.min()
            y_max_idx = y_indices.max()
            x_min_idx = x_indices.min()
            x_max_idx = x_indices.max()
            
            # Convert to visual coordinates (0-1) with overflow handling
            x_start = x_bins[x_min_idx]
            x_end = x_bins[x_max_idx + 1] if x_max_idx + 1 < len(x_bins) else 1.0
            y_start = y_bins[y_min_idx]
            y_end = y_bins[y_max_idx + 1] if y_max_idx + 1 < len(y_bins) else 1.0
            
            # Calculate area
            area = (x_end - x_start) * (y_end - y_start)
            
            if area >= min_gap_area:
                gaps.append({
                    'x_start': x_start,
                    'x_end': x_end,
                    'y_start': y_start,
                    'y_end': y_end,
                    'area': area,
                    'label': int(label),
                    'gap_type': 'spatial_2d'
                })
        
        return gaps
    
    def _map_to_original_coordinates(
        self,
        gaps_visual: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map gaps from visual coordinates (0-1) to original coordinates.
        
        Uses linear inverse scaling: x_orig = x_min + x_norm * (x_max - x_min)
        
        Parameters
        ----------
        gaps_visual : list of dict
            Gaps in visual coordinate space (0-1)
            
        Returns
        -------
        list of dict
            Gaps in original coordinate space
        """
        gaps_original = []
        
        for gap in gaps_visual:
            gap_type = gap.get('gap_type', 'spatial_2d')  # Default for backwards compatibility
            
            # Map X coordinates
            # For 1D gaps with raw_x_start/raw_x_end, use them directly
            if gap_type in ['time_x_interval', 'category_x_interval'] and 'raw_x_start' in gap and 'raw_x_end' in gap:
                # Use raw unnormalized values for perfect restoration
                x_start_orig = gap['raw_x_start']
                x_end_orig = gap['raw_x_end']
            else:
                # Check if x_start/x_end are normalized (0-1) or already in original range
                # 1D gaps: normalized values are in [0, 1]
                # 2D gaps: values are from x_bins, typically > 1 for datetime/numeric
                x_start_norm = gap['x_start']
                x_end_norm = gap['x_end']
                
                # If values are in [0, 1] range, they are normalized (1D gaps)
                # Otherwise, they are already in original range (2D gaps from x_bins)
                if isinstance(x_start_norm, (int, float)) and 0.0 <= x_start_norm <= 1.0 and \
                   isinstance(x_end_norm, (int, float)) and 0.0 <= x_end_norm <= 1.0:
                    # 1D gaps: map normalized X back to original
                    x_start_orig = self.x_min + x_start_norm * (self.x_max - self.x_min)
                    x_end_orig = self.x_min + x_end_norm * (self.x_max - self.x_min)
                else:
                    # 2D gaps: already in original range
                    x_start_orig = gap['x_start']
                    x_end_orig = gap['x_end']
            
            # Convert back to datetime if original was datetime
            if self.x_is_datetime:
                if isinstance(x_start_orig, (int, float)):
                    x_start_orig = pd.Timestamp(int(x_start_orig))
                if isinstance(x_end_orig, (int, float)):
                    x_end_orig = pd.Timestamp(int(x_end_orig))
            
            # Map Y coordinates
            if gap_type == 'time_x_interval':
                # For time_x_interval gaps: set Y to full range (don't compute from normalized)
                if self.y_is_categorical and self.y_category_order is not None:
                    # For categorical Y: use first and last category
                    y_start_orig = self.y_category_order[0]
                    y_end_orig = self.y_category_order[-1]
                else:
                    # For numeric Y: use min/max
                    y_start_orig = self.y_min
                    y_end_orig = self.y_max
            elif gap_type == 'category_x_interval':
                # For category_x_interval gaps: use stored category directly
                # Do NOT remap normalized y_start/y_end
                y_start_orig = gap['y_category']
                y_end_orig = gap['y_category']
            elif self.y_is_categorical and self.y_category_order is not None:
                # For 1D gaps with categorical Y, y_start/y_end are normalized (0-1)
                # For 2D gaps, they might be category indices or normalized
                y_start_norm = gap['y_start']
                y_end_norm = gap['y_end']
                
                # Check if normalized (0-1) or already category labels
                if isinstance(y_start_norm, (int, float)) and 0.0 <= y_start_norm <= 1.0:
                    # Normalized: map to category
                    n_cats = len(self.y_category_order)
                    if n_cats == 1:
                        y_start_orig = self.y_category_order[0]
                        y_end_orig = self.y_category_order[0]
                    else:
                        # For 1D gaps spanning full Y-range (0.0 to 1.0), use all categories
                        if y_start_norm == 0.0 and y_end_norm == 1.0:
                            y_start_orig = self.y_category_order[0]
                            y_end_orig = self.y_category_order[-1]
                        else:
                            # Convert visual Y to category index using floor division
                            cat_band_height = 1.0 / n_cats
                            y_start_idx = int(y_start_norm // cat_band_height)
                            y_end_idx = int(y_end_norm // cat_band_height)
                            y_start_idx = max(0, min(y_start_idx, n_cats - 1))
                            y_end_idx = max(0, min(y_end_idx, n_cats - 1))
                            
                            y_start_orig = self.y_category_order[y_start_idx]
                            y_end_orig = self.y_category_order[y_end_idx]
                else:
                    # Already category labels (shouldn't happen, but handle gracefully)
                    y_start_orig = gap['y_start']
                    y_end_orig = gap['y_end']
            else:
                # Numeric: use linear inverse scaling
                y_start_orig = self.y_min + gap['y_start'] * (self.y_max - self.y_min)
                y_end_orig = self.y_min + gap['y_end'] * (self.y_max - self.y_min)
            
            # Build result dict
            result_gap = {
                'x_start': x_start_orig,
                'x_end': x_end_orig,
                'y_start': y_start_orig,
                'y_end': y_end_orig,
                'area': gap['area'],
                'label': gap['label'],
                'gap_type': gap_type
            }
            
            # Preserve raw_x_start/raw_x_end and y_category for category_x_interval gaps
            if gap_type == 'category_x_interval':
                if 'raw_x_start' in gap:
                    result_gap['raw_x_start'] = gap['raw_x_start']
                if 'raw_x_end' in gap:
                    result_gap['raw_x_end'] = gap['raw_x_end']
                if 'y_category' in gap:
                    result_gap['y_category'] = gap['y_category']
            
            # Add duration for time_x_interval gaps
            if gap_type == 'time_x_interval':
                # Calculate duration from original coordinates
                if isinstance(x_start_orig, pd.Timestamp) and isinstance(x_end_orig, pd.Timestamp):
                    duration = (x_end_orig - x_start_orig).total_seconds()
                elif isinstance(x_start_orig, (int, float)) and isinstance(x_end_orig, (int, float)):
                    duration = x_end_orig - x_start_orig
                else:
                    duration = None
                if duration is not None:
                    result_gap['duration'] = duration
            
            gaps_original.append(result_gap)
        
        return gaps_original
    
    def detect(self, df: pd.DataFrame) -> None:
        """
        Detect gaps in the visual dotted chart space using hybrid 1D/2D detection.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe with x and y coordinates
        """
        if df.empty:
            self.detected = None
            return
        
        try:
            x_col = self.view_config['x']
            y_col = self.view_config['y']
            
            if x_col not in df.columns or y_col not in df.columns:
                self.detected = None
                return
            
            # Normalize coordinates (needed for all detection methods)
            x_norm, y_norm, x_orig, y_orig = self._normalize_coordinates(df)
            
            # Store for mapping
            self.x_original = x_orig
            self.y_original = y_orig
            self.x_normalized = x_norm
            self.y_normalized = y_norm
            
            # Dispatch to appropriate detection method
            if self._is_time_like(df[x_col], x_col):
                # 1D detection for time/ratio-based X-axes
                gaps_visual = self._detect_gaps_1d_x(df, x_col, y_col)
            elif self.y_is_categorical:
                # 1D detection per category for categorical Y
                gaps_visual = self._detect_gaps_1d_per_category(df, x_col, y_col)
            else:
                # 2D detection for numeric X + numeric Y
                gaps_visual = self._detect_gaps_2d(df, x_col, y_col)
            
            if not gaps_visual:
                self.detected = None
                return
            
            # Map gaps back to original coordinates
            gaps_original = self._map_to_original_coordinates(gaps_visual)
            
            # Build result summary
            total_gaps = len(gaps_original)
            total_area = sum(g['area'] for g in gaps_original)
            avg_area = total_area / total_gaps if total_gaps > 0 else 0.0
            
            self.detected = {
                'total_gaps': total_gaps,
                'total_gap_area': total_area,
                'average_gap_area': avg_area,
                'min_gap_threshold': self.threshold_used,
                'threshold_used': self.threshold_used,
                'gaps': gaps_original
            }
            
        except Exception as e:
            self.detected = None
            raise
    
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Overlay detected 2D gaps on a Plotly figure.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        fig : go.Figure
            Plotly figure to annotate
            
        Returns
        -------
        go.Figure
            Figure with gap rectangles added
        """
        if self.detected is None or not self.detected.get('gaps'):
            return fig
        
        gaps = self.detected['gaps']
        
        # Map visualization coordinates
        # Check if Y axis uses string/categorical values in Plotly
        y_axis_type = None
        if fig.data and len(fig.data) > 0:
            # Check the first trace to see if Y is categorical (string)
            first_trace = fig.data[0]
            if hasattr(first_trace, 'y') and first_trace.y is not None:
                y_sample = first_trace.y[0] if len(first_trace.y) > 0 else None
                if isinstance(y_sample, str):
                    y_axis_type = 'categorical_string'
        
        if self.y_is_categorical and self.y_category_order is not None:
            # For categorical Y: map category index to Plotly category order
            y_to_numeric = {cat: i for i, cat in enumerate(self.y_category_order)}
            def map_y(v):
                # v is category label from original mapping
                return y_to_numeric.get(v, 0)
        elif y_axis_type == 'categorical_string':
            # For string-based Y axis: need to map string values to their position
            # Extract unique Y values from figure data in order
            y_values_ordered = []
            if fig.data and len(fig.data) > 0:
                for trace in fig.data:
                    if hasattr(trace, 'y') and trace.y is not None:
                        y_values_ordered.extend(trace.y)
            y_unique_ordered = list(dict.fromkeys(y_values_ordered))  # Preserve order, remove duplicates
            y_to_index = {val: i for i, val in enumerate(y_unique_ordered)}
            
            def map_y(v):
                # v might be numeric (from gap) or string (from original data)
                if isinstance(v, str):
                    return y_to_index.get(v, 0)
                else:
                    # Numeric value - try to find matching string or use as index
                    # For time_x_interval gaps, v is the full range (0, 1142)
                    # Map to the actual string range
                    if isinstance(v, (int, float)):
                        # Use as index into the ordered list
                        idx = int(v) if v >= 0 else 0
                        return min(idx, len(y_unique_ordered) - 1) if y_unique_ordered else 0
                    return 0
        else:
            # For numeric Y: use raw values
            def map_y(v):
                return v
        
        # Get Y axis range for time_x_interval gaps
        # Try to get actual Y data range from the figure
        y_min_range = None
        y_max_range = None
        
        # First, try to get from axis range
        if fig.layout.yaxis and fig.layout.yaxis.range:
            y_min_range = fig.layout.yaxis.range[0]
            y_max_range = fig.layout.yaxis.range[1]
        
        # If not available, extract from figure data
        if y_min_range is None or y_max_range is None:
            if fig.data and len(fig.data) > 0:
                # Get all Y values from all traces
                all_y_values = []
                for trace in fig.data:
                    if hasattr(trace, 'y') and trace.y is not None:
                        all_y_values.extend(trace.y)
                
                if all_y_values:
                    # Filter out None/NaN values
                    import numpy as np
                    all_y_values = [y for y in all_y_values if y is not None and not (isinstance(y, float) and np.isnan(y))]
                    if all_y_values:
                        y_min_range = min(all_y_values)
                        y_max_range = max(all_y_values)
        
        # Final fallback: use gap's y_start/y_end if available
        if y_min_range is None or y_max_range is None:
            if self.y_is_categorical and self.y_category_order:
                y_min_range = 0
                y_max_range = len(self.y_category_order) - 1
            else:
                # Use the gap's own y_start/y_end values (they should be the full range for time_x_interval)
                if gaps and gaps[0].get('gap_type') == 'time_x_interval':
                    y_min_range = gaps[0].get('y_start', 0)
                    y_max_range = gaps[0].get('y_end', 1)
                else:
                    y_min_range = 0
                    y_max_range = 1
        
        for gap in gaps:
            gap_type = gap.get('gap_type', 'spatial_2d')  # Default for backwards compatibility
            
            if gap_type == 'time_x_interval':
                # Draw vertical band across full Y plotting range
                # Use the gap's y_start/y_end which should be the full range
                y0_val = gap.get('y_start', y_min_range)
                y1_val = gap.get('y_end', y_max_range)
            elif gap_type == 'category_x_interval':
                # Use existing logic with map_y to restrict to category band
                y0_val = map_y(gap['y_start'])
                y1_val = map_y(gap['y_end'])
                # Ensure y0 is the minimum and y1 is the maximum
                if y0_val > y1_val:
                    y0_val, y1_val = y1_val, y0_val
            else:  # spatial_2d
                # Use gap coordinates for 2D rectangle
                y0_val = map_y(gap['y_start'])
                y1_val = map_y(gap['y_end'])
                # Ensure y0 is the minimum and y1 is the maximum
                if y0_val > y1_val:
                    y0_val, y1_val = y1_val, y0_val
            
            # Use slightly different styling for time gaps (optional)
            if gap_type == 'time_x_interval':
                fillcolor = "rgba(255, 0, 0, 0.2)"  # Increased opacity for better visibility
                line_color = "rgba(255, 0, 0, 0.5)"  # Increased line opacity
                line_width = 2  # Thicker line for time gaps
            else:
                fillcolor = "rgba(255, 0, 0, 0.15)"
                line_color = "rgba(255, 0, 0, 0.4)"
                line_width = 1
            
            fig.add_shape(
                type="rect",
                x0=gap['x_start'],
                y0=y0_val,
                x1=gap['x_end'],
                y1=y1_val,
                fillcolor=fillcolor,
                line=dict(color=line_color, width=line_width),
                layer="below"
            )
        
        return fig
    
    def get_gap_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected gaps.
        
        Returns
        -------
        dict
            Summary dictionary with gap statistics
        """
        if self.detected is None:
            threshold = self.threshold_used if self.threshold_used is not None else (self.min_gap_x_width if self.y_is_categorical else self.min_gap_area)
            if threshold is None:
                threshold = 0.02 if self.y_is_categorical else 0.01
            return {
                'total_gaps': 0,
                'total_gap_area': 0,
                'average_gap_area': 0,
                'min_gap_threshold': threshold,
                'gaps': []
            }
        
        gaps = self.detected['gaps']
        total_gaps = len(gaps)
        total_area = sum(g['area'] for g in gaps)
        avg_area = total_area / total_gaps if total_gaps > 0 else 0
        
        return {
            'total_gaps': total_gaps,
            'total_gap_area': total_area,
            'average_gap_area': avg_area,
            'min_gap_threshold': self.detected['threshold_used'],
            'gaps': gaps
        }
