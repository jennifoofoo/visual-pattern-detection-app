"""
Data preprocessing module for preparing event logs for pattern detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, List, Optional


class DataPreprocessor:
    """
    Modular preprocessor for event log data across all Dotted Chart views.
    
    This class prepares raw event log data for pattern detection by:
    - Encoding categorical columns as integer codes
    - Normalizing numerical columns using MinMaxScaler or StandardScaler
    
    The behavior is controlled by a `view_config` dictionary that specifies:
    - x: column name for x-axis
    - y: column name for y-axis
    - view: type of view (time, case, resource, activity, performance)
    - scaler: optional, "minmax" (default) or "standard"
    
    This modular design allows easy extension to new views by simply
    adding new view configurations.
    """
    
    def __init__(self):
        """
        Initialize the DataPreprocessor.
        
        Maintains state for encoders and scalers to ensure consistent
        transformations across multiple calls.
        """
        self.categorical_encoders: Dict[str, Dict[str, int]] = {}
        self.numerical_scalers: Dict[str, MinMaxScaler | StandardScaler] = {}
        self._is_fitted = False
    
    def _encode_categoricals(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical columns as integer codes.
        
        Creates mappings from categorical values to integers for each
        specified column. Adds new columns with suffix '_code' for each
        encoded column.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with categorical columns to encode
        cols : list of str
            List of column names to encode
            
        Returns
        -------
        pd.DataFrame
            Dataframe with added '_code' columns for each encoded column
        """
        df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain column '{col}'")
            
            # Initialize encoder for this column if not exists
            if col not in self.categorical_encoders:
                self.categorical_encoders[col] = {}
            
            # Get unique values
            unique_values = df[col].dropna().unique()
            
            if not self._is_fitted:
                # First time: create mapping
                self.categorical_encoders[col] = {
                    val: idx for idx, val in enumerate(sorted(unique_values))
                }
            else:
                # Update mapping with new values (assign new codes)
                existing_max = (
                    max(self.categorical_encoders[col].values())
                    if self.categorical_encoders[col]
                    else -1
                )
                for val in unique_values:
                    if val not in self.categorical_encoders[col]:
                        existing_max += 1
                        self.categorical_encoders[col][val] = existing_max
            
            # Apply encoding
            code_col = f"{col}_code"
            df[code_col] = df[col].map(self.categorical_encoders[col])
        
        return df
    
    def _normalize_numericals(
        self, df: pd.DataFrame, cols: List[str], method: str = "minmax"
    ) -> pd.DataFrame:
        """
        Normalize numerical columns using specified scaling method.
        
        Converts datetime columns to numeric if needed, then applies
        MinMaxScaler (default) or StandardScaler to normalize values.
        Adds new columns with suffix '_scaled' for each normalized column.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with numerical columns to normalize
        cols : list of str
            List of column names to normalize
        method : str, optional
            Scaling method: "minmax" (default) or "standard"
            
        Returns
        -------
        pd.DataFrame
            Dataframe with added '_scaled' columns for each normalized column
        """
        df = df.copy()
        
        # Initialize scaler
        if method not in ["minmax", "standard"]:
            raise ValueError(f"Scaling method must be 'minmax' or 'standard', got '{method}'")
        
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain column '{col}'")
            
            # Initialize scaler for this column if not exists
            scaler_key = f"{col}_{method}"
            if scaler_key not in self.numerical_scalers:
                if method == "minmax":
                    self.numerical_scalers[scaler_key] = MinMaxScaler()
                else:
                    self.numerical_scalers[scaler_key] = StandardScaler()
            
            scaler = self.numerical_scalers[scaler_key]
            
            # Convert to numeric if datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                numeric_values = df[col].astype('int64')  # nanoseconds since epoch
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_values = df[col]
            else:
                # Try to parse as datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                numeric_values = df[col].astype('int64')
            
            # Fit or transform
            values_reshaped = numeric_values.values.reshape(-1, 1)
            
            if not self._is_fitted:
                scaled_values = scaler.fit_transform(values_reshaped)
            else:
                scaled_values = scaler.transform(values_reshaped)
            
            # Add scaled column
            scaled_col = f"{col}_scaled"
            df[scaled_col] = scaled_values.flatten()
        
        return df
    
    def process(
        self, df: pd.DataFrame, view_config: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Process event log dataframe for pattern detection based on view configuration.
        
        Applies encoding and normalization transformations based on the view type.
        The view_config controls which columns are encoded/normalized and which
        scaling method is used.
        
        View configurations:
        - time: encode y (activity), normalize x (timestamp) with MinMax
        - case: encode x (case_id), normalize y (timestamp) with MinMax
        - resource: encode y (resource), normalize x (timestamp) with MinMax
        - activity: encode x (activity), normalize y (timestamp) with MinMax
        - performance: normalize both x and y (timestamp, case_duration) with Standard
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with columns matching view_config
        view_config : dict, optional
            Configuration dictionary with keys:
            - x: column name for x-axis
            - y: column name for y-axis
            - view: type of view (time, case, resource, activity, performance)
            - scaler: optional, "minmax" (default) or "standard"
            If None, defaults to Time View (x="timestamp", y="activity", view="time")
            
        Returns
        -------
        pd.DataFrame
            Processed dataframe with original columns plus encoded/normalized columns
        """
        if df.empty:
            return df.copy()
        
        # Default to Time View for backward compatibility
        if view_config is None:
            view_config = {
                "x": "timestamp",
                "y": "activity",
                "view": "time",
                "scaler": "minmax"
            }
        
        # Validate view_config
        required_keys = ["x", "y", "view"]
        missing_keys = [key for key in required_keys if key not in view_config]
        if missing_keys:
            raise ValueError(f"view_config must contain keys: {missing_keys}")
        
        view = view_config["view"].lower()
        x_col = view_config["x"]
        y_col = view_config["y"]
        scaler_method = view_config.get("scaler", "minmax").lower()
        
        # Validate required columns exist
        required_cols = [x_col, y_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame must contain columns: {missing_cols}")
        
        df_processed = df.copy()
        
        # Apply transformations based on view type
        if view == "time":
            # Time View: encode y (activity), normalize x (timestamp) with MinMax
            df_processed = self._encode_categoricals(df_processed, [y_col])
            df_processed = self._normalize_numericals(df_processed, [x_col], method="minmax")
        
        elif view == "case":
            # Case View: encode x (case_id), normalize y (timestamp) with MinMax
            df_processed = self._encode_categoricals(df_processed, [x_col])
            df_processed = self._normalize_numericals(df_processed, [y_col], method="minmax")
        
        elif view == "resource":
            # Resource View: encode y (resource), normalize x (timestamp) with MinMax
            df_processed = self._encode_categoricals(df_processed, [y_col])
            df_processed = self._normalize_numericals(df_processed, [x_col], method="minmax")
        
        elif view == "activity":
            # Activity View: encode x (activity), normalize y (timestamp) with MinMax
            df_processed = self._encode_categoricals(df_processed, [x_col])
            df_processed = self._normalize_numericals(df_processed, [y_col], method="minmax")
        
        elif view == "performance":
            # Performance View: normalize both x and y with StandardScaler
            df_processed = self._normalize_numericals(df_processed, [x_col, y_col], method="standard")
        
        else:
            raise ValueError(
                f"Unknown view type '{view}'. "
                f"Supported views: time, case, resource, activity, performance"
            )
        
        # Mark as fitted after first successful processing
        self._is_fitted = True
        
        return df_processed
