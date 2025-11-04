"""
Data preprocessing module for preparing event logs for pattern detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict


class DataPreprocessor:
    """
    Preprocessor for event log data, specifically designed for Time View analysis.
    
    Converts timestamps and activities into numerical representations suitable
    for machine learning algorithms (e.g., clustering with OPTICS, DBSCAN).
    
    This class prepares raw event log data by:
    - Normalizing timestamps to [0, 1] range
    - Encoding activity labels as integer codes
    
    Designed to be modular and extensible for other views (Case, Resource, etc.).
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.timestamp_scaler = MinMaxScaler()
        self.activity_encoder: Dict[str, int] = {}
        self._is_fitted = False
    
    def _encode_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode activity labels as integer codes.
        
        Creates a mapping from activity labels to integers and adds
        an 'activity_code' column to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'activity' column
            
        Returns
        -------
        pd.DataFrame
            Dataframe with added 'activity_code' column
        """
        if 'activity' not in df.columns:
            raise ValueError("DataFrame must contain 'activity' column")
        
        # Build or update activity encoder
        unique_activities = df['activity'].dropna().unique()
        if not self._is_fitted:
            # First time: create mapping
            self.activity_encoder = {act: idx for idx, act in enumerate(sorted(unique_activities))}
        else:
            # Update mapping with new activities (assign new codes)
            existing_max = max(self.activity_encoder.values()) if self.activity_encoder else -1
            for act in unique_activities:
                if act not in self.activity_encoder:
                    self.activity_encoder[act] = existing_max + 1
                    existing_max += 1
        
        # Apply encoding
        df = df.copy()
        df['activity_code'] = df['activity'].map(self.activity_encoder)
        
        return df
    
    def _scale_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize timestamps to [0, 1] range.
        
        Converts timestamp column to numeric (if needed) and applies
        MinMaxScaler to normalize values between 0 and 1.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'timestamp' column
            
        Returns
        -------
        pd.DataFrame
            Dataframe with added 'timestamp_scaled' column
        """
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        df = df.copy()
        
        # Convert timestamp to numeric if it's datetime
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            timestamp_numeric = df['timestamp'].astype('int64')  # nanoseconds since epoch
        elif pd.api.types.is_numeric_dtype(df['timestamp']):
            timestamp_numeric = df['timestamp']
        else:
            # Try to parse as datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            timestamp_numeric = df['timestamp'].astype('int64')
        
        # Fit scaler on first call, transform on subsequent calls
        timestamp_values = timestamp_numeric.values.reshape(-1, 1)
        
        if not self._is_fitted:
            timestamp_scaled = self.timestamp_scaler.fit_transform(timestamp_values)
        else:
            timestamp_scaled = self.timestamp_scaler.transform(timestamp_values)
        
        df['timestamp_scaled'] = timestamp_scaled.flatten()
        
        return df
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process event log dataframe for Time View pattern detection.
        
        Applies both activity encoding and timestamp normalization.
        Returns a dataframe with added columns:
        - 'activity_code': integer codes for activities
        - 'timestamp_scaled': normalized timestamps [0, 1]
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'timestamp' and 'activity' columns
            
        Returns
        -------
        pd.DataFrame
            Processed dataframe with original columns plus 'activity_code'
            and 'timestamp_scaled'
        """
        if df.empty:
            return df.copy()
        
        # Validate required columns
        required_cols = ['timestamp', 'activity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame must contain columns: {missing_cols}")
        
        # Apply transformations
        df_processed = self._encode_activity(df)
        df_processed = self._scale_timestamp(df_processed)
        
        # Mark as fitted after first successful processing
        self._is_fitted = True
        
        return df_processed

