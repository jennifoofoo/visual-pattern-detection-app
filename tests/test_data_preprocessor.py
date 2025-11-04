"""
Unit tests for DataPreprocessor class.

This module tests the preprocessing functionality across all Dotted Chart views:
- Time View: timestamp + activity
- Case View: case_id + timestamp
- Resource View: timestamp + resource
- Activity View: activity + timestamp
- Performance View: timestamp + case_duration

Tests verify encoding, normalization, edge cases, and error handling.
"""

import pandas as pd
import numpy as np
import pytest
from core.data_processing.preprocessor import DataPreprocessor


@pytest.fixture
def sample_time_data():
    """Sample DataFrame for Time View testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
        'activity': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'B', 'A']
    })


@pytest.fixture
def sample_case_data():
    """Sample DataFrame for Case View testing."""
    return pd.DataFrame({
        'case_id': ['C1', 'C2', 'C1', 'C3', 'C2', 'C1', 'C3', 'C3', 'C2', 'C1'],
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H')
    })


@pytest.fixture
def sample_resource_data():
    """Sample DataFrame for Resource View testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
        'resource': ['R1', 'R2', 'R1', 'R3', 'R2', 'R1', 'R3', 'R3', 'R2', 'R1']
    })


@pytest.fixture
def sample_activity_data():
    """Sample DataFrame for Activity View testing."""
    return pd.DataFrame({
        'activity': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'B', 'A'],
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H')
    })


@pytest.fixture
def sample_performance_data():
    """Sample DataFrame for Performance View testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
        'case_duration': [100.5, 200.3, 150.7, 300.2, 180.9, 250.1, 120.4, 280.6, 160.8, 220.0]
    })


class TestTimeView:
    """Tests for Time View preprocessing."""
    
    def test_time_view_encoding_and_scaling(self, sample_time_data):
        """Test Time View: encode activity, normalize timestamp."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'timestamp',
            'y': 'activity',
            'view': 'time'
        }
        
        result = preprocessor.process(sample_time_data, view_config)
        
        # Check new columns exist
        assert 'activity_code' in result.columns
        assert 'timestamp_scaled' in result.columns
        
        # Check activity_code is integer
        assert result['activity_code'].dtype in [np.int64, np.int32]
        assert result['activity_code'].notna().all()
        
        # Check timestamp_scaled is float and in [0, 1] range
        assert result['timestamp_scaled'].dtype == np.float64
        assert (result['timestamp_scaled'] >= 0).all()
        assert (result['timestamp_scaled'] <= 1).all()
        
        # Check original columns preserved
        assert 'timestamp' in result.columns
        assert 'activity' in result.columns
        
        # Check no NaNs introduced (unless present in input)
        assert result['activity_code'].notna().all()
        assert result['timestamp_scaled'].notna().all()
    
    def test_time_view_default_behavior(self, sample_time_data):
        """Test that process() without view_config defaults to Time View."""
        preprocessor = DataPreprocessor()
        
        result = preprocessor.process(sample_time_data)
        
        # Should behave like Time View
        assert 'activity_code' in result.columns
        assert 'timestamp_scaled' in result.columns


class TestCaseView:
    """Tests for Case View preprocessing."""
    
    def test_case_view_encoding_and_scaling(self, sample_case_data):
        """Test Case View: encode case_id, normalize timestamp."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'case_id',
            'y': 'timestamp',
            'view': 'case'
        }
        
        result = preprocessor.process(sample_case_data, view_config)
        
        # Check new columns exist
        assert 'case_id_code' in result.columns
        assert 'timestamp_scaled' in result.columns
        
        # Check case_id_code is integer
        assert result['case_id_code'].dtype in [np.int64, np.int32]
        
        # Check timestamp_scaled is in [0, 1] range
        assert (result['timestamp_scaled'] >= 0).all()
        assert (result['timestamp_scaled'] <= 1).all()


class TestResourceView:
    """Tests for Resource View preprocessing."""
    
    def test_resource_view_encoding_and_scaling(self, sample_resource_data):
        """Test Resource View: encode resource, normalize timestamp."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'timestamp',
            'y': 'resource',
            'view': 'resource'
        }
        
        result = preprocessor.process(sample_resource_data, view_config)
        
        # Check new columns exist
        assert 'resource_code' in result.columns
        assert 'timestamp_scaled' in result.columns
        
        # Check resource_code is integer
        assert result['resource_code'].dtype in [np.int64, np.int32]
        
        # Check timestamp_scaled is in [0, 1] range
        assert (result['timestamp_scaled'] >= 0).all()
        assert (result['timestamp_scaled'] <= 1).all()


class TestActivityView:
    """Tests for Activity View preprocessing."""
    
    def test_activity_view_encoding_and_scaling(self, sample_activity_data):
        """Test Activity View: encode activity, normalize timestamp."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'activity',
            'y': 'timestamp',
            'view': 'activity'
        }
        
        result = preprocessor.process(sample_activity_data, view_config)
        
        # Check new columns exist
        assert 'activity_code' in result.columns
        assert 'timestamp_scaled' in result.columns
        
        # Check activity_code is integer
        assert result['activity_code'].dtype in [np.int64, np.int32]
        
        # Check timestamp_scaled is in [0, 1] range
        assert (result['timestamp_scaled'] >= 0).all()
        assert (result['timestamp_scaled'] <= 1).all()


class TestPerformanceView:
    """Tests for Performance View preprocessing."""
    
    def test_performance_view_scaling(self, sample_performance_data):
        """Test Performance View: normalize both columns with StandardScaler."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'timestamp',
            'y': 'case_duration',
            'view': 'performance'
        }
        
        result = preprocessor.process(sample_performance_data, view_config)
        
        # Check new columns exist
        assert 'timestamp_scaled' in result.columns
        assert 'case_duration_scaled' in result.columns
        
        # Check both are float
        assert result['timestamp_scaled'].dtype == np.float64
        assert result['case_duration_scaled'].dtype == np.float64
        
        # StandardScaler centers around mean ≈ 0
        # Allow some tolerance for floating point precision
        assert abs(result['timestamp_scaled'].mean()) < 0.01
        assert abs(result['case_duration_scaled'].mean()) < 0.01
        
        # Check no NaNs introduced
        assert result['timestamp_scaled'].notna().all()
        assert result['case_duration_scaled'].notna().all()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test that empty DataFrame returns empty DataFrame without errors."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'timestamp',
            'y': 'activity',
            'view': 'time'
        }
        
        empty_df = pd.DataFrame(columns=['timestamp', 'activity'])
        result = preprocessor.process(empty_df, view_config)
        
        assert result.empty
        assert len(result.columns) >= 2  # At least original columns
    
    def test_missing_required_columns(self, sample_time_data):
        """Test that missing required columns raise ValueError."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'timestamp',
            'y': 'missing_column',
            'view': 'time'
        }
        
        with pytest.raises(ValueError, match="DataFrame must contain columns"):
            preprocessor.process(sample_time_data, view_config)
    
    def test_invalid_view_type(self, sample_time_data):
        """Test that invalid view type raises ValueError."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'timestamp',
            'y': 'activity',
            'view': 'invalid_view'
        }
        
        with pytest.raises(ValueError, match="Unknown view type"):
            preprocessor.process(sample_time_data, view_config)
    
    def test_missing_view_config_keys(self, sample_time_data):
        """Test that missing required keys in view_config raise ValueError."""
        preprocessor = DataPreprocessor()
        
        # Missing 'view' key
        incomplete_config = {
            'x': 'timestamp',
            'y': 'activity'
        }
        
        with pytest.raises(ValueError, match="view_config must contain keys"):
            preprocessor.process(sample_time_data, incomplete_config)
    
    def test_stateful_encoding_consistency(self, sample_time_data):
        """Test that encoder maintains consistency across multiple calls."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'timestamp',
            'y': 'activity',
            'view': 'time'
        }
        
        # First call
        result1 = preprocessor.process(sample_time_data, view_config)
        
        # Second call with same data
        result2 = preprocessor.process(sample_time_data, view_config)
        
        # Activity codes should be identical
        assert (result1['activity_code'] == result2['activity_code']).all()
        
        # Scaled values should be identical
        assert np.allclose(result1['timestamp_scaled'], result2['timestamp_scaled'])
    
    def test_new_categories_in_subsequent_calls(self, sample_time_data):
        """Test that new categories get new codes on subsequent calls."""
        preprocessor = DataPreprocessor()
        view_config = {
            'x': 'timestamp',
            'y': 'activity',
            'view': 'time'
        }
        
        # First call
        result1 = preprocessor.process(sample_time_data, view_config)
        original_codes = set(result1['activity_code'].unique())
        
        # Second call with new activity
        new_data = sample_time_data.copy()
        new_data.loc[0, 'activity'] = 'NEW_ACTIVITY'
        result2 = preprocessor.process(new_data, view_config)
        
        # Should have more unique codes
        new_codes = set(result2['activity_code'].unique())
        assert len(new_codes) > len(original_codes)
        assert 'NEW_ACTIVITY' in new_data['activity'].values

