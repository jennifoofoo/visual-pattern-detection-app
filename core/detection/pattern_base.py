"""
Abstract base class for pattern detection in Dotted Charts.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import plotly.graph_objects as go


class Pattern(ABC):
    """
    Abstract base class for detecting and visualizing patterns in Dotted Charts.
    
    Subclasses should implement:
    - detect(): analyze the dataframe and set self.detected
    - visualize(): overlay detected patterns on a Plotly figure
    """
    
    def __init__(self, name: str, view_config: Dict[str, str]):
        """
        Initialize a pattern detector.
        
        Parameters
        ----------
        name : str
            Name of the pattern (e.g., "Cluster", "Gap", "Trend")
        view_config : dict
            Configuration describing the Dotted Chart view attributes.
            Expected keys: "x", "y", optionally "view", "color".
            Example: {"x": "timestamp", "y": "activity", "view": "time"}
        """
        self.name = name
        self.view_config = view_config
        self.detected = None
    
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> None:
        """
        Run pattern detection on the dataframe.
        
        Should analyze df using self.view_config to identify pattern occurrences.
        Results should be stored in self.detected (format depends on pattern type).
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe with columns matching view_config (x, y, etc.)
        """
        pass
    
    @abstractmethod
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Overlay detected pattern areas on a Plotly figure.
        
        Should add visual elements (shapes, annotations, etc.) to highlight
        where the pattern was detected in the chart.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        fig : go.Figure
            Plotly figure to annotate
            
        Returns
        -------
        go.Figure
            Annotated figure with pattern overlays
        """
        pass

