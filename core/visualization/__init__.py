"""
Visualization module for dotted charts.
"""

from .visualizer import plot_dotted_chart
from .registry import (
    VisualizationRegistry,
    register_visualization,
    unregister_visualization,
    apply_visualizations
)

__all__ = [
    'plot_dotted_chart',
    'VisualizationRegistry',
    'register_visualization',
    'unregister_visualization',
    'apply_visualizations'
]

