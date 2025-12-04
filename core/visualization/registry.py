"""
Visualization Registry for custom visualization callbacks.

This module provides a simple way to register custom visualization functions
without needing to create a full Pattern class. Perfect for quick prototypes
or custom visualizations that don't need full pattern detection logic.
"""

from typing import Callable, Dict, Optional
import pandas as pd
import plotly.graph_objects as go


class VisualizationRegistry:
    """
    Registry for custom visualization functions.
    
    Allows registering simple visualization callbacks that will be automatically
    applied to charts. Each visualization function receives (df, fig) and returns fig.
    """
    
    _visualizations: Dict[str, Callable[[pd.DataFrame, go.Figure], go.Figure]] = {}
    _visibility: Dict[str, bool] = {}  # Track visibility per visualization
    
    @classmethod
    def register(
        cls,
        name: str,
        visualize_func: Callable[[pd.DataFrame, go.Figure], go.Figure],
        visible: bool = True
    ) -> None:
        """
        Register a custom visualization function.
        
        Parameters
        ----------
        name : str
            Unique name for the visualization (used as key)
        visualize_func : callable
            Function that takes (df: pd.DataFrame, fig: go.Figure) and returns go.Figure
            Example:
                def my_viz(df, fig):
                    fig.add_trace(...)
                    return fig
        visible : bool, default True
            Whether the visualization should be visible by default
            
        Examples
        --------
        >>> def my_custom_viz(df, fig):
        ...     fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3]))
        ...     return fig
        >>> VisualizationRegistry.register("my_pattern", my_custom_viz)
        """
        if not callable(visualize_func):
            raise TypeError(f"visualize_func must be callable, got {type(visualize_func)}")
        
        # Validate function signature (basic check)
        import inspect
        sig = inspect.signature(visualize_func)
        if len(sig.parameters) < 2:
            raise ValueError(
                f"visualize_func must accept at least 2 parameters (df, fig), "
                f"got {len(sig.parameters)}"
            )
        
        cls._visualizations[name] = visualize_func
        cls._visibility[name] = visible
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a visualization by name."""
        if name in cls._visualizations:
            del cls._visualizations[name]
        if name in cls._visibility:
            del cls._visibility[name]
    
    @classmethod
    def set_visibility(cls, name: str, visible: bool) -> None:
        """Set visibility for a registered visualization."""
        if name not in cls._visualizations:
            raise KeyError(f"Visualization '{name}' not registered")
        cls._visibility[name] = visible
    
    @classmethod
    def is_visible(cls, name: str) -> bool:
        """Check if a visualization is visible."""
        return cls._visibility.get(name, True)
    
    @classmethod
    def apply_all(
        cls,
        df: pd.DataFrame,
        fig: go.Figure,
        filter_visible: bool = True
    ) -> go.Figure:
        """
        Apply all registered visualizations to a figure.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        fig : go.Figure
            Plotly figure to annotate
        filter_visible : bool, default True
            If True, only apply visible visualizations
            
        Returns
        -------
        go.Figure
            Figure with all registered visualizations applied
        """
        for name, visualize_func in cls._visualizations.items():
            if filter_visible and not cls.is_visible(name):
                continue
            
            try:
                fig = visualize_func(df, fig)
            except Exception as e:
                # Log error but continue with other visualizations
                print(f"Error applying visualization '{name}': {e}")
                continue
        
        return fig
    
    @classmethod
    def apply(
        cls,
        name: str,
        df: pd.DataFrame,
        fig: go.Figure
    ) -> go.Figure:
        """
        Apply a specific registered visualization.
        
        Parameters
        ----------
        name : str
            Name of the visualization to apply
        df : pd.DataFrame
            Event log dataframe
        fig : go.Figure
            Plotly figure to annotate
            
        Returns
        -------
        go.Figure
            Figure with the visualization applied
            
        Raises
        ------
        KeyError
            If visualization name is not registered
        """
        if name not in cls._visualizations:
            raise KeyError(f"Visualization '{name}' not registered")
        
        return cls._visualizations[name](df, fig)
    
    @classmethod
    def list_registered(cls) -> list:
        """Get list of all registered visualization names."""
        return list(cls._visualizations.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered visualizations."""
        cls._visualizations.clear()
        cls._visibility.clear()


# Convenience functions for easier usage
def register_visualization(
    name: str,
    visualize_func: Callable[[pd.DataFrame, go.Figure], go.Figure],
    visible: bool = True
) -> None:
    """
    Register a custom visualization function (convenience wrapper).
    
    Parameters
    ----------
    name : str
        Unique name for the visualization
    visualize_func : callable
        Function that takes (df, fig) and returns fig
    visible : bool, default True
        Whether visualization should be visible by default
        
    Examples
    --------
    >>> def highlight_weekends(df, fig):
    ...     # Add weekend highlighting
    ...     return fig
    >>> register_visualization("weekend_highlights", highlight_weekends)
    """
    VisualizationRegistry.register(name, visualize_func, visible)


def unregister_visualization(name: str) -> None:
    """Unregister a visualization (convenience wrapper)."""
    VisualizationRegistry.unregister(name)


def apply_visualizations(df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    """
    Apply all registered visualizations (convenience wrapper).
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log dataframe
    fig : go.Figure
        Plotly figure to annotate
        
    Returns
    -------
    go.Figure
        Figure with all visualizations applied
    """
    return VisualizationRegistry.apply_all(df, fig)

