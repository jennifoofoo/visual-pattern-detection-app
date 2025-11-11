import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List


def plot_dotted_chart(
    df,
    x: str,
    y: str,
    color: str,
    title: str,
    labels: dict,
    hover_data: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a dotted chart for event log visualization.

    Args:
        df: DataFrame with event data
        x: Column name for x-axis
        y: Column name for y-axis
        color: Column name for color encoding
        title: Chart title
        labels: Dictionary mapping column names to display labels
        hover_data: List of columns to include in hover tooltip

    Returns:
        Plotly Figure object
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        labels=labels,
        hover_data=hover_data
    )
    return fig
