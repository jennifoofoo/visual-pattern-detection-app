import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Set
import numpy as np


def _build_hovertemplate(labels: dict, x: str, y: str, has_pattern_info: bool = False) -> str:
    """Build hovertemplate with optional pattern info."""
    base = (f"<b>{labels.get(y, y)}:</b> %{{y}}<br>"
            f"<b>{labels.get(x, x)}:</b> %{{x}}")
    if has_pattern_info:
        # customdata[1] contains _pattern_info - only show if not empty
        base += "<br>%{customdata[1]}"
    return base + "<extra></extra>"


def _apply_consistent_xaxis_format(fig: go.Figure, df, x: str) -> go.Figure:
    """Apply consistent x-axis tick format based on the data range."""
    import pandas as pd

    # Only apply to datetime columns
    if not pd.api.types.is_datetime64_any_dtype(df[x]):
        return fig

    # Calculate the time span
    time_range = df[x].max() - df[x].min()
    days = time_range.days if hasattr(time_range, 'days') else time_range.total_seconds() / 86400

    # Choose tick format based on data span
    if days > 365 * 2:  # More than 2 years: show years only
        fig.update_xaxes(tickformat="%Y", dtick="M12")
    elif days > 365:  # 1-2 years: show year with optional month
        fig.update_xaxes(tickformat="%b %Y", dtick="M6")
    elif days > 90:  # 3-12 months
        fig.update_xaxes(tickformat="%b %Y", dtick="M3")
    elif days > 30:  # 1-3 months
        fig.update_xaxes(tickformat="%d %b", dtick="M1")
    else:  # Less than a month
        fig.update_xaxes(tickformat="%d %b")

    return fig


def plot_dotted_chart(
    df,
    x: str,
    y: str,
    color: str,
    title: str,
    labels: dict,
    hover_data: Optional[List[str]] = None,
    custom_data: Optional[List[str]] = None,
    focus_mode_indices: Optional[Set[int]] = None
) -> go.Figure:
    """Create a dotted chart. If focus_mode_indices provided, non-pattern points are grayed out."""

    # Check if we have pattern info in custom_data
    has_pattern_info = custom_data and '_pattern_info' in custom_data

    if focus_mode_indices is None or len(focus_mode_indices) == 0:
        # Normal mode - use px.scatter and update hovertemplate
        fig = px.scatter(df, x=x, y=y, color=color, title=title, labels=labels,
                          hover_data=hover_data, custom_data=custom_data)
        # Update hovertemplate to include pattern info
        if has_pattern_info:
            fig.update_traces(
                hovertemplate=_build_hovertemplate(labels, x, y, has_pattern_info=True)
            )
        # Fix x-axis tick format for datetime columns to ensure consistency
        fig = _apply_consistent_xaxis_format(fig, df, x)
        return fig

    # Focus mode: Use px.scatter first to get correct axis handling, then modify colors
    # Add a helper column to mark focused vs non-focused points
    df = df.copy()
    df['_is_focused'] = df.index.isin(focus_mode_indices)

    # Create chart with px.scatter (preserves correct categorical axis handling)
    fig = px.scatter(df, x=x, y=y, color=color, title=title, labels=labels,
                      hover_data=hover_data, custom_data=custom_data)

    # Update hovertemplate
    if has_pattern_info:
        fig.update_traces(
            hovertemplate=_build_hovertemplate(labels, x, y, has_pattern_info=True)
        )

    # Now modify the marker colors: gray out non-focused points
    # Each trace corresponds to one color category (activity)
    gray = 'rgba(150,150,150,0.5)'

    for trace in fig.data:
        # Get the activity/color value for this trace
        trace_name = trace.name

        # Find which points in this trace are focused vs not
        # trace.customdata contains the custom_data values, first column is _point_id
        if trace.customdata is not None and len(trace.customdata) > 0:
            # Get the point_ids from customdata (first column)
            point_ids = [cd[0] for cd in trace.customdata]

            # Create color array: original color for focused, gray for non-focused
            original_color = trace.marker.color
            colors = []
            for pid in point_ids:
                if pid in focus_mode_indices:
                    colors.append(original_color)
                else:
                    colors.append(gray)

            # Update marker colors
            trace.marker.color = colors

    # Fix x-axis tick format for datetime columns to ensure consistency
    fig = _apply_consistent_xaxis_format(fig, df, x)

    return fig
