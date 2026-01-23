import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Set


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
    if focus_mode_indices is None or len(focus_mode_indices) == 0:
        return px.scatter(df, x=x, y=y, color=color, title=title, labels=labels,
                          hover_data=hover_data, custom_data=custom_data)

    # Focus mode: Create separate traces per activity to preserve legend
    colors_palette = px.colors.qualitative.Plotly
    unique_vals = df[color].unique()
    val_to_color = {v: colors_palette[i % len(colors_palette)] for i, v in enumerate(unique_vals)}
    gray = 'rgba(150,150,150,0.5)'

    fig = go.Figure()

    # Create two traces per activity value: one for focused points, one for non-focused
    for i, val in enumerate(unique_vals):
        # Get all points for this activity value
        mask = df[color] == val
        df_val = df[mask]

        # Split into focused and non-focused points
        focused_mask = df_val.index.isin(focus_mode_indices)
        df_focused = df_val[focused_mask]
        df_non_focused = df_val[~focused_mask]

        # Add focused points with original color (visible in legend)
        if len(df_focused) > 0:
            fig.add_trace(go.Scatter(
                x=df_focused[x], y=df_focused[y], mode='markers',
                marker=dict(color=val_to_color[val], size=5, opacity=0.8),
                name=str(val),  # Activity name in legend
                legendgroup=str(val),  # Group for toggling
                showlegend=True,
                customdata=df_focused[custom_data].values if custom_data else None,
                hovertemplate=(f"<b>{labels.get(y, y)}:</b> %{{y}}<br>"
                             f"<b>{labels.get(x, x)}:</b> %{{x}}<extra></extra>")
            ))

        # Add non-focused points in gray (hide from legend to avoid duplicates)
        if len(df_non_focused) > 0:
            fig.add_trace(go.Scatter(
                x=df_non_focused[x], y=df_non_focused[y], mode='markers',
                marker=dict(color=gray, size=5, opacity=0.8),
                name=str(val),  # Same name for grouping
                legendgroup=str(val),  # Same group
                showlegend=False,  # Don't show in legend (only show focused trace)
                customdata=df_non_focused[custom_data].values if custom_data else None,
                hovertemplate=(f"<b>{labels.get(y, y)}:</b> %{{y}}<br>"
                             f"<b>{labels.get(x, x)}:</b> %{{x}}<extra></extra>")
            ))

    fig.update_layout(title=title, xaxis_title=labels.get(x, x), yaxis_title=labels.get(y, y))
    return fig
