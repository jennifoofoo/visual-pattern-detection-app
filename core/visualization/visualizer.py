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
    if focus_mode_indices is None:
        return px.scatter(df, x=x, y=y, color=color, title=title, labels=labels,
                          hover_data=hover_data, custom_data=custom_data)

    # Focus mode: per-point coloring
    colors_palette = px.colors.qualitative.Plotly
    unique_vals = df[color].unique()
    val_to_color = {v: colors_palette[i % len(colors_palette)] for i, v in enumerate(unique_vals)}
    gray = 'rgba(150,150,150,0.3)'

    point_colors = [val_to_color[df.loc[idx, color]] if idx in focus_mode_indices else gray
                    for idx in df.index]

    fig = go.Figure(go.Scatter(
        x=df[x], y=df[y], mode='markers',
        marker=dict(color=point_colors, size=5, opacity=0.8),
        customdata=df[custom_data].values if custom_data else None,
        hovertemplate=(f"<b>{labels.get(y, y)}:</b> %{{y}}<br>"
                       f"<b>{labels.get(x, x)}:</b> %{{x}}<extra></extra>")
    ))
    fig.update_layout(title=title, xaxis_title=labels.get(x, x), yaxis_title=labels.get(y, y))
    return fig
