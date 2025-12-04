"""
Example: How to register a custom visualization function.

This demonstrates the simple API for adding custom visualizations
without creating a full Pattern class.
"""

import pandas as pd
import plotly.graph_objects as go
from core.visualization.registry import register_visualization


# Example 1: Simple weekend highlighting
def highlight_weekends(df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    """
    Highlight weekend periods in the chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log dataframe
    fig : go.Figure
        Plotly figure to annotate
        
    Returns
    -------
    go.Figure
        Figure with weekend highlights added
    """
    if 'actual_time' not in df.columns:
        return fig  # Can't highlight without time column
    
    # Find weekend periods
    df['is_weekend'] = pd.to_datetime(df['actual_time']).dt.weekday >= 5
    
    # Add vertical lines for weekends (simplified example)
    weekend_dates = df[df['is_weekend']]['actual_time'].unique()
    
    for date in weekend_dates[:10]:  # Limit to first 10 for performance
        fig.add_vline(
            x=date,
            line_dash="dash",
            line_color="gray",
            opacity=0.3,
            annotation_text="Weekend"
        )
    
    return fig


# Example 2: Custom annotation for specific cases
def highlight_case(df: pd.DataFrame, fig: go.Figure, case_id: str = None) -> go.Figure:
    """
    Highlight a specific case in the chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log dataframe
    fig : go.Figure
        Plotly figure to annotate
    case_id : str, optional
        Case ID to highlight (if None, highlights first case)
        
    Returns
    -------
    go.Figure
        Figure with case highlighted
    """
    if 'case_id' not in df.columns:
        return fig
    
    if case_id is None:
        case_id = df['case_id'].iloc[0]
    
    case_df = df[df['case_id'] == case_id]
    
    if len(case_df) == 0:
        return fig
    
    # Add annotation
    fig.add_annotation(
        x=case_df['actual_time'].iloc[0] if 'actual_time' in case_df.columns else 0,
        y=case_df['case_id'].iloc[0] if 'case_id' in case_df.columns else 0,
        text=f"Case: {case_id}",
        showarrow=True,
        arrowhead=2,
        bgcolor="yellow",
        opacity=0.7
    )
    
    return fig


# Example 3: Custom trace overlay
def add_custom_trace(df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    """
    Add a custom trace to the chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        Event log dataframe
    fig : go.Figure
        Plotly figure to annotate
        
    Returns
    -------
    go.Figure
        Figure with custom trace added
    """
    # Example: Add a trend line
    if 'actual_time' in df.columns and 'activity' in df.columns:
        # Count events per day
        df['date'] = pd.to_datetime(df['actual_time']).dt.date
        daily_counts = df.groupby('date').size()
        
        if len(daily_counts) > 0:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(daily_counts.index),
                y=daily_counts.values,
                mode='lines',
                name='Daily Event Count',
                line=dict(color='blue', width=2),
                yaxis='y2'  # Secondary y-axis
            ))
            
            # Add secondary y-axis
            fig.update_layout(yaxis2=dict(
                title="Event Count",
                overlaying='y',
                side='right'
            ))
    
    return fig


# Register the visualizations
if __name__ == "__main__":
    # Register weekend highlighting
    register_visualization("weekend_highlights", highlight_weekends)
    
    # Register case highlighting (with closure to pass case_id)
    def highlight_first_case(df, fig):
        return highlight_case(df, fig, case_id=None)
    
    register_visualization("case_highlight", highlight_first_case)
    
    # Register custom trace
    register_visualization("daily_trend", add_custom_trace)
    
    print("✅ Custom visualizations registered!")
    print("Available visualizations:", VisualizationRegistry.list_registered())

