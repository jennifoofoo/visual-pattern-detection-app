import pandas as pd
import numpy as np
import plotly.express as px


def _ensure_case_duration(df: pd.DataFrame) -> pd.DataFrame:
    if 'timestamp' not in df.columns or 'case_id' not in df.columns:
        return df
    durations = (
        df.groupby('case_id')['timestamp']
        .agg(lambda s: (s.max() - s.min()).total_seconds() if pd.notna(s.max()) and pd.notna(s.min()) else np.nan)
    )
    out = df.copy()
    out['case_duration'] = out['case_id'].map(durations)
    return out


def make_dotted_chart(df: pd.DataFrame, view: str = "Time"):
    view_key = (view or 'Time').strip().lower()
    dfx = _ensure_case_duration(df)

    if view_key == 'time':
        if not {'timestamp', 'activity'} <= set(dfx.columns):
            raise ValueError("Time View requires 'timestamp' and 'activity'")
        fig = px.scatter(dfx, x='timestamp', y='activity', opacity=0.7, title='Dotted Chart - Time View')
    elif view_key == 'case':
        if 'case_id' not in dfx.columns or 'timestamp' not in dfx.columns:
            raise ValueError("Case View requires 'case_id' and 'timestamp'")
        fig = px.scatter(dfx, x='case_id', y='timestamp', opacity=0.7, title='Dotted Chart - Case View')
    elif view_key == 'resource':
        if not {'timestamp', 'resource'} <= set(dfx.columns):
            raise ValueError("Resource View requires 'timestamp' and 'resource'")
        fig = px.scatter(dfx, x='timestamp', y='resource', opacity=0.7, title='Dotted Chart - Resource View')
    elif view_key == 'performance':
        if 'case_duration' not in dfx.columns:
            dfx = _ensure_case_duration(dfx)
        if not {'timestamp', 'case_duration'} <= set(dfx.columns):
            raise ValueError("Performance View requires 'timestamp' and 'case_duration'")
        fig = px.scatter(dfx, x='timestamp', y='case_duration', opacity=0.7, title='Dotted Chart - Performance View (sec)')
    else:
        raise ValueError("Unknown view. Use one of: Time, Case, Resource, Performance")

    fig.update_layout(height=420, legend=dict(orientation='h'))
    return fig
