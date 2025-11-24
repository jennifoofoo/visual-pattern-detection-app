"""
Demo-Mode Sampling for Process-Aware Gap Detection.

Provides case-aware sampling for large event logs to enable fast demonstration.
"""

import pandas as pd


def sample_small_eventlog(
    df: pd.DataFrame,
    max_cases: int = 100,
    max_events_per_case: int = 30,
    time_col: str = 'actual_time',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample a smaller event log while preserving process structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full event log with case_id column
    max_cases : int, default 100
        Maximum number of cases to include
    max_events_per_case : int, default 30
        Maximum events per case (samples if exceeded)
    time_col : str, default 'actual_time'
        Time column to use for sorting events within cases
    random_state : int, default 42
        Random seed for reproducible sampling
    
    Returns
    -------
    pd.DataFrame
        Sampled event log preserving process structure
    """
    if df.empty or 'case_id' not in df.columns:
        return df
    
    # Get first N unique cases
    all_cases = df['case_id'].unique()
    selected_cases = all_cases[:max_cases]
    
    # Filter to selected cases
    df_subset = df[df['case_id'].isin(selected_cases)].copy()
    
    # Sample within each case if needed
    sampled_frames = []
    
    for case_id, case_df in df_subset.groupby('case_id', sort=False):
        if len(case_df) > max_events_per_case:
            # Sample events within this case
            case_sampled = case_df.sample(
                n=max_events_per_case,
                random_state=random_state
            )
            # Re-sort by time to maintain transition order
            if time_col in case_sampled.columns:
                case_sampled = case_sampled.sort_values(time_col)
        else:
            case_sampled = case_df
        
        sampled_frames.append(case_sampled)
    
    # Concatenate all sampled cases
    df_sampled = pd.concat(sampled_frames, ignore_index=True)
    
    return df_sampled

