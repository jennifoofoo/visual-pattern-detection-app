from .pattern_base import Pattern

import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from typing import List, Dict, Any, Optional,Tuple
from collections import defaultdict # needed for pattern counting

from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP

from prefixspan import PrefixSpan

'''
TODO:
1. improve visualize method
currently pass because too many sequences make application crash

2. add min support parameter to UI

TODO LATER: 
preprocessing: make sets of events when same timestamp
'''

class ChartConfig:
    """Stub for chart configuration, e.g., x_axis, y_axis, dot"""
    def __init__(self):

        x_axis_key, y_axis_key, dot_key, full_df = self.get_chart_config_from_streamlit_state()

        # self.x_axis_label = x_axis_input
        # self.x_axis_df_key = X_AXIS_COLUMN_MAP[x_axis_input]
        self.x_axis_df_key = x_axis_key

        # self.y_axis_label = y_axis_input
        # self.y_axis_df_key = Y_AXIS_COLUMN_MAP[y_axis_input]
        self.y_axis_df_key = y_axis_key

        # self.dot_label = dot_input
        # self.dot_df_key = DOTS_COLOR_MAP[dot_input]
        self.dot_df_key = dot_key

        # Keep the relevant columns for detection and sorting
        self.df = full_df[[self.x_axis_df_key, self.y_axis_df_key, self.dot_df_key]].copy()

    def get_chart_config_from_streamlit_state(self):
        x_axis = st.session_state.view_config['x']
        y_axis = st.session_state.view_config['y']
        dot_input = st.session_state.view_config['color']
        df_full = st.session_state.get('df', None)

        assert x_axis is not None, "X-Axis user input is missing from session state."
        assert y_axis is not None, "Y-Axis user input is missing from session state."
        assert dot_input is not None, "Dot user input is missing from session state."
        assert df_full is not None, "Event Log DataFrame (df_full) is missing from session state."

        return x_axis, y_axis, dot_input, df_full


class HorizontalSequencePatternDetector(Pattern):
    def __init__(
        self,
        min_support = 50,
    ):
        chart_config = ChartConfig() # make sure data is available in session state
        self.df_by_configuration = chart_config.df

        # find sequences on x-axis (horizontal) and use y axis for grouping
        self.sequence_detection_axis = chart_config.x_axis_df_key
        self.grouping_key = chart_config.y_axis_df_key
        self.event_key = chart_config.dot_df_key

        # validate data for squence detection
        self.warn_grouping_key_dot_config = self.grouping_key == chart_config.dot_df_key # prevent nonsensical sequences
        if self.warn_grouping_key_dot_config:
            print(
                f"Warning: Grouping Key ('{self.grouping_key}') "
                f"and Dot Input ('{chart_config.dot_df_key}') "
                f"are the same. Sequential pattern detection skipped."
            )
            return

        # detection parameters
        # self.min_support = min_support # for absolute count
        mind_support_percentage = min_support/100
        unique_count = self.df_by_configuration[self.grouping_key].nunique()
        min_support_count = math.ceil(unique_count * mind_support_percentage)
        self.min_support = min_support_count # absolute count calculated from given percentage

        # results
        self.detected = False
        self.prefix_span = None
        self.df_found_patterns = pd.DataFrame()

        self.results = pd.DataFrame()
        self.topkresults = pd.DataFrame()

    def detect(self):
        if self.warn_grouping_key_dot_config:
            return False
        # 1. Preprocessing Data for PrefixSpan
        sequence_data = self.prepare_data_for_prefixspan() 
        # print(sequence_data)
        # 2. apply PrefixSpan
        found_subsequences = self.extract_all_subsequences_prefixspan(sequence_data)
        # print(found_subsequences)
        # 3. post-processing PrefixSpan to add event details
        pattern_map_df = self.postprocess_prefixspan_results(found_subsequences)
        # print(pattern_map_df)
        # 4. generate result dataframe? --> depends on visualization needs
        # 5. save final dataframe to session state
        self.df_found_patterns = pattern_map_df

        # 6. join back to original dataframe for context
        full_df = st.session_state.get('df', pd.DataFrame())
        self.results = self._add_original_df_context(full_df, self.df_found_patterns)

        self.detected = True
        return True

    def prepare_data_for_prefixspan(self) -> Dict[Any, pd.DataFrame]:
        """Prepare data for PrefixSpan sequence detection.
        Returns:
            Dict[Any, pd.DataFrame]: A dictionary mapping each group ID to its corresponding DataFrame.
        """

        if self.warn_grouping_key_dot_config:
            return {}

        # sort by y-axis (grouping key) and x-axis (sequence detection axis)
        # needed for post-mining verification later again
        self.df_sorted = self.df_by_configuration.sort_values(
            by=[self.grouping_key, self.sequence_detection_axis], 
            ascending=[True, True]
        )

        sequences_by_group = self.df_sorted.groupby(self.grouping_key).agg(
            sequence=(self.event_key, list)
        )

        sequence_data = sequences_by_group['sequence'].tolist()

        return sequence_data

    def extract_all_subsequences_prefixspan(self, sequence_data: List[List[Any]]) -> List[List[Any]]:
        min_support_count = self.min_support

        ps = PrefixSpan(sequence_data)
        frequent_patterns = ps.frequent(min_support_count)

        self.prefix_span = ps

        return frequent_patterns

    def postprocess_prefixspan_results(
        self,
        found_subsequences: List[Tuple[int, List[Any]]],
        min_pattern_length: int = 2
    ) -> pd.DataFrame:
        """
        Perform post-mining verification to map frequent patterns back to
        specific event row indices in the original DataFrame for visualization.

        Parameters
        ----------
        found_subsequences : List[Tuple[int, List[Any]]]
            List of (support_count, pattern) tuples from PrefixSpan.
        min_pattern_length : int
            Minimum pattern length to include (default: 2).

        Returns
        -------
        pd.DataFrame
            A mapping table with 'index', 'group_id', and pattern details.
        """
        if not found_subsequences:
            return pd.DataFrame()

        # Filter to patterns with minimum length
        filtered_subsequences = [
            (support, pattern) for support, pattern in found_subsequences
            if len(pattern) >= min_pattern_length
        ]

        if not filtered_subsequences:
            return pd.DataFrame()

        # --- 1. Prepare Data for Verification ---
        df_with_index = self.df_sorted.reset_index()

        # Group the sorted data to get the sequence list and the original row indices
        grouped_data = df_with_index.groupby(self.grouping_key).agg(
            sequence_list=(self.event_key, list),
            original_indices=('index', list))

        total_groups = len(grouped_data)

        mapping_data = []
        pattern_instance_counter = 0

        # 2. Iterate through all found frequent patterns
        for support_count, pattern in filtered_subsequences:

            relative_support = support_count / total_groups

            # 3. Iterate through every single Group ID (Case ID) in the grouped data
            for group_id, group_row in grouped_data.iterrows():
                sequence = group_row['sequence_list']

                # Find all sets of event indices that form the pattern in this group
                # match_indices_list is a list of lists, where each inner list contains the 
                # positional indices within the 'sequence' list where the pattern occurred.
                match_indices_list = find_subsequence_matches(pattern, sequence)

                if match_indices_list:
                    original_df_indices = group_row['original_indices']

                    # 4. Map positional indices to original DataFrame row IDs
                    for match_indices in match_indices_list:
                        pattern_instance_counter += 1

                        # Convert positional indices [0, 2, 5] to actual row index values [10, 15, 22]
                        matched_df_indices = [original_df_indices[i] for i in match_indices]

                        # 5. Create a row in the mapping table for every matched event
                        for df_index in matched_df_indices:
                            mapping_data.append({
                                'index': df_index, # The row ID (index) of df_by_configuration
                                'group_id': group_id,
                                'pattern_instance_id': pattern_instance_counter,
                                'pattern': pattern,
                                'support_count': support_count,
                                'support_percentile': relative_support,
                                'is_part_of_pattern': True
                            })

        # 6. Create the final mapping table
        pattern_map_df = pd.DataFrame(mapping_data).set_index('index')

        return pattern_map_df
    
    def get_sequences_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for detected sequences.

        Uses topkresults if available (after get_top_k_sequences was called),
        otherwise falls back to all detected patterns in df_found_patterns.

        Returns
        -------
        Dict[str, Any]
            Summary containing total_patterns_found, max_pattern_length, and details.
        """
        # Use topkresults if available, otherwise use all patterns
        source_df = self.topkresults if not self.topkresults.empty else self.df_found_patterns
        
        if source_df.empty:
            return {'total_patterns_found': 0, 'details': {'pattern_stats': {}}}

        # 1. Group by the pattern tuple to aggregate stats
        df_stats = source_df.copy()
        df_stats['pattern_tuple'] = df_stats['pattern'].apply(tuple)
        
        pattern_groups = df_stats.groupby('pattern_tuple').agg({
            'group_id': lambda x: sorted(list(set(x))),
            'support_count': 'first',
            'support_percentile': 'first',
            'pattern_instance_id': 'nunique'
        })

        # 2. Build the pattern_stats dictionary for the UI
        pattern_stats = {}
        for pattern_tuple, row in pattern_groups.iterrows():
            pattern_str = " -> ".join(map(str, pattern_tuple))
            pattern_stats[pattern_str] = {
                'sequence': list(pattern_tuple),
                'group_ids': row['group_id'],
                'count': int(row['support_count']),
                'occurrence_count': int(row['pattern_instance_id']),
                'support_percentile': float(row['support_percentile'])
            }

        return {
            'total_patterns_found': len(source_df['pattern_instance_id'].unique()),
            'max_pattern_length': source_df['pattern'].apply(len).max(),
            'details': {
                'pattern_stats': pattern_stats
            }
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get standardized pattern summary.
        
        Returns
        -------
        Dict[str, Any]
            Standardized summary with pattern_type, detected, count, and details
        """
        sequences_summary = self.get_sequences_summary()
        
        return {
            'pattern_type': 'sequence',
            'detected': self.detected,
            'count': sequences_summary['total_patterns_found'],
            'details': sequences_summary['details']
        }

    def get_top_k_sequences(self, k: int) -> pd.DataFrame:
        """
        Returns the top k sequences found by support_count and updates self.topkresults.
        
        Parameters
        ----------
        k : int
            Number of top sequences to return (sorted by support_count descending)
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing only the top k sequences by support_count
        """
        if self.df_found_patterns.empty:
            self.topkresults = pd.DataFrame()
            return self.topkresults
        
        # Get unique patterns with their support_count
        # Group by pattern tuple to get unique support_count for each pattern
        df_stats = self.df_found_patterns.copy()
        df_stats['pattern_tuple'] = df_stats['pattern'].apply(tuple)
        
        # Aggregate to get unique patterns with their support_count
        pattern_groups = df_stats.groupby('pattern_tuple').agg({
            'support_count': 'first'  # All rows for same pattern have same support_count
        }).reset_index()
        
        # Sort by support_count descending and get top k
        pattern_groups_sorted = pattern_groups.sort_values(
            by='support_count', 
            ascending=False
        ).head(k)
        
        # Get the pattern tuples for top k
        top_k_pattern_tuples = set(pattern_groups_sorted['pattern_tuple'].tolist())
        
        # Filter df_found_patterns to only include top k patterns
        topk_df = df_stats[df_stats['pattern_tuple'].isin(top_k_pattern_tuples)].copy()
        
        # Remove the temporary pattern_tuple column
        topk_df = topk_df.drop(columns=['pattern_tuple'])
        
        # Store in instance variable
        self.topkresults = topk_df
        
        return self.topkresults

    def _add_original_df_context(self, full_df: pd.DataFrame, pattern_map_df: pd.DataFrame) -> pd.DataFrame:
            """
            Joins the detected pattern information back to the original full dataframe.
            Because an event can belong to multiple patterns, this performs a left join 
            that may result in duplicated rows (one for each pattern match).
            """
            if full_df.empty:
                return pd.DataFrame()
                
            if pattern_map_df.empty:
                # If no patterns found, return original df with empty pattern columns
                results = full_df.copy()
                results['is_part_of_pattern'] = False
                results['pattern_instance_id'] = None
                return results

            # 
            
            # Perform the join. Since pattern_map_df index refers to full_df index:
            # 'how=left' keeps all original data. 
            # If an index exists multiple times in pattern_map_df, rows are duplicated here.
            results = full_df.join(pattern_map_df, how='left')

            # Clean up and add helper columns for the UI
            results['is_part_of_pattern'] = results['is_part_of_pattern'].fillna(False)
            
            # Create a readable label for charts/tooltips
            results['pattern_display_name'] = results.apply(
                lambda x: f"Pattern {int(x['pattern_instance_id'])}: {' -> '.join(map(str, x['pattern']))}" 
                if x['is_part_of_pattern'] else "No Pattern",
                axis=1
            )

            return results
    
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Overlay lines on the Plotly figure to connect events in detected sequences.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the event data (currently unused, kept for interface).
        fig : go.Figure
            The Plotly figure to add sequence lines to.

        Returns
        -------
        go.Figure
            The figure with sequence lines added.
        """
        # Get filtered data for selected patterns
        filtered_df = self._get_selected_pattern_data()
        if filtered_df is None or filtered_df.empty:
            return fig

        # Build color map for pattern templates
        color_map = self._build_color_map(filtered_df['pattern_str'].unique())

        # Add a trace for each pattern instance
        for instance_id in filtered_df['pattern_instance_id'].unique():
            instance_data = filtered_df[filtered_df['pattern_instance_id'] == instance_id]
            instance_data = instance_data.sort_values(by=self.sequence_detection_axis)
            self._add_sequence_trace(fig, instance_data, color_map)

        return fig

    def _get_selected_pattern_data(self) -> Optional[pd.DataFrame]:
        """
        Filter results to only include user-selected patterns.

        Note: Top k filtering is applied at detection time via get_top_k_sequences(),
        so this method only needs to filter by user selection.

        Returns
        -------
        Optional[pd.DataFrame]
            Filtered dataframe with pattern_str column, or None if no valid selection.
        """
        if not self.detected or self.results.empty:
            return None

        # Get selected patterns from session state
        # Distinguish between None (not set yet) and [] (explicitly empty)
        selected_sequences = st.session_state.get('selected_seq_patterns')
        
        # If explicitly set to empty list, user deselected all - show nothing
        if selected_sequences is not None and len(selected_sequences) == 0:
            return None
        
        # Start with full results, filter to pattern rows only
        df = self.results[self.results['is_part_of_pattern'] == True].copy()
        if df.empty:
            return None

        # Create pattern string column for matching with UI selection
        df['pattern_str'] = df['pattern'].apply(
            lambda p: " -> ".join(map(str, p)) if isinstance(p, list) else None
        )

        # If selected_sequences is None (not set yet), show all patterns
        if selected_sequences is None:
            return df

        # Filter to user-selected patterns
        filtered = df[df['pattern_str'].isin(selected_sequences)]

        return filtered if not filtered.empty else None

    def _build_color_map(self, pattern_templates: List[str]) -> Dict[str, str]:
        """
        Build a color mapping for pattern templates.

        Parameters
        ----------
        pattern_templates : List[str]
            List of unique pattern template strings.

        Returns
        -------
        Dict[str, str]
            Mapping from pattern string to hex color.
        """
        colors = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
            '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]
        return {
            template: colors[i % len(colors)]
            for i, template in enumerate(pattern_templates)
        }

    def _add_sequence_trace(
        self,
        fig: go.Figure,
        instance_data: pd.DataFrame,
        color_map: Dict[str, str]
    ) -> None:
        """
        Add a single sequence instance trace to the figure.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to add the trace to.
        instance_data : pd.DataFrame
            Data for a single pattern instance (sorted by sequence axis).
        color_map : Dict[str, str]
            Mapping from pattern string to color.
        """
        pattern_str = instance_data['pattern_str'].iloc[0]
        instance_id = instance_data['pattern_instance_id'].iloc[0]
        support_count = instance_data['support_count'].iloc[0] if 'support_count' in instance_data.columns else 'N/A'
        group_id = instance_data['group_id'].iloc[0] if 'group_id' in instance_data.columns else 'N/A'

        # Build hover text for each point in the sequence
        hover_text = (
            f"<b>Pattern:</b> {pattern_str}<br>"
            f"<b>Support:</b> {support_count} cases<br>"
            f"<b>Group:</b> {group_id}<br>"
        )

        fig.add_trace(
            go.Scatter(
                x=instance_data[self.sequence_detection_axis],
                y=instance_data[self.grouping_key],
                mode='lines+markers',
                line=dict(color=color_map[pattern_str], width=2),
                marker=dict(size=8, color=color_map[pattern_str]),
                name=f"Instance {int(instance_id)}",
                legendgroup=pattern_str,
                showlegend=False,
                hovertemplate=hover_text + "<extra></extra>"
            )
        )

def find_subsequence_matches(sub: List[Any], seq: List[Any]) -> List[List[int]]:
    """
    Finds all occurrences of 'sub' in 'seq' and returns the list of lists 
    where each inner list contains the indices within 'seq' that form the match.
    (Non-overlapping search based on the last matched event's index)
    """
    matches = []
    start_index = 0
    while start_index < len(seq):
        try:
            # 1. Search for the first element of the sub-sequence
            current_match_start = seq.index(sub[0], start_index)
        except ValueError:
            break # First element not found from this point onward

        # 2. Check the rest of the sub-sequence
        is_match = True
        match_indices = [current_match_start]
        current_seq_search_index = current_match_start + 1

        for item in sub[1:]:
            try:
                # Find the next item strictly *after* the previous one
                current_seq_search_index = seq.index(item, current_seq_search_index)
                match_indices.append(current_seq_search_index)
                current_seq_search_index += 1
            except ValueError:
                is_match = False
                break

        if is_match:
            matches.append(match_indices)
            # Advance the main search beyond the last event of the current match
            start_index = match_indices[-1] + 1
        else:
            # If no match from current start_index, try starting from the next item
            start_index = current_match_start + 1 

    return matches
