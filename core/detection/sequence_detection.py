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
        # self.min_support = min_support
        mind_support_percentage = min_support/100
        # min_support_count = math.ceil(len(self.df_by_configuration[self.grouping_key].unique()) * mind_support_percentage)  
        unique_count = self.df_by_configuration[self.grouping_key].nunique()
        min_support_count = math.ceil(unique_count * mind_support_percentage)
        self.min_support = min_support_count # absolute count

        # results
        self.detected = False
        self.prefix_span = None
        self.df_found_patterns = pd.DataFrame()

        # TODO: at the end put found patterns back into the context of original DataFrame
        # this is also needed for visualization
        self.results = pd.DataFrame()

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

        # todo: add original df context for visualization
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
        # or use percentage?
        # total_sequences = len(sequence_data)
        # min_support_count = math.ceil(total_sequences * MIN_SUPPORT_PERCENTAGE)

        ps = PrefixSpan(sequence_data)
        frequent_patterns = ps.frequent(min_support_count)

        # save ps:
        self.prefix_span = ps

        return frequent_patterns

    def postprocess_prefixspan_results(self, found_subsequences: List[Tuple[int, List[Any]]]) -> pd.DataFrame:
        """
        Performs post-mining verification to map frequent patterns back to 
        specific event row indices in the original DataFrame for visualization.
        Returns:
            pd.DataFrame: A mapping table with 'index', 'group_id', and pattern details.
        """

        if not found_subsequences:
            return pd.DataFrame()

        # --- 1. Prepare Data for Verification ---
        df_with_index = self.df_sorted.reset_index()

        # Group the sorted data to get the sequence list and the original row indices (index of the DataFrame)
        grouped_data = df_with_index.groupby(self.grouping_key).agg(
            sequence_list=(self.event_key, list),
            original_indices=('index', list))

        total_groups = len(grouped_data)

        mapping_data = []
        pattern_instance_counter = 0

        # 2. Iterate through all found frequent patterns
        for support_count, pattern in found_subsequences:

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
        if self.df_found_patterns.empty:
            return {'total_patterns_found': 0, 'details': {'pattern_stats': {}}}

        # 1. Group by the pattern tuple to aggregate stats
        df_stats = self.df_found_patterns.copy()
        df_stats['pattern_tuple'] = df_stats['pattern'].apply(tuple)
        
        pattern_groups = df_stats.groupby('pattern_tuple').agg({
            'group_id': lambda x: sorted(list(set(x))),
            'support_count': 'first',
            'support_percentile': 'first',
            'pattern_instance_id': 'nunique' # Count how many times this specific pattern type appears
        })

        # 2. Build the pattern_stats dictionary for the UI
        pattern_stats = {}
        for pattern_tuple, row in pattern_groups.iterrows():
            pattern_str = " -> ".join(map(str, pattern_tuple))
            pattern_stats[pattern_str] = {
                'sequence': list(pattern_tuple),
                'group_ids': row['group_id'],
                'count': int(row['support_count']), # used for the label in the UI
                'occurrence_count': int(row['pattern_instance_id']),
                'support_percentile': float(row['support_percentile'])
            }

        return {
            'total_patterns_found': len(self.df_found_patterns['pattern_instance_id'].unique()),
            'max_pattern_length': self.df_found_patterns['pattern'].apply(len).max(),
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
    
    def visualize(self, df, fig, selected_patterns=None):
        pass
        """
        Overlays lines on the Plotly figure to connect events in detected sequences.
        """
        if not self.detected or self.results.empty:
            return fig

        # If no specific patterns are selected from the UI, 
        # we might want to show nothing or all. 
        # Based on your UI logic, let's assume we filter by selected_patterns.
        if not selected_patterns:
            return fig

        # 1. Filter results to only include the selected patterns
        # We create the same string representation used in your checkbox
        plot_df = self.results.copy()
        plot_df['pattern_str'] = plot_df.apply(
            lambda x: " -> ".join(map(str, x['pattern'])) if x['is_part_of_pattern'] else None, 
            axis=1
        )
        
        filtered_results = plot_df[plot_df['pattern_str'].isin(selected_patterns)]

        if filtered_results.empty:
            return fig

        # 2. Draw a line for each specific pattern instance (unique occurrence)
        # We group by pattern_instance_id to ensure we don't connect different users together
        instance_ids = filtered_results['pattern_instance_id'].unique()
        
        # Color mapping for pattern types (to keep the same color for same sequence templates)
        unique_templates = list(filtered_results['pattern_str'].unique())
        colors = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
            '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]
        color_map = {template: colors[i % len(colors)] for i, template in enumerate(unique_templates)}

        for instance_id in instance_ids:
            # Get events for this specific sequence instance
            instance_data = filtered_results[filtered_results['pattern_instance_id'] == instance_id]
            
            # Sort by the x-axis (sequence axis) to ensure lines connect chronologically
            instance_data = instance_data.sort_values(by=self.sequence_detection_axis)

            template_str = instance_data['pattern_str'].iloc[0]

            # Add the line trace
            fig.add_trace(
                go.Scatter(
                    x=instance_data[self.sequence_detection_axis],
                    y=instance_data[self.grouping_key],
                    mode='lines',
                    line=dict(color=color_map[template_str], width=2),
                    name=f"Instance {int(instance_id)}",
                    legendgroup=template_str, # Groups multiple instances of the same pattern together in legend
                    showlegend=False,         # Hide individual instance IDs to avoid legend clutter
                    hoverinfo='skip'          # Prevent line hover from blocking dot hover
                )
            )

        return fig

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
