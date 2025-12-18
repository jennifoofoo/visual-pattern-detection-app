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
        x_axis = st.session_state.current_view['x_axis']
        y_axis = st.session_state.current_view['y_axis']
        dot_input = st.session_state.current_view['dot']
        df_full = st.session_state.get('df_full', None)

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
        self.prefix_span = None
        self.df_found_patterns = pd.DataFrame()

        # TODO: at the end put found patterns back into the context of original DataFrame
        # this is also needed for visualization
        self.results = pd.DataFrame()

    def detect_sequence(self):
        if self.warn_grouping_key_dot_config:
            return {}
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
        return pattern_map_df

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