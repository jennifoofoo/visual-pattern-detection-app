import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from typing import List, Dict, Any, Optional
from collections import defaultdict # needed for pattern counting

from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP


class ChartConfig:
    """Stub for chart configuration, e.g., x_axis, y_axis, dot"""
    def __init__(self, x_axis_label: str, y_axis_label: str, dot_label: str, df_full: pd.DataFrame):
        self.x_axis_label = x_axis_label
        self.x_axis_df_name = X_AXIS_COLUMN_MAP[x_axis_label]

        self.y_axis_label = y_axis_label
        self.y_axis_df_name = Y_AXIS_COLUMN_MAP[y_axis_label]
        
        self.dot_label = dot_label
        self.dot_df_name = DOTS_COLOR_MAP[dot_label]

        # Grouping key is always the Y-axis column for sequence detection
        self.grouping_key_df_name = self.y_axis_df_name
        self.grouping_key_label = self.y_axis_label
        
        # Keep the relevant columns for detection and sorting
        self.df = df_full[[self.x_axis_df_name, self.y_axis_df_name, self.dot_df_name]].copy()


class SequencePatternDetector:
    def __init__(
        self,
        config: ChartConfig,
        min_support: float,
        min_length: int,
        max_length: int,
        min_occurrences: int
    ):
        self.config = config

        self.min_support = min_support
        self.min_length = min_length
        self.max_length = max_length
        self.min_occurrences = min_occurrences

        self.df_patterns_summary = pd.DataFrame()
        
    def _extract_all_subsequences(self, series: pd.Series) -> List[List[Any]]:
        """
        Extracts all contiguous subsequences from a sequence of events (series) 
        within the min/max length constraints.
        """
        sequence = series.tolist()
        subsequences = []
        n = len(sequence)
        
        for i in range(n):
            # The inner loop determines the end of the subsequence
            for j in range(i + self.min_length, min(i + self.max_length + 1, n + 1)):
                subsequences.append(sequence[i:j])
        return subsequences
    
    def detect(self) -> pd.DataFrame:
        """
        Detects frequent sequential patterns in the process data.
        Returns a DataFrame summarizing the detected patterns, or an empty DataFrame 
        if constraints are not met or configuration is invalid.
        """
        
        # 1. Configuration Check and Empty Data Check
        if self.config.grouping_key_df_name == self.config.dot_df_name:
            print(
                f"Warning: Grouping Key ('{self.config.y_axis_label}') "
                f"and Sequence Element ('{self.config.dot_label}') "
                f"are the same. Sequential pattern detection skipped."
            )
            return pd.DataFrame()

        if self.config.df.empty:
             print("Input DataFrame is empty.")
             return pd.DataFrame()

        # 2. Group and create ordered sequences
        df_sorted = self.config.df.sort_values(
            by=[self.config.grouping_key_df_name, self.config.x_axis_df_name], 
            ascending=[True, True]
        )
        
        sequences_by_group = df_sorted.groupby(self.config.grouping_key_df_name).agg(
            sequence=(self.config.dot_df_name, list)
        )
        
        total_traces = len(sequences_by_group) 
        if total_traces == 0:
            print("No traces found for sequence detection.")
            return pd.DataFrame()

        # 3. Generate all contiguous subsequences 
        all_subsequences_data = [] # Stores {'subsequence': tuple, 'group_id': str}
        
        for group_id, row in sequences_by_group.iterrows():
            sequence = pd.Series(row['sequence'])
            
            subsequences = self._extract_all_subsequences(sequence)
            
            # Store the patterns and the group ID they came from
            for sub in subsequences:
                all_subsequences_data.append({
                    'subsequence': tuple(sub), 
                    'group_id': group_id
                })
        
        if not all_subsequences_data:
            print("No subsequences found within min/max length constraints.")
            return pd.DataFrame()

        # 4. Count and Filter
        
        # Aggregate the data to get counts and unique groups
        pattern_data = defaultdict(lambda: {'count': 0, 'group_ids': set()})
        
        for item in all_subsequences_data:
            seq_tuple = item['subsequence']
            group_id = item['group_id']
            
            pattern_data[seq_tuple]['count'] += 1 
            pattern_data[seq_tuple]['group_ids'].add(group_id)

        frequent_patterns_list = []
        grouping_key_name = self.config.y_axis_label
        grouping_key_col = self.config.grouping_key_df_name

        for seq, data in pattern_data.items():
            support_groups = len(data['group_ids'])
            support_ratio = support_groups / total_traces
            
            # Filter based on user-defined constraints
            if (data['count'] >= self.min_occurrences and support_ratio >= self.min_support):
                
                frequent_patterns_list.append({
                    'sequence': seq,
                    'length': len(seq),
                    'count': data['count'],
                    f'support_{grouping_key_name}s': support_groups,
                    'support_ratio': round(support_ratio, 4),
                    grouping_key_col: list(data['group_ids']),
                    'sequence_element': self.config.dot_label
                })

        frequent_patterns_df = pd.DataFrame(frequent_patterns_list)
        
        # 5. Final Formatting and Return
        if frequent_patterns_df.empty:
            print("No frequent sequences found based on the provided constraints.")
            self.df_patterns_summary = pd.DataFrame()
            return pd.DataFrame()

        frequent_patterns_df.sort_values(
            by=['count', 'support_ratio'], 
            ascending=[False, False], 
            inplace=True
        )
        frequent_patterns_df['sequence'] = frequent_patterns_df['sequence'].apply(
            lambda x: ' -> '.join(map(str, x))
        )
        
        self.df_patterns_summary = frequent_patterns_df

        return frequent_patterns_df


    def get_detected_sequence_list(self) -> List[str]:
        """
        Returns a list of all detected frequent sequence strings for UI filtering.
        """
        if self.df_patterns_summary.empty:
            return []
            
        return self.df_patterns_summary['sequence'].tolist()


    def _prepare_df_for_pattern_plot(
        self, 
        df_full: pd.DataFrame,
        df_summary_filtered: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Helper to add 'is_frequent_sequence' status column to the full DataFrame 
        based on the provided *filtered* pattern summary.
        """
        if df_summary_filtered.empty:
            df_full['is_frequent_sequence'] = 'No Pattern Found'
            df_full['sequence_name'] = 'N/A'
            return df_full

        grouping_key_col = self.config.grouping_key_df_name
        all_pattern_groups = set()
        
        for group_list in df_summary_filtered[grouping_key_col]:
            all_pattern_groups.update(group_list)

        df_full['is_frequent_sequence'] = df_full[grouping_key_col].isin(all_pattern_groups).map(
            {True: 'Sequence Detected', False: 'Other Events'}
        )
        df_full['sequence_name'] = df_full['is_frequent_sequence']
        
        return df_full
    
    def apply_pattern_highlight(
        self,
        df_full: pd.DataFrame,
        base_fig: go.Figure,
        patterns_to_highlight: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Public method to apply the pattern highlighting overlay to a base figure.
        Ensures detection runs if needed before applying the visual overlay.
        """
        # Ensure detection has run before visualizing
        if self.df_patterns_summary.empty:
            # We assume detect() is ready to run and will populate self.df_patterns_summary
            self.detect() 
            
        # Call the core visualization logic
        return self.visualize(
            df=df_full, 
            fig=base_fig, 
            patterns_to_highlight=patterns_to_highlight
        )

    def visualize(
        self, 
        df: pd.DataFrame, 
        fig: go.Figure,
        patterns_to_highlight: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Overlays the detected frequent sequences onto an existing Plotly figure, 
        managing trace visibility to ensure only selected highlights are shown.
        """
        
        # 1. TRACE CLEANUP (Crucial for fixing the update issue)
        # Remove any existing highlight trace before processing the new request.
        try:
            fig.data = tuple(
                trace for trace in fig.data 
                if getattr(trace, 'name', None) != 'Detected Sequence'
            )
        except Exception as e:
            print(f"Warning: Trace cleanup failed: {e}")
            
        # 2. Filter the patterns summary
        df_summary = self.df_patterns_summary
        
        if df_summary.empty:
            fig.update_layout(title_text=fig.layout.title.text.split(" with Sequences")[0] + " (No Sequences Found)")
            return fig
            
        if patterns_to_highlight:
            df_patterns_filtered = df_summary[df_summary['sequence'].isin(patterns_to_highlight)].copy()
        else:
            df_patterns_filtered = df_summary.copy()

        # 3. Handle Empty Filter Result (The Reset Case)
        if df_patterns_filtered.empty:
            # Resetting marker styling to default (undo the fade)
            fig.update_traces(
                marker=dict(opacity=0.8, size=5),
                selector=dict(mode='markers')
            )
            # Resetting the figure title
            title_text = fig.layout.title.text.split(" with Sequences")[0]
            fig.update_layout(title_text=title_text)
            
            return fig
        
        # 4. Prepare Data for Highlighting (Runs only if patterns are selected)
        df_plot_highlighted = self._prepare_df_for_pattern_plot(
            df_full=df.copy(), 
            df_summary_filtered=df_patterns_filtered
        )

        # 5. Extract Pattern Events and Column Names
        df_pattern = df_plot_highlighted[df_plot_highlighted['is_frequent_sequence'] == 'Sequence Detected']
        
        # ... (Column extraction remains here) ...
        x_col = self.config.x_axis_df_name
        y_col = self.config.y_axis_df_name
        color_col = self.config.dot_df_name

        # 6. Add the Highlighted Trace (Overlay)
        
        # Fade the original points (background)
        fig.update_traces(marker=dict(opacity=0.3, size=5), selector=dict(mode='markers'))
        
        highlight_fig = px.scatter(
            df_pattern,
            x=x_col, y=y_col, color=color_col,
            # ... (labels and hover_data) ...
        )

        highlight_trace = highlight_fig.update_traces(
            name='Detected Sequence', 
            marker=dict(size=10, line=dict(width=2, color='DarkRed'), opacity=1.0),
            showlegend=False
        )['data']
        
        fig.add_traces(highlight_trace)

        # 7. Update Figure Title
        title_suffix = " (Filtered)" if patterns_to_highlight and len(patterns_to_highlight) < len(self.get_detected_sequence_list()) else " (All Frequent)"
        current_title_base = fig.layout.title.text.split(" with Sequences")[0]
        new_title = f"{current_title_base} with Sequences{title_suffix}:"
        fig.update_layout(title_text=new_title, title_font_size=18)
        
        return fig
    