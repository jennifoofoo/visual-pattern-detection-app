from .pattern_base import Pattern
import math
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from prefixspan import PrefixSpan
import os

# Keep your existing imports
from core.app_utils.mappings import X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP


class HorizontalSequencePatternDetector(Pattern):
    """Detector for horizontal sequence patterns in event data.
    
    Args:
        x_axis: Column key for the x-axis (sequence detection axis).
        y_axis: Column key for the y-axis (grouping key).
        dot_color: Column key for the dot color (event key).
        df: Full DataFrame containing the event log data.
        min_support: Minimum support percentage (0-100) for pattern detection.
    """
    
    def __init__(
        self,
        x_axis: str,
        y_axis: str,
        dot_color: str,
        df: pd.DataFrame,
        min_support: int = 50,
        is_strict: bool = False,
    ) -> None:
        # Initialize configuration
        self.x_axis_df_key = x_axis
        self.y_axis_df_key = y_axis
        self.dot_df_key = dot_color
        self.is_strict = is_strict
        
        # Store data
        self.full_df = df
        self.df_by_configuration = df[[self.x_axis_df_key, self.y_axis_df_key, self.dot_df_key]].copy()
        
        # Set detection parameters
        self.sequence_detection_axis = self.x_axis_df_key
        self.grouping_key = self.y_axis_df_key
        self.event_key = self.dot_df_key
        
        # Validate configuration
        self.warn_grouping_key_dot_config = self.grouping_key == self.dot_df_key
        if self.warn_grouping_key_dot_config:
            print(
                f"Warning: Grouping Key ('{self.grouping_key}') "
                f"and Dot Input ('{self.dot_df_key}') "
                f"are the same. Sequential pattern detection skipped."
            )
            return
        
        # Calculate minimum support
        unique_count = self.df_by_configuration[self.grouping_key].nunique()
        min_support_percentage = min_support / 100
        self.min_support = math.ceil(unique_count * min_support_percentage)
        
        # Initialize results
        self.topk = None
        self.detected = False
        self.df_sorted = None
        self.df_found_patterns = pd.DataFrame()
        self.results = pd.DataFrame()
        self.topkresults = pd.DataFrame()

    def detect(self) -> bool:
        """Main detection method."""
        if self.warn_grouping_key_dot_config:
            return False
            
        # 1. Prepare data for PrefixSpan
        # (Preprocessing is same for strict/non-strict: based on grouped timestamps)
        self.df_sorted, sequence_data = self._prepare_data_for_prefixspan()
        
        # 2. Extract subsequences using PrefixSpan
        # Note: PrefixSpan finds frequent loose subsequences. 
        # For strict mode, we still use this as a candidate generator, 
        # and strictness is enforced during mapping/validation.
        found_subsequences = self._extract_subsequences(sequence_data)
        
        # 3. Post-process results to map patterns to original indices
        pattern_map_df = self._postprocess_prefixspan_results(found_subsequences)
        
        # 4. Store results
        self.df_found_patterns = pattern_map_df
        if not pattern_map_df.empty:
            self.results = self._join_with_original_context(pattern_map_df)
            self.detected = True
        else:
            self.detected = False
            self.results = pd.DataFrame()
        
        return self.detected

    def _prepare_data_for_prefixspan(self) -> Tuple[pd.DataFrame, List[List[Any]]]:
        """
        Prepare sorted data and sequences for PrefixSpan.

        Important preprocessing step:
        - Events that share the same (grouping_key, timestamp) are treated as ONE item (an itemset).
          Example: if events A and B happen at the same timestamp, the sequence element becomes (A,B),
          enabling patterns like (A,B) -> C.
        """
        # Sort by (grouping key, timestamp) for stable itemset creation
        df_sorted = self.df_by_configuration.sort_values(
            by=[self.grouping_key, self.sequence_detection_axis],
            ascending=[True, True]
        )

        # Build per-(group, timestamp) itemsets (unique events at that timestamp)
        grouped_ts = df_sorted.groupby([self.grouping_key, self.sequence_detection_axis]).agg(
            events_at_ts=(self.event_key, lambda s: sorted(set(s.tolist())))
        )

        def _encode_itemset(events: List[Any]) -> Any:
            # PrefixSpan elements must be hashable/comparable; encode multi-event timestamps as tuples.
            # Singletons remain as the raw event value for readability.
            if len(events) == 1:
                return events[0]
            return tuple(events)

        grouped_ts["item"] = grouped_ts["events_at_ts"].apply(_encode_itemset)

        # Now build a sequence per grouping key, ordered by timestamp
        sequences_by_group = grouped_ts.reset_index().sort_values(
            by=[self.grouping_key, self.sequence_detection_axis],
            ascending=[True, True],
        ).groupby(self.grouping_key).agg(sequence=("item", list))

        sequence_data = sequences_by_group["sequence"].tolist()
        return df_sorted, sequence_data
    
    def _extract_subsequences(self, sequence_data: List[List[Any]]) -> List[Tuple[int, List[Any]]]:
        """Extract subsequences using PrefixSpan."""
        ps = PrefixSpan(sequence_data)
        frequent_patterns = ps.frequent(self.min_support)
        return frequent_patterns
    
    def _postprocess_prefixspan_results(
        self,
        found_subsequences: List[Tuple[int, List[Any]]],
        min_pattern_length: int = 2
    ) -> pd.DataFrame:
        """
        Map frequent patterns back to specific event row indices for visualization.
        Optimized to use inverted index for faster candidate filtering.
        """
        if not found_subsequences:
            return pd.DataFrame()
            
        # Filter patterns by minimum length
        filtered_subsequences = [
            (support, pattern) for support, pattern in found_subsequences
            if len(pattern) >= min_pattern_length
        ]
        
        if not filtered_subsequences:
            return pd.DataFrame()
            
        # Prepare data for verification/mapping
        df_with_index = self.df_sorted.reset_index()

        grouped_ts = df_with_index.groupby([self.grouping_key, self.sequence_detection_axis]).agg(
            events_at_ts=(self.event_key, lambda s: sorted(set(s.tolist()))),
            original_indices_at_ts=("index", lambda s: list(s.tolist())),
        )

        def _encode_itemset(events: List[Any]) -> Any:
            if len(events) == 1:
                return events[0]
            return tuple(events)

        grouped_ts["item"] = grouped_ts["events_at_ts"].apply(_encode_itemset)

        grouped_data = grouped_ts.reset_index().sort_values(
            by=[self.grouping_key, self.sequence_detection_axis],
            ascending=[True, True],
        ).groupby(self.grouping_key).agg(
            sequence_list=("item", list),
            original_indices_nested=("original_indices_at_ts", list),
        )
        
        # --- OPTIMIZATION START: Build Inverted Index ---
        # Map item -> set of group_ids that contain this item
        item_to_groups = defaultdict(set)
        for group_id, row in grouped_data.iterrows():
            # Use set for faster lookups and to avoid duplicates within same group
            unique_items = set(row['sequence_list'])
            for item in unique_items:
                item_to_groups[item].add(group_id)
        # --- OPTIMIZATION END ---
        
        total_groups = len(grouped_data)
        mapping_data = []
        pattern_instance_counter = 0
        
        # Map patterns to original indices
        for support_count, pattern in filtered_subsequences:
            relative_support = support_count / total_groups
            
            # --- OPTIMIZATION START: Filter Candidates ---
            # Only check groups that contain ALL items from the pattern
            if not pattern:
                candidate_groups = []
            else:
                # Start with groups containing the first item
                candidate_groups = item_to_groups[pattern[0]]
                # Intersect with groups for subsequent items
                for item in pattern[1:]:
                    candidate_groups = candidate_groups & item_to_groups[item]
                    if not candidate_groups:
                        break
            # --- OPTIMIZATION END ---
            
            # Only iterate over candidate groups
            for group_id in candidate_groups:
                # We can access the row directly if group_id is the index
                # grouped_data index is self.grouping_key, which is group_id
                try:
                    group_row = grouped_data.loc[group_id]
                except KeyError:
                    continue # Should not happen if index is built correctly, but safety first

                sequence = group_row['sequence_list']
                
                # Double check: match must observe order. 
                # The intersection only guarantees presence, not order.
                match_indices_list = self._find_subsequence_matches(pattern, sequence)
                
                if match_indices_list:
                    original_indices_nested = group_row["original_indices_nested"]
                    
                    for match_indices in match_indices_list:
                        pattern_instance_counter += 1
                        matched_df_indices: List[int] = []
                        for pos in match_indices:
                            matched_df_indices.extend(original_indices_nested[pos])
                        
                        for df_index in matched_df_indices:
                            mapping_data.append({
                                'index': df_index,
                                'group_id': group_id,
                                'pattern_instance_id': pattern_instance_counter,
                                'pattern': pattern,
                                'support_count': support_count,
                                'support_percentile': relative_support,
                                'is_part_of_pattern': True
                            })
        
        return pd.DataFrame(mapping_data).set_index('index')

    def _find_subsequence_matches(self, sub: List[Any], seq: List[Any]) -> List[List[int]]:
        """
        Find all non-overlapping occurrences of sub in seq.
        Returns list of lists with indices of matches.
        
        If self.is_strict is True, enforces elements to be strictly consecutive (gap=0).
        """
        matches = []
        if not sub:
            return []
            
        sub_len = len(sub)
        seq_len = len(seq)
        
        if self.is_strict:
            # STRICT MODE: Elements must be adjacent in the preprocessed sequence list
            # Note: The 'seq' here is a list of itemsets (events at same timestamp).
            
            # Re-implement strict with while loop for non-overlapping
            matches = []
            i = 0
            while i <= seq_len - sub_len:
                if seq[i : i + sub_len] == sub:
                    matches.append(list(range(i, i + sub_len)))
                    i += sub_len # Skip past this match
                else:
                    i += 1
            return matches

        else:
            # LOOSE MODE (Existing Logic)
            start_index = 0
            while start_index < len(seq):
                try:
                    current_match_start = seq.index(sub[0], start_index)
                except ValueError:
                    break
                    
                # Check the rest of the sub-sequence
                is_match = True
                match_indices = [current_match_start]
                current_seq_search_index = current_match_start + 1
                
                for item in sub[1:]:
                    try:
                        current_seq_search_index = seq.index(item, current_seq_search_index)
                        match_indices.append(current_seq_search_index)
                        current_seq_search_index += 1
                    except ValueError:
                        is_match = False
                        break
                
                if is_match:
                    matches.append(match_indices)
                    start_index = match_indices[-1] + 1
                else:
                    start_index = current_match_start + 1
                    
            return matches
    
    def _join_with_original_context(self, pattern_map_df: pd.DataFrame) -> pd.DataFrame:
        """Join pattern information with original dataframe."""
        if self.full_df.empty:
            return pd.DataFrame()
            
        if pattern_map_df.empty:
            results = self.full_df.copy()
            results['is_part_of_pattern'] = False
            results['pattern_instance_id'] = None
            results['pattern_display_name'] = "No Pattern"
            return results
            
        # Join patterns with original data
        results = self.full_df.join(pattern_map_df, how='left')
        results['is_part_of_pattern'] = results['is_part_of_pattern'].fillna(False)
        
        def _format_pattern_element(el: Any) -> str:
            # Multi-event timestamps are encoded as tuples, displayed as "(a,b)".
            if isinstance(el, tuple):
                return "(" + ",".join(map(str, el)) + ")"
            return str(el)

        # --- OPTIMIZATION START: Vectorized String Creation ---
        # 1. Create a unique mapping of pattern_instance_id -> pattern string
        # Use groupby().first() because 'pattern' column contains lists which are unhashable
        unique_patterns = pattern_map_df.groupby('pattern_instance_id')['pattern'].first().reset_index()
        
        pattern_strings = {}
        for _, row in unique_patterns.iterrows():
            pid = row['pattern_instance_id']
            pat = row['pattern']
            pat_str = f"Pattern {int(pid)}: {' -> '.join(_format_pattern_element(e) for e in pat)}"
            pattern_strings[pid] = pat_str
            
        # 2. Map the strings back to the results
        results['pattern_display_name'] = results['pattern_instance_id'].map(pattern_strings)
        results['pattern_display_name'] = results['pattern_display_name'].fillna("No Pattern")
        # --- OPTIMIZATION END ---
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get standardized pattern summary."""
        sequences_summary = self._get_sequences_summary()
        
        # Add standard fields
        sequences_summary['pattern_type'] = 'sequence'
        sequences_summary['detected'] = self.detected
        # 'count' is often used by UI as the main metric
        sequences_summary['count'] = sequences_summary.get('total_patterns_found', 0)
        
        return sequences_summary
    
    def _get_sequences_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for detected sequences.
        """
        source_df = self.topkresults if not self.topkresults.empty else self.df_found_patterns
        
        if source_df.empty:
            return {'total_patterns_found': 0, 'details': {'pattern_stats': {}}}
            
        df_stats = source_df.copy()
        df_stats['pattern_tuple'] = df_stats['pattern'].apply(tuple)
        
        pattern_groups = df_stats.groupby('pattern_tuple').agg({
            'group_id': lambda x: sorted(list(set(x))),
            'support_count': 'first',
            'support_percentile': 'first',
            'pattern_instance_id': 'nunique'
        })
        
        # Additional Metrics
        unique_patterns_count = len(pattern_groups)
        
        total_groups = self.df_by_configuration[self.grouping_key].nunique()
        groups_with_pattern = source_df['group_id'].nunique()
        group_coverage = groups_with_pattern / total_groups if total_groups > 0 else 0.0
        
        total_events = len(self.df_by_configuration)
        events_in_pattern = len(df_stats)
        event_coverage = events_in_pattern / total_events if total_events > 0 else 0.0
        
        supports = pattern_groups['support_count']
        avg_support = supports.mean()
        min_support_val = supports.min()
        max_support_val = supports.max()
        
        pattern_lengths = pattern_groups.index.map(len)
        length_dist = pattern_lengths.value_counts().sort_index().to_dict()
        
        # Element frequency
        # Use existing indices to fetch actual event values
        event_values = self.df_by_configuration.loc[source_df.index, self.event_key]
        element_counts = event_values.value_counts().head(10).to_dict()
        
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
            'unique_patterns_count': unique_patterns_count,
            'max_pattern_length': int(source_df['pattern'].apply(len).max()) if not source_df.empty else 0,
            'min_pattern_length': int(source_df['pattern'].apply(len).min()) if not source_df.empty else 0,
            'avg_pattern_length': float(source_df['pattern'].apply(len).mean()) if not source_df.empty else 0.0,
            'avg_support': float(avg_support) if not supports.empty else 0.0,
            'min_support_found': int(min_support_val) if not supports.empty else 0,
            'max_support_found': int(max_support_val) if not supports.empty else 0,
            'min_support_percentage': float(min_support_val / total_groups) if not supports.empty and total_groups > 0 else 0.0,
            'max_support_percentage': float(max_support_val / total_groups) if not supports.empty and total_groups > 0 else 0.0,
            'group_coverage': float(group_coverage),
            'event_coverage': float(event_coverage),
            'length_distribution': length_dist,
            'top_frequent_elements': element_counts,
            'top_k_sequences': self.topk,
            'min_support': self.min_support,
            'details': {
                'pattern_stats': pattern_stats
            }
        }
    
    def get_top_k_sequences(self, k: int) -> pd.DataFrame:
        """
        Returns the top k sequences found by support_count.
        """
        self.topk = k
        
        if self.df_found_patterns.empty:
            self.topkresults = pd.DataFrame()
            return self.topkresults
        
        # Get unique patterns with their support_count
        df_stats = self.df_found_patterns.copy()
        df_stats['pattern_tuple'] = df_stats['pattern'].apply(tuple)
        
        pattern_groups = df_stats.groupby('pattern_tuple').agg({
            'support_count': 'first'
        }).reset_index()
        
        # Sort and get top k
        pattern_groups_sorted = pattern_groups.sort_values(
            by='support_count', 
            ascending=False
        ).head(k)
        
        # Filter to top k patterns
        top_k_pattern_tuples = set(pattern_groups_sorted['pattern_tuple'].tolist())
        topk_df = df_stats[df_stats['pattern_tuple'].isin(top_k_pattern_tuples)].copy()
        topk_df = topk_df.drop(columns=['pattern_tuple'])
        
        self.topkresults = topk_df
        return self.topkresults
    
    def visualize(
        self,
        df: pd.DataFrame,
        fig: go.Figure,
        selected_seq_patterns: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Overlay lines on the Plotly figure to connect events in detected sequences.
        """
        filtered_df = self._get_selected_pattern_data(selected_seq_patterns)
        if filtered_df is None or filtered_df.empty:
            return fig
            
        color_map = self._build_color_map(filtered_df['pattern_str'].unique())
        
        for instance_id in filtered_df['pattern_instance_id'].unique():
            instance_data = filtered_df[filtered_df['pattern_instance_id'] == instance_id]
            instance_data = instance_data.sort_values(by=self.sequence_detection_axis)
            self._add_sequence_trace(fig, instance_data, color_map)
            
        return fig
    
    def _get_selected_pattern_data(
        self,
        selected_seq_patterns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Filter results to only include user-selected patterns."""
        if not self.detected or self.results.empty:
            return None
            
        if not selected_seq_patterns:
            return None
            
        df = self.results[self.results['is_part_of_pattern'] == True].copy()
        if df.empty:
            return None
            
        def _format_pattern_element(el: Any) -> str:
            if isinstance(el, tuple):
                return "(" + ",".join(map(str, el)) + ")"
            return str(el)

        df['pattern_str'] = df['pattern'].apply(
            lambda p: " -> ".join(_format_pattern_element(e) for e in p) if isinstance(p, list) else None
        )
        
        filtered = df[df['pattern_str'].isin(selected_seq_patterns)]
        return filtered if not filtered.empty else None
    
    def _build_color_map(self, pattern_templates: List[str]) -> Dict[str, str]:
        """Build color mapping for pattern templates."""
        colors = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
            '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]
        return {
            template: colors[hash(template) % len(colors)]
            for template in pattern_templates
        }
    
    def _add_sequence_trace(
        self,
        fig: go.Figure,
        instance_data: pd.DataFrame,
        color_map: Dict[str, str]
    ) -> None:
        """Add a single sequence instance trace to the figure."""
        pattern_str = instance_data['pattern_str'].iloc[0]
        instance_id = instance_data['pattern_instance_id'].iloc[0]
        support_count = instance_data['support_count'].iloc[0] if 'support_count' in instance_data.columns else 'N/A'
        group_id = instance_data['group_id'].iloc[0] if 'group_id' in instance_data.columns else 'N/A'
        
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
    
    @staticmethod
    def test_with_hospital_first2_traces(
        x_axis: str = 'actual_time',
        y_axis: str = 'case_id',
        dot_color: str = 'activity',
        min_support: int = 50
    ) -> None:
        """
        Test/debug function that loads the first 2 traces from Hospital_log.xes
        and prints output at each step for verification.
        
        Args:
            x_axis: Column key for the x-axis (sequence detection axis).
            y_axis: Column key for the y-axis (grouping key).
            dot_color: Column key for the dot color (event key).
            min_support: Minimum support percentage (0-100) for pattern detection.
        """
        from core.data_processing.loader import load_xes_log
        
        # Find Hospital_log.xes path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '../..')
        xes_path = os.path.join(project_root, 'data', 'Hospital_log.xes')
        
        if not os.path.exists(xes_path):
            print(f"Error: Hospital_log.xes not found at {xes_path}")
            return
        
        # Load data
        print("=" * 80)
        print("Loading Hospital_log.xes...")
        df_full = load_xes_log(xes_path)
        print(f"Loaded {len(df_full)} total events from {df_full['case_id'].nunique()} cases")
        
        # Get first 2 unique case IDs
        first_2_cases = df_full['case_id'].unique()[:2]
        df = df_full[df_full['case_id'].isin(first_2_cases)].copy()
        print(f"\nFiltered to first 2 cases: {list(first_2_cases)}")
        print(f"Filtered DataFrame: {len(df)} events")
        
        # Initialize detector
        print("\n" + "=" * 80)
        print("STEP 1: Initializing detector...")
        detector = HorizontalSequencePatternDetector(
            x_axis=x_axis,
            y_axis=y_axis,
            dot_color=dot_color,
            df=df,
            min_support=min_support
        )
        print(f"Min support: {detector.min_support}")
        print(f"Unique {y_axis} count: {detector.df_by_configuration[y_axis].nunique()}")
        
        # Step 2: Prepare data for PrefixSpan
        print("\n" + "=" * 80)
        print("STEP 2: Preparing data for PrefixSpan (preprocessing)...")
        df_sorted, sequence_data = detector._prepare_data_for_prefixspan()
        detector.df_sorted = df_sorted  # Set instance variable for postprocessing
        print(f"\nSorted DataFrame shape: {df_sorted.shape}")
        print(f"\nSequence data (one sequence per {y_axis}):")
        for i, seq in enumerate(sequence_data):
            group_id = df_sorted[y_axis].unique()[i] if i < len(df_sorted[y_axis].unique()) else f"Group_{i}"
            print(f"  {y_axis}={group_id}: {seq}")
            print(f"    (Length: {len(seq)} items)")
        
        # Step 3: Extract subsequences
        print("\n" + "=" * 80)
        print("STEP 3: Extracting subsequences with PrefixSpan...")
        found_subsequences = detector._extract_subsequences(sequence_data)
        print(f"\nFound {len(found_subsequences)} frequent patterns (support >= {detector.min_support}):")
        for support, pattern in found_subsequences[:10]:  # Show first 10
            pattern_str = " -> ".join(
                f"({','.join(map(str, p))})" if isinstance(p, tuple) else str(p)
                for p in pattern
            )
            print(f"  Support={support}: {pattern_str}")
        
        # Step 4: Post-process results
        print("\n" + "=" * 80)
        print("STEP 4: Post-processing results (mapping patterns to original indices)...")
        pattern_map_df = detector._postprocess_prefixspan_results(found_subsequences)
        print(f"\nPattern mapping DataFrame shape: {pattern_map_df.shape}")
        if not pattern_map_df.empty:
            print(f"Columns: {list(pattern_map_df.columns)}")
            print(f"\nFirst few pattern mappings:")
            print(pattern_map_df.head(10))
        
        # Step 5: Join with original context
        print("\n" + "=" * 80)
        print("STEP 5: Joining with original context...")
        results = detector._join_with_original_context(pattern_map_df)
        print(f"\nResults DataFrame shape: {results.shape}")
        print(f"Columns: {list(results.columns)}")
        if 'is_part_of_pattern' in results.columns:
            pattern_count = results['is_part_of_pattern'].sum()
            print(f"\nEvents part of patterns: {pattern_count} / {len(results)}")
            if pattern_count > 0:
                print(f"\nSample pattern rows:")
                pattern_rows = results[results['is_part_of_pattern'] == True].head(10)
                for col in [x_axis, y_axis, dot_color, 'pattern_display_name']:
                    if col in pattern_rows.columns:
                        print(f"  {col}: {pattern_rows[col].tolist()}")
        
        # Step 6: Run full detect()
        print("\n" + "=" * 80)
        print("STEP 6: Running full detect() method...")
        detector.detect()
        print(f"Detection completed: {detector.detected}")
        if detector.detected:
            print(f"Results shape: {detector.results.shape}")
            if not detector.df_found_patterns.empty:
                print(f"Found patterns DataFrame shape: {detector.df_found_patterns.shape}")
        
        print("\n" + "=" * 80)
        print("Test completed!")
        print("=" * 80)