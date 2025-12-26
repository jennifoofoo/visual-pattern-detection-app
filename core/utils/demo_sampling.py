"""
Demo-Mode Sampling for Process-Aware Pattern Detection.

Provides trace-preserving, variant-aware sampling for large event logs to enable fast demonstration.

Sampling Strategies:
- FULL: No sampling, use complete dataset
- MINIMAL: Aggressive sampling for quick demos (1 trace per variant)
- SQRT: Balanced sampling using sqrt(n) for frequent variants (default for demo)
- OPTIMIZED: Gentle reduction preserving data characteristics (~70% of original)
- LEGACY: Original simple first-N sampling (backward compatible)
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


class SamplingMode(Enum):
    """Available sampling strategies for event logs."""
    FULL = "full"           # No sampling - use complete dataset
    MINIMAL = "minimal"     # Aggressive: 1 trace per variant, max 50 variants
    SQRT = "sqrt"           # Balanced: sqrt(n) traces per frequent variant
    OPTIMIZED = "optimized"  # Gentle: ~70% of data, preserving distribution
    LEGACY = "legacy"       # Original simple first-N sampling


@dataclass
class SamplingConfig:
    """Configuration for a sampling strategy."""
    mode: SamplingMode
    name: str
    description: str
    # Parameters for each mode
    min_traces_per_variant: int = 1
    max_traces_per_variant: Optional[int] = None
    rare_variant_threshold: int = 5  # Keep all traces if variant has <= this many
    target_reduction_ratio: float = 0.7  # For OPTIMIZED mode
    max_total_traces: Optional[int] = None  # Overall cap on traces
    # Overall cap on events (for MINIMAL)
    max_total_events: Optional[int] = None
    preserve_weights: bool = False  # Whether to add weight column


# Pre-defined sampling configurations
SAMPLING_CONFIGS = {
    SamplingMode.FULL: SamplingConfig(
        mode=SamplingMode.FULL,
        name="Full Dataset",
        description="No sampling - analyze complete event log"
    ),
    SamplingMode.MINIMAL: SamplingConfig(
        mode=SamplingMode.MINIMAL,
        name="Minimal Sample",
        description="Ultra-fast: 1-2 traces per variant, max 50 variants, capped at 5K events",
        min_traces_per_variant=1,
        max_traces_per_variant=2,
        rare_variant_threshold=2,
        max_total_traces=100,
        max_total_events=5000  # Hard cap on events for truly minimal sample
    ),
    SamplingMode.SQRT: SamplingConfig(
        mode=SamplingMode.SQRT,
        name="Balanced Sample (√n)",
        description="Rare variants: keep all, Frequent: √n traces per variant",
        min_traces_per_variant=1,
        max_traces_per_variant=None,  # sqrt determines this
        rare_variant_threshold=5,
        max_total_traces=500,
        max_total_events=20000  # Cap at 20K events
    ),
    SamplingMode.OPTIMIZED: SamplingConfig(
        mode=SamplingMode.OPTIMIZED,
        name="Optimized Sample",
        description="Gentle reduction (~70%) preserving variant distribution",
        min_traces_per_variant=1,
        max_traces_per_variant=None,
        rare_variant_threshold=10,
        target_reduction_ratio=0.7,
        max_total_events=50000,  # Cap at 50K events
        preserve_weights=True
    ),
    SamplingMode.LEGACY: SamplingConfig(
        mode=SamplingMode.LEGACY,
        name="Legacy Sample",
        description="Original first-N cases sampling (backward compatible)",
        max_total_traces=100
    ),
}


class VariantAwareSampler:
    """
    Trace-preserving, variant-aware sampler for event logs.

    A variant is a unique sequence of activities within a trace. This sampler:
    1. Groups traces by their variant (activity sequence)
    2. Applies variant-aware sampling based on the configured strategy
    3. Samples traces uniformly within each variant group
    4. Preserves complete traces (never samples events within a trace)

    This ensures pattern detection sees representative examples from all
    process variants while reducing data volume for faster analysis.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the sampler.

        Parameters
        ----------
        random_state : int
            Random seed for reproducible sampling
        """
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def _compute_variants(
        self,
        df: pd.DataFrame,
        case_col: str = 'case_id',
        activity_col: str = 'activity',
        time_col: str = 'actual_time'
    ) -> pd.DataFrame:
        """
        Compute the variant (activity sequence) for each trace.

        Returns a DataFrame with case_id and variant columns.
        """
        # Sort by case and time to get correct activity sequence
        df_sorted = df.sort_values([case_col, time_col])

        # Build variant string for each case (tuple of activities)
        variants = (
            df_sorted.groupby(case_col, sort=False)[activity_col]
            .apply(lambda x: tuple(x.values))
            .reset_index()
        )
        variants.columns = [case_col, 'variant']

        return variants

    def _get_variant_stats(self, variants_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for each variant.

        Returns DataFrame with columns: variant, trace_count, is_rare
        """
        stats = (
            variants_df.groupby('variant', sort=False)
            .size()
            .reset_index(name='trace_count')
        )
        stats = stats.sort_values(
            'trace_count', ascending=False).reset_index(drop=True)
        return stats

    def _compute_sample_sizes(
        self,
        variant_stats: pd.DataFrame,
        config: SamplingConfig
    ) -> pd.DataFrame:
        """
        Compute how many traces to sample from each variant.
        """
        variant_stats = variant_stats.copy()

        if config.mode == SamplingMode.FULL:
            variant_stats['sample_size'] = variant_stats['trace_count']

        elif config.mode == SamplingMode.MINIMAL:
            # Keep 1-2 traces per variant
            variant_stats['sample_size'] = variant_stats['trace_count'].apply(
                lambda n: min(n, config.max_traces_per_variant or 2)
            )

        elif config.mode == SamplingMode.SQRT:
            # Rare variants: keep all; Frequent: keep sqrt(n)
            def sqrt_sample(row):
                n = row['trace_count']
                if n <= config.rare_variant_threshold:
                    return n  # Keep all for rare variants
                else:
                    # sqrt(n) for frequent, but at least min_traces
                    sample = max(int(np.sqrt(n)),
                                 config.min_traces_per_variant)
                    return min(sample, n)

            variant_stats['sample_size'] = variant_stats.apply(
                sqrt_sample, axis=1)

        elif config.mode == SamplingMode.OPTIMIZED:
            # Proportional reduction targeting ~70% total reduction
            total_traces = variant_stats['trace_count'].sum()
            target_total = int(total_traces * config.target_reduction_ratio)

            def optimized_sample(row):
                n = row['trace_count']
                if n <= config.rare_variant_threshold:
                    return n  # Keep all for rare variants
                else:
                    # Proportional reduction
                    ratio = target_total / total_traces
                    sample = max(int(n * ratio), config.min_traces_per_variant)
                    return min(sample, n)

            variant_stats['sample_size'] = variant_stats.apply(
                optimized_sample, axis=1)

        elif config.mode == SamplingMode.LEGACY:
            # Will be handled differently - just mark all for potential inclusion
            variant_stats['sample_size'] = variant_stats['trace_count']

        # Apply max_total_traces cap if specified (except FULL mode)
        if config.max_total_traces and config.mode != SamplingMode.FULL:
            current_total = variant_stats['sample_size'].sum()
            if current_total > config.max_total_traces:
                # Scale down proportionally
                scale = config.max_total_traces / current_total
                variant_stats['sample_size'] = (
                    variant_stats['sample_size'] * scale
                ).apply(lambda x: max(1, int(x)))

        # Compute weight (for potential weighted analysis)
        variant_stats['weight'] = (
            variant_stats['trace_count'] / variant_stats['sample_size']
        )

        return variant_stats

    def _cap_by_events(
        self,
        df: pd.DataFrame,
        max_events: int,
        case_col: str,
        time_col: str
    ) -> pd.DataFrame:
        """
        Further reduce sampled data to meet max_events cap.

        Removes traces (preferring shorter ones to keep diversity) until
        total events is under the cap.
        """
        if len(df) <= max_events:
            return df

        # Compute events per case
        events_per_case = df.groupby(
            case_col).size().reset_index(name='event_count')

        # Sort by event count (keep shorter traces for more diversity)
        events_per_case = events_per_case.sort_values('event_count')

        # Greedily select traces until we hit the cap
        selected_cases = []
        total_events = 0

        for _, row in events_per_case.iterrows():
            if total_events + row['event_count'] <= max_events:
                selected_cases.append(row[case_col])
                total_events += row['event_count']
            elif not selected_cases:
                # Always include at least one trace even if it exceeds cap
                selected_cases.append(row[case_col])
                total_events += row['event_count']
                break

        return df[df[case_col].isin(selected_cases)].copy()

    def sample(
        self,
        df: pd.DataFrame,
        mode: SamplingMode = SamplingMode.SQRT,
        config: Optional[SamplingConfig] = None,
        case_col: str = 'case_id',
        activity_col: str = 'activity',
        time_col: str = 'actual_time'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Sample the event log using variant-aware strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Full event log with case_id, activity, and time columns
        mode : SamplingMode
            Sampling strategy to use
        config : SamplingConfig, optional
            Custom configuration (overrides mode default)
        case_col, activity_col, time_col : str
            Column names for case ID, activity, and timestamp

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            - Sampled event log (preserving complete traces)
            - Sampling statistics dictionary
        """
        if df.empty or case_col not in df.columns:
            return df.copy(), {"error": "Empty or missing case_id column"}

        # Get configuration
        if config is None:
            config = SAMPLING_CONFIGS[mode]

        # LEGACY mode: use original simple sampling
        if config.mode == SamplingMode.LEGACY:
            return self._legacy_sample(df, config, case_col, time_col)

        # FULL mode: return as-is
        if config.mode == SamplingMode.FULL:
            stats = {
                "mode": config.mode.value,
                "original_traces": df[case_col].nunique(),
                "sampled_traces": df[case_col].nunique(),
                "original_events": len(df),
                "sampled_events": len(df),
                "reduction_ratio": 1.0,
                "variants_total": None,  # Computed if needed
            }
            return df.copy(), stats

        # Compute variants
        variants_df = self._compute_variants(
            df, case_col, activity_col, time_col)
        variant_stats = self._get_variant_stats(variants_df)

        # Compute sample sizes per variant
        sample_plan = self._compute_sample_sizes(variant_stats, config)

        # Merge sample sizes back to cases
        cases_with_samples = variants_df.merge(
            sample_plan[['variant', 'sample_size', 'weight']],
            on='variant'
        )

        # Sample traces within each variant
        sampled_cases = []
        for variant, group in cases_with_samples.groupby('variant', sort=False):
            sample_size = group['sample_size'].iloc[0]
            weight = group['weight'].iloc[0]

            if len(group) <= sample_size:
                selected = group[case_col].tolist()
            else:
                selected = self._rng.choice(
                    group[case_col].values,
                    size=int(sample_size),
                    replace=False
                ).tolist()

            for case in selected:
                sampled_cases.append({
                    case_col: case,
                    '_sample_weight': weight if config.preserve_weights else 1.0
                })

        sampled_case_df = pd.DataFrame(sampled_cases)

        # Filter original dataframe to sampled cases
        df_sampled = df[df[case_col].isin(sampled_case_df[case_col])].copy()

        # Enforce max_total_events cap if specified
        # This is important when traces are long (many events per trace)
        if config.max_total_events and len(df_sampled) > config.max_total_events:
            df_sampled = self._cap_by_events(
                df_sampled, config.max_total_events, case_col, time_col
            )
            # Update sampled_case_df to match
            sampled_case_df = sampled_case_df[
                sampled_case_df[case_col].isin(df_sampled[case_col].unique())
            ]

        # Add weights if configured
        if config.preserve_weights:
            weight_map = sampled_case_df.set_index(
                case_col)['_sample_weight'].to_dict()
            df_sampled['_sample_weight'] = df_sampled[case_col].map(weight_map)

        # Compute statistics
        stats = {
            "mode": config.mode.value,
            "original_traces": df[case_col].nunique(),
            "sampled_traces": df_sampled[case_col].nunique(),
            "original_events": len(df),
            "sampled_events": len(df_sampled),
            "reduction_ratio": len(df_sampled) / len(df) if len(df) > 0 else 1.0,
            "variants_total": len(variant_stats),
            "variants_sampled": len(sample_plan[sample_plan['sample_size'] > 0]),
            "rare_variants_preserved": len(
                sample_plan[sample_plan['trace_count']
                            <= config.rare_variant_threshold]
            ),
        }

        return df_sampled, stats

    def _legacy_sample(
        self,
        df: pd.DataFrame,
        config: SamplingConfig,
        case_col: str,
        time_col: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Original simple first-N sampling for backward compatibility.
        """
        max_cases = config.max_total_traces or 100

        all_cases = df[case_col].unique()
        selected_cases = all_cases[:max_cases]
        df_sampled = df[df[case_col].isin(selected_cases)].copy()

        stats = {
            "mode": "legacy",
            "original_traces": len(all_cases),
            "sampled_traces": len(selected_cases),
            "original_events": len(df),
            "sampled_events": len(df_sampled),
            "reduction_ratio": len(df_sampled) / len(df) if len(df) > 0 else 1.0,
        }

        return df_sampled, stats


# =============================================================================
# Convenience Functions (for easy integration)
# =============================================================================

def sample_eventlog_variant_aware(
    df: pd.DataFrame,
    mode: SamplingMode = SamplingMode.SQRT,
    case_col: str = 'case_id',
    activity_col: str = 'activity',
    time_col: str = 'actual_time',
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Sample an event log using variant-aware strategy.

    This is the main entry point for variant-aware sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Full event log
    mode : SamplingMode
        Sampling strategy (FULL, MINIMAL, SQRT, OPTIMIZED, LEGACY)
    case_col, activity_col, time_col : str
        Column names
    random_state : int
        Random seed

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Sampled DataFrame and statistics

    Examples
    --------
    >>> df_sampled, stats = sample_eventlog_variant_aware(df, SamplingMode.SQRT)
    >>> print(f"Reduced from {stats['original_events']} to {stats['sampled_events']} events")
    """
    sampler = VariantAwareSampler(random_state=random_state)
    return sampler.sample(
        df, mode=mode,
        case_col=case_col,
        activity_col=activity_col,
        time_col=time_col
    )


def get_sampling_mode_options() -> Dict[str, SamplingConfig]:
    """
    Get all available sampling modes with their configurations.

    Returns a dict suitable for UI selection.
    """
    return {config.name: config for config in SAMPLING_CONFIGS.values()}


# =============================================================================
# Legacy Function (backward compatible)
# =============================================================================

def sample_small_eventlog(
    df: pd.DataFrame,
    max_cases: int = 100,
    max_events_per_case: int = 30,
    time_col: str = 'actual_time',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample a smaller event log while preserving process structure.

    DEPRECATED: Use sample_eventlog_variant_aware() for variant-aware sampling.
    This function is kept for backward compatibility.

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
