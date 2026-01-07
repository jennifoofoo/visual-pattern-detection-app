"""
Case Arrival Trend Pattern for Dotted Charts.

Detects trends in the arrival of new cases over time.
Case arrival time = minimum actual_time per case_id.

This is different from event-frequency trends (TrendPattern):
- TrendPattern: counts ALL events over time
- CaseArrivalTrendPattern: counts only CASE STARTS (first event per case)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
from scipy import stats
from .pattern_base import Pattern


class CaseArrivalTrendPattern(Pattern):
    """
    Detects trends in case arrival times.

    Definition: Case arrival = min(actual_time) per case_id.
    Analyzes how the number of newly starting cases changes over time.

    Uses Mann-Kendall test for statistical significance
    and Sen's slope for trend magnitude estimation.
    """

    def __init__(
        self,
        view_config: Dict[str, str],
        aggregation_period: str = 'W',
        min_periods: int = 5,
        significance_level: float = 0.05
    ):
        super().__init__(name="Case Arrival Trend", view_config=view_config)
        self.aggregation_period = aggregation_period
        self.min_periods = min_periods
        self.significance_level = significance_level
        self.trend_result = None

    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Mann-Kendall trend test with Sen's slope estimator."""
        n = len(data)

        if n < self.min_periods:
            return {'direction': 'insufficient_data', 'p_value': 1.0, 'slope': 0.0, 'slope_percent': 0.0}

        # S statistic
        s = sum(np.sign(data[j] - data[i]) for i in range(n - 1) for j in range(i + 1, n))

        # Variance (with tie correction)
        unique, counts = np.unique(data, return_counts=True)
        tie_sum = sum(t * (t - 1) * (2 * t + 5) for t in counts if t > 1)
        var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18

        # Z score
        if var_s > 0:
            z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s) if s < 0 else 0
        else:
            z = 0

        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Sen's slope
        slopes = [(data[j] - data[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
        slope = np.median(slopes) if slopes else 0.0
        mean_val = np.mean(data)
        slope_percent = (slope / mean_val * 100) if mean_val > 0 else 0.0

        # Direction (require practical significance >= 0.5%)
        if p_value <= self.significance_level and abs(slope_percent) >= 0.5:
            direction = 'increasing' if z > 0 else 'decreasing'
        elif p_value <= self.significance_level:
            direction = 'stable'
        else:
            direction = 'no_trend'

        return {'direction': direction, 'p_value': p_value, 'slope': slope, 'slope_percent': slope_percent}

    def detect(self, df: pd.DataFrame) -> bool:
        """
        Detect case arrival trends.

        Steps:
        1. Extract min(actual_time) per case_id (case arrival time)
        2. Aggregate arrivals by time period (weekly)
        3. Apply Mann-Kendall test
        """
        if 'case_id' not in df.columns or 'actual_time' not in df.columns:
            self.detected = None
            return False

        try:
            # Case arrival = min(actual_time) per case_id
            case_arrivals = df.groupby('case_id')['actual_time'].min().reset_index()
            case_arrivals.columns = ['case_id', 'arrival_time']
            case_arrivals['arrival_time'] = pd.to_datetime(case_arrivals['arrival_time'])

            # Aggregate by time period
            arrival_counts = case_arrivals.set_index('arrival_time').resample(self.aggregation_period).size()

            if len(arrival_counts) < self.min_periods:
                self.detected = None
                return False

            # Mann-Kendall test
            self.trend_result = self._mann_kendall_test(arrival_counts.values)
            self.trend_result['total_cases'] = len(case_arrivals)
            self.trend_result['num_periods'] = len(arrival_counts)

            self.detected = self.trend_result
            return self.trend_result['direction'] != 'no_trend'

        except Exception:
            self.detected = None
            return False

    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """No visual overlay - results shown in tab only."""
        return fig

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of detected case arrival trend."""
        if self.detected is None or self.trend_result is None:
            return {'count': 0, 'details': None}

        direction = self.trend_result['direction']
        slope_pct = self.trend_result['slope_percent']
        slope_str = f"{slope_pct:+.1f}%" if abs(slope_pct) >= 0.1 else "~0%"

        return {
            'count': 1 if direction not in ['no_trend', 'insufficient_data'] else 0,
            'direction': direction,
            'slope_percent': slope_pct,
            'p_value': self.trend_result['p_value'],
            'total_cases': self.trend_result['total_cases'],
            'details': {'summary_text': f"Case arrivals: {direction} ({slope_str}/week)"}
        }
