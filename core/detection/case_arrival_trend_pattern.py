"""
Case Arrival Trend Pattern for Dotted Charts.

Detects trends in the arrival of new cases over time.
Case arrival time = minimum actual_time per case_id.

This is different from event-frequency trends (TrendPattern):
- TrendPattern: counts ALL events over time
- CaseArrivalTrendPattern: counts only CASE STARTS (first event per case)

Enhanced with optional Prophet integration for:
- Weekly seasonality detection (weekday vs weekend patterns)
- Changepoint detection (sudden process changes)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
from scipy import stats
from .pattern_base import Pattern

# Optional Prophet import
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class CaseArrivalTrendPattern(Pattern):
    """
    Detects trends in case arrival times.

    Definition: Case arrival = min(actual_time) per case_id.
    Analyzes how the number of newly starting cases changes over time.

    Primary method: Mann-Kendall test for statistical significance
    Optional enhancement: Prophet for seasonality and changepoint detection
    """

    def __init__(
        self,
        view_config: Dict[str, str],
        aggregation_period: str = 'W',
        min_periods: int = 5,
        significance_level: float = 0.05,
        use_prophet: bool = True  # Enable Prophet if available
    ):
        super().__init__(name="Case Arrival Trend", view_config=view_config)
        self.aggregation_period = aggregation_period
        self.min_periods = min_periods
        self.significance_level = significance_level
        self.use_prophet = use_prophet and PROPHET_AVAILABLE
        self.trend_result = None
        self.prophet_insights = None  # Store Prophet results separately

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

    def _prophet_analysis(self, arrival_counts: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Run Prophet analysis for seasonality and changepoint detection.

        Parameters
        ----------
        arrival_counts : pd.Series
            Time series of case arrival counts (index=date, values=count)

        Returns
        -------
        dict or None
            Prophet insights including seasonality and changepoints
        """
        if not PROPHET_AVAILABLE:
            return None

        # Need at least 2 weeks of data for meaningful Prophet analysis
        if len(arrival_counts) < 14:
            return None

        try:
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            df_prophet = pd.DataFrame({
                'ds': arrival_counts.index,
                'y': arrival_counts.values
            })

            # Suppress Prophet's verbose output
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

            # Initialize Prophet with minimal settings
            model = Prophet(
                yearly_seasonality=False,  # Usually not enough data
                weekly_seasonality=True,   # Detect weekday patterns
                daily_seasonality=False,   # Too granular for case arrivals
                changepoint_prior_scale=0.05,  # Default sensitivity
                seasonality_mode='multiplicative'  # Better for count data
            )

            # Fit model (suppress output)
            model.fit(df_prophet)

            # Get predictions for the historical period
            forecast = model.predict(df_prophet)

            # Extract insights
            insights = {
                'prophet_available': True,
                'weekly_seasonality': None,
                'weekend_effect': None,
                'changepoints': [],
                'num_changepoints': 0
            }

            # 1. Weekly seasonality effect
            if 'weekly' in forecast.columns:
                weekly_effect = forecast['weekly'].values
                # Calculate weekend vs weekday difference
                insights['weekly_seasonality'] = {
                    'min': float(np.min(weekly_effect)),
                    'max': float(np.max(weekly_effect)),
                    'range': float(np.max(weekly_effect) - np.min(weekly_effect))
                }

            # 2. Weekend effect (if we have daily data)
            weekend_effect = self._calculate_weekend_effect(model, df_prophet)
            if weekend_effect is not None:
                insights['weekend_effect'] = weekend_effect

            # 3. Changepoints (dates where trend changed significantly)
            if hasattr(model, 'changepoints') and model.changepoints is not None:
                changepoints = model.changepoints.tolist()
                # Filter to significant changepoints (where delta is large)
                if hasattr(model, 'params') and 'delta' in model.params:
                    deltas = model.params['delta'].flatten()
                    significant_mask = np.abs(deltas) > np.std(deltas)
                    significant_changepoints = [
                        cp for cp, sig in zip(changepoints, significant_mask) if sig
                    ]
                    insights['changepoints'] = [cp.strftime('%Y-%m-%d') for cp in significant_changepoints[:3]]
                else:
                    insights['changepoints'] = [cp.strftime('%Y-%m-%d') for cp in changepoints[:3]]

                insights['num_changepoints'] = len(insights['changepoints'])

            return insights

        except Exception as e:
            # Prophet failed - return None, fall back to Mann-Kendall only
            print(f"Prophet analysis failed: {e}")
            return None

    def _calculate_weekend_effect(self, model: 'Prophet', df: pd.DataFrame) -> Optional[float]:
        """
        Calculate the weekend effect as percentage difference from weekdays.

        Returns
        -------
        float or None
            Percentage difference (negative = fewer cases on weekend)
        """
        try:
            # Generate predictions for a full week
            week_dates = pd.date_range(start='2024-01-01', periods=7, freq='D')  # Mon-Sun
            week_df = pd.DataFrame({'ds': week_dates})
            week_pred = model.predict(week_df)

            if 'weekly' not in week_pred.columns:
                return None

            weekly_effect = week_pred['weekly'].values

            # Monday=0, Sunday=6
            weekday_avg = np.mean(weekly_effect[:5])  # Mon-Fri
            weekend_avg = np.mean(weekly_effect[5:])  # Sat-Sun

            if weekday_avg != 0:
                effect_pct = ((weekend_avg - weekday_avg) / abs(weekday_avg)) * 100
                return round(effect_pct, 1)

            return None

        except Exception:
            return None

    def detect(self, df: pd.DataFrame) -> bool:
        """
        Detect case arrival trends.

        Steps:
        1. Extract min(actual_time) per case_id (case arrival time)
        2. Aggregate arrivals by time period (weekly)
        3. Apply Mann-Kendall test (primary)
        4. Optionally run Prophet for additional insights
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

            # Primary: Mann-Kendall test
            self.trend_result = self._mann_kendall_test(arrival_counts.values)
            self.trend_result['total_cases'] = len(case_arrivals)
            self.trend_result['num_periods'] = len(arrival_counts)

            # Optional: Prophet analysis for additional insights
            if self.use_prophet:
                # For Prophet, use daily aggregation for better seasonality detection
                daily_counts = case_arrivals.set_index('arrival_time').resample('D').size()
                self.prophet_insights = self._prophet_analysis(daily_counts)

            self.detected = self.trend_result
            return self.trend_result['direction'] != 'no_trend'

        except Exception:
            self.detected = None
            return False

    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """No visual overlay - results shown in tab only."""
        return fig

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of detected case arrival trend with Prophet insights."""
        if self.detected is None or self.trend_result is None:
            return {'count': 0, 'details': None}

        direction = self.trend_result['direction']
        slope_pct = self.trend_result['slope_percent']
        slope_str = f"{slope_pct:+.1f}%" if abs(slope_pct) >= 0.1 else "~0%"

        summary = {
            'count': 1 if direction not in ['no_trend', 'insufficient_data'] else 0,
            'direction': direction,
            'slope_percent': slope_pct,
            'p_value': self.trend_result['p_value'],
            'total_cases': self.trend_result['total_cases'],
            'details': {
                'summary_text': f"Case arrivals: {direction} ({slope_str}/week)"
            }
        }

        # Add Prophet insights if available
        if self.prophet_insights:
            pi = self.prophet_insights

            # Weekend effect
            if pi.get('weekend_effect') is not None:
                effect = pi['weekend_effect']
                if abs(effect) >= 10:  # Only show if significant (>=10%)
                    summary['weekend_effect'] = effect
                    effect_str = f"{effect:+.0f}%" if effect != 0 else "~0%"
                    summary['details']['weekend_text'] = f"Weekend effect: {effect_str}"

            # Changepoints
            if pi.get('changepoints'):
                summary['changepoints'] = pi['changepoints']
                summary['num_changepoints'] = pi['num_changepoints']
                if pi['changepoints']:
                    summary['details']['changepoint_text'] = f"Trend changes: {', '.join(pi['changepoints'][:2])}"

            # Weekly seasonality strength
            if pi.get('weekly_seasonality'):
                ws = pi['weekly_seasonality']
                if ws['range'] > 0.1:  # Meaningful variation
                    summary['has_weekly_pattern'] = True
                    summary['details']['seasonality_text'] = "Weekly pattern detected"

        return summary
