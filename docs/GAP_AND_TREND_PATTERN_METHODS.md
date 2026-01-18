# Gap Detection and Trend Analysis Methods

This document provides a scientific description of the gap detection and trend analysis algorithms implemented in the Visual Pattern Detection system. It is intended for inclusion in the Methods section of a tool paper.

---

## Table of Contents

1. [Gap Detection](#1-gap-detection)
   - 1.1 [Overview: Two Detection Modes](#11-overview-two-detection-modes)
   - 1.2 [Mode 1: Transition Gap Detection (Process-Aware)](#12-mode-1-transition-gap-detection-process-aware)
   - 1.3 [Mode 2: Resource Inactivity Detection](#13-mode-2-resource-inactivity-detection)
   - 1.4 [Statistical Threshold Computation](#14-statistical-threshold-computation)
   - 1.5 [Architectural Decisions](#15-architectural-decisions)
   - 1.6 [Complexity Analysis](#16-complexity-analysis)
   - 1.7 [Visualization Strategy](#17-visualization-strategy)
2. [Trend Detection](#2-trend-detection)
   - 2.1 [Case Arrival Trend Pattern](#21-case-arrival-trend-pattern)
   - 2.2 [General Trend Pattern](#22-general-trend-pattern)
   - 2.3 [Mann-Kendall Test](#23-mann-kendall-test)
   - 2.4 [Sen's Slope Estimator](#24-sens-slope-estimator)
   - 2.5 [Architectural Decisions](#25-architectural-decisions)
   - 2.6 [Prophet Integration for Advanced Insights](#26-prophet-integration-for-advanced-insights)
3. [References](#3-references)

---

## 1. Gap Detection

### 1.1 Overview: Two Detection Modes

The gap detection system supports **two semantically distinct modes** that serve different analytical purposes:

| Mode | Grouping | Semantic Meaning | Process-Flow Gap? |
|------|----------|------------------|-------------------|
| **Transition** (default) | Per Case | Delays between activities within a case | **Yes** |
| **Resource Inactivity** | Per Resource | Periods of resource unavailability | **No** |

**Critical Distinction:** These modes answer fundamentally different questions:
- **Transition Mode:** "Which activity transitions in the process are taking abnormally long?"
- **Resource Inactivity Mode:** "Which resources have unusually long periods without any events?"

The modes are **mutually exclusive**—they are never mixed or overlaid. Users select the mode via a dropdown in the UI, and the detection re-runs with the appropriate algorithm.

---

### 1.2 Mode 1: Transition Gap Detection (Process-Aware)

#### 1.2.1 Theoretical Foundation

Transition gap detection identifies abnormal waiting times between consecutive activities **within process instances (cases)**. This approach is **process-aware**: it considers the semantic meaning of activity transitions and learns normality per transition type.

**Definition (Transition Gap):** Given a case $c$ with ordered events $e_1, e_2, \ldots, e_n$, the gap $g_{i,i+1}$ between consecutive events $e_i$ and $e_{i+1}$ is defined as:

$$g_{i,i+1} = t(e_{i+1}) - t(e_i)$$

where $t(e)$ denotes the timestamp of event $e$.

**Definition (Activity Transition):** A transition $\tau$ is a tuple $(A_{\text{from}}, A_{\text{to}})$ representing the movement from activity $A_{\text{from}}$ to activity $A_{\text{to}}$.

The key insight is that different transitions have inherently different expected durations. For example, the transition "Lab Test → Diagnosis" may typically take 2 hours, while "Registration → Triage" may take 10 minutes. A 30-minute gap would be normal for the former but anomalous for the latter.

This approach aligns with the concept of **performance analysis** in process mining [1], where waiting times between activities are analyzed to identify bottlenecks.

#### 1.2.2 Algorithm Design

```
Algorithm 1a: Transition Gap Detection (Process-Aware)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Event log L = {(c, a, t, r) | case c, activity a, time t, resource r}
        Minimum samples threshold k (default: 15)
Output: Set of abnormal gaps G_abnormal

Phase 1: Transition Extraction (Case-Grouped)
  for each case c ∈ L do
    Sort events by timestamp
    for i = 1 to |events(c)| - 1 do
      τ ← (activity(e_i), activity(e_{i+1}))
      g ← t(e_{i+1}) - t(e_i)
      Add (τ, g, c, e_i, e_{i+1}) to transition_gaps
    end for
  end for

Phase 2: Normality Learning (per transition)
  for each unique transition τ do
    D_τ ← {g | (τ, g, _, _, _) ∈ transition_gaps}
    if |D_τ| ≥ k then
      threshold_τ ← max(P95, Q3 + 1.5 × IQR)
      Store (τ, threshold_τ, statistics)
    end if
  end for

Phase 3: Anomaly Identification
  G_abnormal ← ∅
  for each (τ, g, c, e_from, e_to) ∈ transition_gaps do
    if threshold_τ exists AND g > threshold_τ then
      severity ← g / threshold_τ
      Add (τ, g, c, e_from, e_to, severity) to G_abnormal
    end if
  end for

Phase 4: Return G_abnormal with transition-specific metadata
```

**Requirements:** `case_id` and `activity` columns must be present.

---

### 1.3 Mode 2: Resource Inactivity Detection

#### 1.3.1 Theoretical Foundation

Resource inactivity detection identifies **periods when resources have no events**. This is fundamentally different from transition gap detection:

- **Not a process-flow gap:** Resource inactivity does not represent delays within a case's process flow
- **Resource availability analysis:** Detects when resources are unavailable, offline, or idle
- **Cross-case:** Gaps span across cases (events before and after the gap may belong to different cases)

**Definition (Resource Inactivity Gap):** Given a resource $r$ with all events $E_r = \{e_1, e_2, \ldots, e_n\}$ sorted by timestamp, the inactivity gap $g_{i,i+1}^r$ between consecutive events is:

$$g_{i,i+1}^r = t(e_{i+1}) - t(e_i)$$

Note that $e_i$ and $e_{i+1}$ may belong to **different cases**—this is expected and correct for resource inactivity analysis.

**Use Cases:**
- Detecting resource unavailability (vacation, illness, maintenance)
- Identifying equipment downtime
- Finding shift gaps or operational hours
- Capacity planning analysis

#### 1.3.2 Algorithm Design

```
Algorithm 1b: Resource Inactivity Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Event log L = {(c, a, t, r) | case c, activity a, time t, resource r}
        Minimum samples threshold k (default: 15)
Output: Set of abnormal inactivity gaps G_abnormal

Phase 1: Resource Gap Extraction (Resource-Grouped, NOT Case-Grouped)
  for each resource r ∈ L do
    Sort events by timestamp (across ALL cases)
    for i = 1 to |events(r)| - 1 do
      g ← t(e_{i+1}) - t(e_i)
      Add (r, g, e_i, e_{i+1}) to resource_gaps
    end for
  end for

Phase 2: Normality Learning (per resource)
  for each unique resource r do
    D_r ← {g | (r, g, _, _) ∈ resource_gaps}
    if |D_r| ≥ k then
      threshold_r ← max(P95, Q3 + 1.5 × IQR)
      Store (r, threshold_r, statistics)
    end if
  end for

Phase 3: Anomaly Identification
  G_abnormal ← ∅
  for each (r, g, e_from, e_to) ∈ resource_gaps do
    if threshold_r exists AND g > threshold_r then
      severity ← g / threshold_r
      Add (r, g, e_from, e_to, severity) to G_abnormal
    end if
  end for

Phase 4: Return G_abnormal with resource-specific metadata
         (includes case_from, case_to for context)
```

**Requirements:** `resource` column must be present. Resource inactivity mode is only meaningful when the Y-axis represents resources.

---

### 1.4 Statistical Threshold Computation

The threshold computation combines two robust statistical methods to handle skewed distributions commonly found in waiting time data:

**Tukey's Interquartile Range (IQR) Method [2]:**

$$\text{threshold}_{\text{IQR}} = Q_3 + 1.5 \times \text{IQR}$$

where $\text{IQR} = Q_3 - Q_1$ and $Q_1, Q_3$ are the first and third quartiles.

**95th Percentile (P95):**

$$\text{threshold}_{P95} = F^{-1}(0.95)$$

where $F^{-1}$ is the inverse cumulative distribution function of the gap durations.

**Combined Threshold:**

$$\text{threshold}_\tau = \max(\text{threshold}_{P95}, \text{threshold}_{\text{IQR}})$$

**Rationale:** The IQR method is robust against outliers and assumes approximate symmetry, while P95 directly captures the upper tail of the distribution. Taking the maximum ensures we only flag truly extreme gaps while accommodating both symmetric and skewed distributions.

**Minimum Sample Requirement:** We require $|D_\tau| \geq k$ (default $k=15$) samples per transition before computing thresholds. This ensures stable percentile and IQR estimates—with fewer than 15 samples, quartile calculations become unreliable, especially for skewed waiting time distributions [3]. The threshold is user-configurable (range: 3-30) to accommodate domain-specific needs.

### 1.5 Architectural Decisions

#### 1.5.1 Mode Selection Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Two distinct modes** | transition, resource_inactivity | Fundamentally different semantic meanings. Transition = process delays; Resource = availability. Mixing would confuse interpretation. |
| **Mutual exclusivity** | Never mix modes | Each mode answers a different question. Overlaying both would create visual and semantic confusion. |
| **Mode-specific UI labels** | "Transitions affected" vs "Resources affected" | Clear communication of what the numbers mean in each mode. |
| **Resource mode constraint** | Only when Y=resource | Resource inactivity is only meaningful when viewing resources on Y-axis. Other Y-axes would produce misleading visualizations. |

#### 1.5.2 Common Decisions (Both Modes)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Threshold formula** | max(P95, Q3+1.5×IQR) | Combines distribution tail capture (P95) with robust outlier detection (IQR). Handles both normal and skewed distributions. |
| **Minimum samples** | k=15 (configurable 3-30) | Ensures stable percentile estimation. Research suggests ≥15 samples for reliable quartile estimates [3]. |
| **Severity metric** | duration/threshold | Provides interpretable measure: severity=2.0 means gap is twice the normal threshold. Enables prioritization. |
| **Time validation** | duration > 0 | Filters data quality issues (negative durations from timestamp errors). |
| **Max display limit** | N=50 gaps | Prevents visual clutter. Only top N gaps by severity are rendered; all are still available in data. |
| **Severity categories** | 4 groups | Mild (1-2×), Moderate (2-3×), Severe (3-5×), Critical (>5×). Enables filtering and legend toggle. |

#### 1.5.3 Mode-Specific Decisions

| Decision | Transition Mode | Resource Inactivity Mode |
|----------|-----------------|--------------------------|
| **Grouping key** | Activity transition (A→B) | Resource |
| **Extraction scope** | Within each case | Within each resource (across cases) |
| **Required columns** | case_id, activity | resource |
| **Hover text** | Transition, Case ID, Duration | Resource, Duration, "Not a process-flow gap" |
| **Summary label** | "Transitions with anomalies" | "Resources with anomalies" |
| **Selection widget** | Transition multiselect | Resource multiselect |

#### 1.5.4 Out of Scope (By Design)

The following features are **explicitly not implemented** to keep the detection focused and interpretable:

- **Shift calendars:** No modeling of working hours, weekends, or holidays
- **Organizational availability:** No HR or capacity models
- **Mixed mode display:** Never overlay transition gaps and resource inactivity
- **Automatic mode selection:** User must explicitly choose the mode

### 1.6 Complexity Analysis

Let $n$ be the number of events, $c$ the number of cases, $\tau$ the number of unique transitions, and $r$ the number of unique resources.

#### Transition Mode

| Phase | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| Transition Extraction | $O(n \log n)$ | $O(n)$ |
| Normality Learning | $O(\tau \cdot m \log m)$ | $O(\tau)$ |
| Anomaly Identification | $O(n)$ | $O(g)$ |
| **Total** | $O(n \log n + \tau \cdot m \log m)$ | $O(n + \tau + g)$ |

where $m$ is the average number of gaps per transition and $g$ is the number of detected abnormal gaps.

#### Resource Inactivity Mode

| Phase | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| Resource Gap Extraction | $O(n \log n)$ | $O(n)$ |
| Normality Learning | $O(r \cdot m_r \log m_r)$ | $O(r)$ |
| Anomaly Identification | $O(n)$ | $O(g)$ |
| **Total** | $O(n \log n + r \cdot m_r \log m_r)$ | $O(n + r + g)$ |

where $m_r$ is the average number of gaps per resource.

### 1.7 Visualization Strategy

Effective visualization of abnormal gaps requires balancing informativeness with clarity. We employ a multi-dimensional encoding scheme based on visualization research principles [16].

#### 1.7.1 Severity-Based Visual Encoding

Gaps are visualized using **dashed connecting lines** between the source and target events. Visual properties encode severity:

**Color Gradient (Categorical):**

| Severity Category | Range | Color | RGBA |
|-------------------|-------|-------|------|
| Mild | 1-2× threshold | Yellow | `rgba(255, 193, 7, 0.7)` |
| Moderate | 2-3× threshold | Orange | `rgba(255, 152, 0, 0.8)` |
| Severe | 3-5× threshold | Red | `rgba(220, 53, 69, 0.85)` |
| Critical | >5× threshold | Dark Red | `rgba(139, 0, 0, 0.9)` |

**Line Width (Continuous):**

$$w = \min(1.5 + 0.4 \cdot s, 5)$$

where $s$ is the severity and $w$ is the line width in pixels. This provides a range of 1.9px (mild) to 5px (critical).

**Marker Size:**

$$m = 6 + \bar{s}$$

where $\bar{s}$ is the average severity of the gap group.

#### 1.7.2 Display Limiting

To prevent visual clutter in logs with many abnormal gaps, we limit display to the **top N gaps by severity** (default N=50):

```
gaps_to_display = sorted(abnormal_gaps, key=severity, reverse=True)[:N]
```

**Rationale:** In large event logs, hundreds of gaps may exceed the threshold. Displaying all of them creates visual noise and obscures the most critical issues. By showing only the top 50 (configurable), users can focus on the most severe anomalies while still having access to the complete data for analysis.

#### 1.7.3 Grouped Traces for Interactivity

Gaps are grouped by severity category into separate Plotly traces:

```
traces = [
    "Critical (>5x) (n)",
    "Severe (3-5x) (n)",
    "Moderate (2-3x) (n)",
    "Mild (1-2x) (n)"
]
```

**Benefits:**
- Users can toggle visibility of specific severity levels via the legend
- Provides immediate overview of severity distribution
- Reduces visual complexity by hiding less important gaps

#### 1.7.4 Mode-Specific Hover Information

Each gap displays **mode-specific contextual information** on hover:

**Transition Mode:**
```
<b>Activity A → Activity B</b>
Duration: 4.2h (threshold: 1.1h)
Severity: 3.8x
Case: case_123
```

**Resource Inactivity Mode:**
```
<b>Resource: ResourceX</b>
Inactivity: 48.5h (threshold: 12.3h)
Severity: 3.9x
<i>Not a process-flow gap</i>
```

The explicit "Not a process-flow gap" disclaimer in resource inactivity mode reinforces the semantic distinction to users.

---

## 2. Trend Detection

The system implements two complementary trend detection approaches: **Case Arrival Trend** (analyzing when new process instances start) and **General Trend** (analyzing overall event frequency). Both use the Mann-Kendall test for statistical significance.

### 2.1 Case Arrival Trend Pattern

**Definition (Case Arrival Time):** For a case $c$ with events $E_c = \{e_1, \ldots, e_n\}$, the arrival time is:

$$t_{\text{arrival}}(c) = \min_{e \in E_c} t(e)$$

**Motivation:** Case arrival trends reveal changes in process demand over time—are more cases being initiated? Is workload increasing or decreasing? This is distinct from event frequency, which could increase due to longer cases rather than more cases.

```
Algorithm 2: Case Arrival Trend Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Event log L, aggregation period P (default: weekly)
        Minimum periods k (default: 5)
        Significance level α (default: 0.05)
Output: Trend direction, slope, p-value

Step 1: Extract case arrivals
  for each unique case c ∈ L do
    t_arrival(c) ← min timestamp in case c
  end for

Step 2: Aggregate by time period
  counts ← resample(arrivals, period=P).count()

Step 3: Statistical test
  if |counts| < k then
    return "insufficient_data"
  end if

  result ← MannKendall(counts)
  slope ← SensSlope(counts)

Step 4: Determine direction
  if result.p_value ≤ α AND |slope_percent| ≥ 0.5% then
    direction ← "increasing" if slope > 0 else "decreasing"
  else if result.p_value ≤ α then
    direction ← "stable"
  else
    direction ← "no_trend"
  end if

  return (direction, slope, p_value)
```

### 2.2 General Trend Pattern

The general trend pattern analyzes overall event frequency over time, with optional per-category breakdown (e.g., trends per resource or per activity).

**Key Difference from Case Arrival:**
- **Case Arrival:** Counts first events per case → measures process initiation rate
- **General Trend:** Counts all events → measures overall activity volume

Both are important: a stable case arrival rate with increasing event frequency suggests cases are becoming more complex or taking longer.

### 2.3 Mann-Kendall Test

The Mann-Kendall test [4, 5] is a non-parametric test for monotonic trends in time series data. It is particularly suitable for process mining data because:

1. **Non-parametric:** No assumption about data distribution
2. **Robust to outliers:** Based on ranks, not values
3. **Handles tied values:** Common in count data

**Test Statistic S:**

$$S = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \text{sgn}(x_j - x_i)$$

where $\text{sgn}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x = 0 \\ -1 & \text{if } x < 0 \end{cases}$

**Variance (with tie correction):**

$$\text{Var}(S) = \frac{n(n-1)(2n+5) - \sum_{p=1}^{g} t_p(t_p-1)(2t_p+5)}{18}$$

where $g$ is the number of tied groups and $t_p$ is the size of the $p$-th tied group.

**Z-Score:**

$$Z = \begin{cases}
\frac{S-1}{\sqrt{\text{Var}(S)}} & \text{if } S > 0 \\
0 & \text{if } S = 0 \\
\frac{S+1}{\sqrt{\text{Var}(S)}} & \text{if } S < 0
\end{cases}$$

**P-Value:** Two-tailed p-value from standard normal distribution:

$$p = 2 \cdot (1 - \Phi(|Z|))$$

where $\Phi$ is the standard normal CDF.

### 2.4 Sen's Slope Estimator

Sen's slope [6] provides a robust estimate of trend magnitude:

$$\beta = \text{median}\left\{ \frac{x_j - x_i}{j - i} : 1 \leq i < j \leq n \right\}$$

**Interpretation:** The slope represents the median rate of change per time period. We convert this to a percentage of the mean:

$$\beta_{\%} = \frac{\beta}{\bar{x}} \times 100\%$$

This allows comparison across different scales (e.g., a slope of 2 cases/week is more significant when the average is 10 cases than when it's 1000).

### 2.5 Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Test selection** | Mann-Kendall | Non-parametric, robust to outliers, handles ties. More appropriate than linear regression for count data with potential non-normality. |
| **Slope estimator** | Sen's slope | Robust to outliers (uses median). Consistent with Mann-Kendall's non-parametric nature. |
| **Practical significance** | ≥0.5% slope | Statistical significance alone insufficient. A p<0.05 with 0.01%/week change is meaningless in practice. |
| **Aggregation period** | Weekly (default) | Balances granularity with noise reduction. Too fine (hourly) captures noise; too coarse (monthly) misses trends. Configurable. |
| **Minimum periods** | k=5 | Ensures sufficient data for reliable trend estimation. Mann-Kendall requires at least 4-5 observations for meaningful results [7]. |
| **Per-category analysis** | Optional | Enables drill-down (e.g., which resource shows increasing workload?). Excluded for case_id (unique per case). |
| **Visualization** | Annotation box | Trend is about frequency over time, but Y-axis is often categorical. Annotation communicates trend without misleading visual. |
| **Dual patterns** | Case Arrival + General | Different semantic meanings. Case arrival = demand; General = workload. Both needed for complete picture. |

### 2.6 Trend Classification Logic

```
Classification Decision Tree:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ┌─────────────────────┐
                    │  Mann-Kendall Test  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   p-value ≤ 0.05?   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │ YES            │                │ NO
              ▼                ▼                ▼
    ┌─────────────────┐ ┌──────────────┐ ┌─────────────┐
    │ |slope| ≥ 0.5%? │ │              │ │  NO_TREND   │
    └────────┬────────┘ │              │ └─────────────┘
             │          │              │
     ┌───────┴───────┐  │              │
     │ YES           │ NO              │
     ▼               ▼                 │
┌─────────────┐ ┌─────────┐           │
│ slope > 0?  │ │ STABLE  │           │
└──────┬──────┘ └─────────┘           │
       │                              │
   ┌───┴───┐                          │
   ▼       ▼                          │
┌──────┐ ┌──────────┐                 │
│  ↗   │ │    ↘     │                 │
│ INC  │ │   DEC    │                 │
└──────┘ └──────────┘                 │
```

### 2.7 Prophet Integration for Advanced Insights

While Mann-Kendall provides robust monotonic trend detection, real-world process data often exhibits complex temporal patterns that require more sophisticated analysis. We optionally integrate **Prophet** [11], Meta's open-source forecasting library, to provide additional insights.

#### 2.7.1 Theoretical Background

Prophet implements a **decomposable time series model** [11, 12]:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

where:
- $g(t)$ is the **trend function** modeling non-periodic changes
- $s(t)$ is the **seasonality function** capturing periodic patterns (weekly, yearly)
- $h(t)$ represents **holiday/event effects**
- $\epsilon_t$ is the error term (assumed normally distributed)

**Trend Component:** Prophet uses a piecewise linear or logistic growth model with automatic changepoint detection:

$$g(t) = (k + \mathbf{a}(t)^T \boldsymbol{\delta}) \cdot t + (m + \mathbf{a}(t)^T \boldsymbol{\gamma})$$

where:
- $k$ is the base growth rate
- $\boldsymbol{\delta}$ is a vector of rate adjustments at changepoints
- $\mathbf{a}(t)$ is an indicator function for changepoints before time $t$
- $m$ is the offset parameter

**Seasonality Component:** Seasonality is modeled using Fourier series [13]:

$$s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi nt}{P}\right) + b_n \sin\left(\frac{2\pi nt}{P}\right) \right)$$

where $P$ is the period (e.g., 7 for weekly seasonality) and $N$ determines the smoothness.

#### 2.7.2 Process Mining Application

In process mining contexts, Prophet provides three key insights not available from Mann-Kendall:

**1. Weekly Seasonality Detection**

Healthcare and business processes often exhibit strong weekly patterns. For example:
- Fewer case arrivals on weekends
- Higher activity on Mondays (backlog from weekend)
- Mid-week peaks for certain activities

We quantify the **weekend effect** as:

$$\text{Weekend Effect} = \frac{\bar{s}_{\text{weekend}} - \bar{s}_{\text{weekday}}}{\bar{s}_{\text{weekday}}} \times 100\%$$

where $\bar{s}_{\text{weekend}}$ and $\bar{s}_{\text{weekday}}$ are the mean seasonal components for weekend and weekday periods.

**2. Changepoint Detection**

Process changes (new policies, system updates, organizational restructuring) often cause sudden shifts in case arrival patterns. Prophet automatically detects these changepoints using a sparse prior on the rate change vector $\boldsymbol{\delta}$ [11]:

$$\delta_j \sim \text{Laplace}(0, \tau)$$

where $\tau$ controls the flexibility of the model. We filter to **significant changepoints** where:

$$|\delta_j| > \sigma_\delta$$

where $\sigma_\delta$ is the standard deviation of all changepoint magnitudes.

**3. Multiplicative Seasonality**

For count data (case arrivals), we use multiplicative seasonality mode:

$$y(t) = g(t) \cdot (1 + s(t)) + \epsilon_t$$

This is more appropriate than additive seasonality because a 20% weekend reduction should scale with the overall volume—if weekdays have 100 cases, weekends have 80; if weekdays have 1000, weekends have 800.

#### 2.7.3 Algorithm Integration

```
Algorithm 3: Prophet-Enhanced Trend Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Daily case arrival counts (minimum 14 days)
Output: Seasonality insights, changepoints

Step 1: Data preparation
  df_prophet ← DataFrame with columns 'ds' (date), 'y' (count)

Step 2: Model configuration
  model ← Prophet(
    yearly_seasonality = False,      # Usually insufficient data
    weekly_seasonality = True,       # Primary interest
    daily_seasonality = False,       # Too granular
    seasonality_mode = 'multiplicative',
    changepoint_prior_scale = 0.05   # Default sensitivity
  )

Step 3: Fit and predict
  model.fit(df_prophet)
  forecast ← model.predict(df_prophet)

Step 4: Extract insights
  weekly_effect ← forecast['weekly']
  weekend_effect ← calculate_weekend_effect(model)
  changepoints ← filter_significant_changepoints(model)

Step 5: Return insights
  return {
    'weekend_effect': weekend_effect,      # e.g., -72%
    'changepoints': changepoints,          # e.g., ['2024-04-15']
    'has_weekly_pattern': range(weekly_effect) > 0.1
  }
```

#### 2.7.4 Architectural Decisions for Prophet Integration

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Optional dependency** | Prophet via try/except | Heavy dependency (~100MB). System works without it. |
| **Primary vs. enhancement** | Mann-Kendall primary | Mann-Kendall is simpler, more interpretable, scientifically established. Prophet adds optional depth. |
| **Minimum data requirement** | 14 days | Prophet needs sufficient data for weekly pattern detection. Less than 2 weeks is unreliable. |
| **Seasonality mode** | Multiplicative | Better for count data where effects scale with volume. |
| **Yearly seasonality** | Disabled | Most process logs don't span full years. Enabling would overfit. |
| **Changepoint sensitivity** | 0.05 (default) | Balanced between detecting real changes and avoiding false positives. |
| **Significant changepoint filter** | \|δ\| > σ_δ | Only report changepoints with meaningful trend changes. |
| **Weekend effect threshold** | ≥10% | Only report if practically significant. Small effects are noise. |

#### 2.7.5 Interpretation Guidelines

| Insight | Example | Interpretation |
|---------|---------|----------------|
| Weekend effect: -72% | Hospital ER | 72% fewer cases on weekends—expected for elective procedures, concerning for emergency care |
| Changepoint: 2024-04-15 | Process log | Investigate: new policy? system change? external event? |
| Weekly pattern detected | Business process | Strong weekday/weekend distinction—consider separate SLAs |
| No weekly pattern | 24/7 operation | Process operates continuously without weekly cycles |

---

## 3. References

[1] van der Aalst, W.M.P. (2016). **Process Mining: Data Science in Action** (2nd ed.). Springer. Chapter 8: Performance Analysis.

[2] Tukey, J.W. (1977). **Exploratory Data Analysis**. Addison-Wesley. (Original IQR method for outlier detection)

[3] Hyndman, R.J. & Fan, Y. (1996). Sample Quantiles in Statistical Packages. **The American Statistician**, 50(4), 361-365. (Percentile estimation methods)

[4] Mann, H.B. (1945). Nonparametric Tests Against Trend. **Econometrica**, 13(3), 245-259.

[5] Kendall, M.G. (1975). **Rank Correlation Methods** (4th ed.). Charles Griffin.

[6] Sen, P.K. (1968). Estimates of the Regression Coefficient Based on Kendall's Tau. **Journal of the American Statistical Association**, 63(324), 1379-1389.

[7] Yue, S., Pilon, P., Phinney, B., & Cavadias, G. (2002). The influence of autocorrelation on the ability to detect trend in hydrological series. **Hydrological Processes**, 16(9), 1807-1829. (Minimum sample size recommendations)

[8] Cleveland, W.S. (1979). Robust Locally Weighted Regression and Smoothing Scatterplots. **Journal of the American Statistical Association**, 74(368), 829-836. (LOWESS smoothing)

[9] Augusto, A., et al. (2019). Automated Discovery of Process Models from Event Logs: Review and Benchmark. **IEEE Transactions on Knowledge and Data Engineering**, 31(4), 686-705. (Process mining benchmarks)

[10] Suriadi, S., Andrews, R., ter Hofstede, A.H.M., & Wynn, M.T. (2017). Event log imperfection patterns for process mining: Towards a systematic approach to cleaning event logs. **Information Systems**, 64, 132-150. (Data quality in process mining)

[11] Taylor, S.J. & Letham, B. (2018). Forecasting at Scale. **The American Statistician**, 72(1), 37-45. (Prophet algorithm and implementation)

[12] Harvey, A.C. & Peters, S. (1990). Estimation procedures for structural time series models. **Journal of Forecasting**, 9(2), 89-108. (Structural time series models foundation)

[13] Bloomfield, P. (2000). **Fourier Analysis of Time Series: An Introduction** (2nd ed.). Wiley. (Fourier series for seasonality modeling)

[14] Adams, R.P. & MacKay, D.J.C. (2007). Bayesian Online Changepoint Detection. **arXiv preprint arXiv:0710.3742**. (Changepoint detection theory)

[15] Aminikhanghahi, S. & Cook, D.J. (2017). A Survey of Methods for Time Series Change Point Detection. **Knowledge and Information Systems**, 51(2), 339-367. (Comprehensive changepoint detection survey)

[16] Munzner, T. (2014). **Visualization Analysis and Design**. CRC Press. (Visual encoding principles and effectiveness rankings)

---

## Appendix A: Implementation Details

### A.1 Gap Pattern Class Structure

```python
class GapPattern(Pattern):
    """
    Gap detector supporting two distinct modes:
    - transition: Process-aware gaps within cases (default)
    - resource_inactivity: Resource-timeline gaps (NOT process-flow)

    Class Constants:
        MIN_SAMPLES_FOR_NORMALITY: int = 15  # Minimum samples for stable thresholds
        MAX_GAPS_TO_DISPLAY: int = 50        # Limit visualization to top N

    Attributes:
        view_config: Dict[str, str]      # x, y column configuration
        y_is_categorical: bool           # Affects Y-position computation
        gap_mode: str                    # "transition" or "resource_inactivity"
        detected: Optional[Dict]         # Detection results (includes gap_mode)
        transition_stats: Dict           # Per-group statistics (transition or resource)

    Key Methods:
        detect(df) → None                # Main detection entry point (mode-aware)
        visualize(df, fig) → Figure      # Add severity-colored gap overlays
        get_summary() → Dict             # Standardized pattern summary
        get_gap_summary() → Dict         # Mode-specific gap statistics

    Internal Methods:
        _extract_transition_gaps(df, x_col, y_col) → List[Dict]  # Case-grouped extraction
        _extract_resource_gaps(df, x_col) → List[Dict]           # Resource-grouped extraction
        _compute_normality_per_group(gaps, group_key) → Dict     # Generic threshold computation

    Static Methods:
        _severity_to_color(severity) → str    # Map severity to RGBA color
        _severity_to_width(severity) → float  # Map severity to line width
        _format_duration(seconds) → str       # Human-readable duration format

    Mode-Specific Output Fields:
        Transition Mode:
            - total_transitions, transitions_with_anomalies
            - gap['transition'], gap['case_id'], gap['activity_from'], gap['activity_to']
        Resource Inactivity Mode:
            - total_resources, resources_with_anomalies
            - gap['resource'], gap['case_from'], gap['case_to']
    """
```

### A.2 Trend Pattern Class Structure

```python
# Optional Prophet availability check
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class CaseArrivalTrendPattern(Pattern):
    """
    Attributes:
        aggregation_period: str = 'W'    # Pandas frequency string
        min_periods: int = 5             # Minimum periods for analysis
        significance_level: float = 0.05 # Alpha for Mann-Kendall
        use_prophet: bool = True         # Enable Prophet if available
        trend_result: Optional[Dict]     # Mann-Kendall results
        prophet_insights: Optional[Dict] # Prophet results (if available)

    Key Methods:
        detect(df) → bool                # Returns True if trend detected
        get_summary() → Dict             # Summary with direction, slope, p-value
        _mann_kendall_test(data) → Dict  # Primary trend detection
        _prophet_analysis(counts) → Dict # Optional: seasonality & changepoints
        _calculate_weekend_effect(model) → float  # Weekend vs weekday difference
    """

class TrendPattern(Pattern):
    """
    Additional Attributes:
        analyze_per_category: bool = True  # Per-Y-category breakdown
        global_trend: Dict                 # Overall trend results
        category_trends: Dict              # Per-category results
        trend_line_data: Dict              # LOWESS smoothed data

    Additional Methods:
        _lowess_smooth(x, y, frac) → array  # Trend line computation
        _aggregate_events(df, x, y) → DataFrame  # Time aggregation
    """
```

### A.3 Prophet Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CaseArrivalTrendPattern                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐    ┌─────────────────────────────┐   │
│  │   Mann-Kendall Test  │    │   Prophet Analysis          │   │
│  │   (Primary - Always) │    │   (Optional Enhancement)    │   │
│  ├──────────────────────┤    ├─────────────────────────────┤   │
│  │ • Trend direction    │    │ • Weekly seasonality        │   │
│  │ • p-value            │    │ • Weekend effect (%)        │   │
│  │ • Sen's slope        │    │ • Changepoint dates         │   │
│  │ • Slope percentage   │    │ • Pattern strength          │   │
│  └──────────┬───────────┘    └──────────────┬──────────────┘   │
│             │                                │                   │
│             └────────────┬───────────────────┘                   │
│                          │                                       │
│                          ▼                                       │
│             ┌────────────────────────┐                          │
│             │     get_summary()      │                          │
│             │  Combined Results      │                          │
│             └────────────────────────┘                          │
│                                                                  │
│  Fallback: If Prophet unavailable or fails,                     │
│            returns Mann-Kendall results only                     │
└─────────────────────────────────────────────────────────────────┘
```

### A.4 View Configuration Compatibility

Both patterns require time-based X-axis columns. The following configurations are supported:

| X-Axis | Gap Detection | Trend Detection | Notes |
|--------|---------------|-----------------|-------|
| `actual_time` | ✓ | ✓ | Wall-clock time, most common |
| `relative_time` | ✓ | ✓ | Time since case start |
| `relative_ratio` | ✓ | ✗ | Normalized [0,1], no absolute time |
| `logical_time` | ✓ | ✗ | Event sequence index |
| `logical_relative` | ✓ | ✗ | Normalized event index |

Gap detection requires `activity` and `case_id` columns for transition extraction. Trend detection only requires the time column and optionally a category column for breakdown analysis.

---

## Appendix B: Validation Strategy

### B.1 Synthetic Data Testing

Both patterns are validated using synthetic event logs with known, guaranteed-detectable patterns:

**Gap Detection Validation:**
- Normal gaps: ~0.05 time units (baseline)
- Injected abnormal gaps: 0.4-0.5 time units (8-10× normal)
- Expected: Abnormal gaps detected with severity ≈ 8-10

**Trend Detection Validation:**
- Increasing trend: Linear growth from 10 to 50 cases/period
- Decreasing trend: Linear decline from 50 to 10 cases/period
- No trend: Random fluctuation around mean
- Expected: Correct classification with p < 0.05 for trends

### B.2 Real-World Validation

Both patterns are tested on the Hospital_log.xes dataset to ensure:
1. No crashes on real data
2. Reasonable execution time
3. Interpretable results
4. Graceful handling of edge cases (missing columns, sparse data)

---

*Document Version: 1.3*
*Last Updated: January 2026*
*Authors: Visual Pattern Detection Team*

### Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-08 | Initial version with Gap Detection and Mann-Kendall Trend Analysis |
| 1.1 | 2026-01-08 | Added Prophet integration for seasonality and changepoint detection |
| 1.2 | 2026-01-08 | Enhanced Gap Detection: MIN_SAMPLES=15, severity-based visualization, display limiting, grouped traces |
| 1.3 | 2026-01-09 | **Two Gap Detection Modes:** Added `resource_inactivity` mode alongside existing `transition` mode. Modes are mutually exclusive with mode-specific thresholds, hover text, and UI labels. Resource inactivity explicitly marked as "not a process-flow gap." |
