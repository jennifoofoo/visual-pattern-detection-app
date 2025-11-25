# Gap Detection Guide

## 📋 Overview

Gap Detection identifies **abnormally long waiting times** between process activities using **transition-specific normality learning**. Unlike traditional gap detection that uses fixed thresholds, this implementation learns what's "normal" for each specific activity transition (A → B) and flags deviations.

---

## 🎯 What It Detects

**Process-Aware Gaps:**
- Unusually long delays between consecutive activities **within a case**
- Transition-specific: What's normal for "A → B" may be abnormal for "C → D"
- Case-aware: Only analyzes gaps within case boundaries (respects process structure)

**Example:**
```
Normal: "Order → Payment" usually takes 2 days (learned from 100 cases)
Abnormal: One case has "Order → Payment" taking 14 days
→ Detected as abnormal gap with severity 7.0x
```

---

## 🧠 How It Works: Transition-Specific Normality

### 1. **Extract Transition Gaps**
For each case, analyze consecutive events:

```
Case 001:
Event 1: Order (2024-01-01 10:00)
Event 2: Payment (2024-01-03 11:00)  → Gap: 2.04 days
Event 3: Shipping (2024-01-04 09:00) → Gap: 0.92 days
```

Each gap is labeled with its transition:
- `Order → Payment`: 2.04 days
- `Payment → Shipping`: 0.92 days

### 2. **Learn Normality Per Transition**
Group all gaps by transition and compute statistics:

```
Transition: Order → Payment (n=100 samples)
- Q1: 1.5 days
- Q3: 2.8 days
- IQR: 1.3 days
- P95: 3.2 days
- Threshold: max(P95, Q3 + 1.5×IQR) = max(3.2, 4.75) = 4.75 days
```

### 3. **Detect Abnormal Gaps**
Mark gaps that exceed their transition-specific threshold:

```
Gap: Order → Payment, 14 days
Threshold: 4.75 days
→ Abnormal! (Severity: 14/4.75 = 2.95x)
```

### 4. **Visualize**
Red rectangles overlay the chart showing:
- **X-position**: Time span of the gap
- **Y-position**: Resource/activity where gap occurred (FROM-resource)

---

## ⚙️ Configuration

### Required View Configuration

```python
view_config = {
    'x': 'actual_time',    # Must be time-like
    'y': 'resource'        # Categorical or numeric
}

detector = GapPattern(view_config, y_is_categorical=True)
```

### Supported X-Axis (Time-like)
- `actual_time` ✅ (timestamps)
- `relative_time` ✅ (seconds since case start)
- `relative_ratio` ✅ (normalized [0,1] within case)
- `logical_time` ❌ (not time-based)
- `logical_relative` ❌ (not time-based)

### Supported Y-Axis
- **Categorical:** `resource`, `activity`, `case_id`
  - Gaps drawn on specific resource/activity band
- **Numeric:** Any numeric column
  - Gaps drawn spanning full Y-range

### Parameters

```python
GapPattern(
    view_config: Dict[str, str],
    y_is_categorical: bool = False,
    **kwargs
)

# Set minimum samples per transition (default: 5)
detector.MIN_SAMPLES_FOR_NORMALITY = 5
```

---

## 📊 Meaningful Configurations

### ✅ **Recommended Configurations**

| X-Axis | Y-Axis | Use Case | Interpretation |
|--------|--------|----------|----------------|
| `actual_time` | `resource` | **Resource delays** | Which resources have abnormal waiting times? |
| `actual_time` | `activity` | **Activity delays** | Which activity transitions are delayed? |
| `actual_time` | `case_id` | **Case-specific delays** | Which cases experience delays? |
| `relative_time` | `resource` | **Process delays** | Where in the process do delays occur? |
| `relative_time` | `activity` | **Activity sequence delays** | Time between activities in process flow |

### ❌ **Not Meaningful**

| X-Axis | Y-Axis | Why Not? |
|--------|--------|----------|
| `logical_time` | any | Not time-based, just sequential counter |
| `event_index` | any | Position-based, not time-based |
| any | `variant` | Too many categories, low sample sizes |

---

## 🔍 Interpretation Guide

### Understanding the Results

**1. Transition Statistics**
```
vervolgconsult → vervolgconsult
- 48 occurrences
- Threshold: 472.2 days (P95: 371.2d, Median: 145.3d)
- 3 abnormal gaps detected
```

**Interpretation:**
- This transition happens frequently (48 times)
- Normal duration: ~145 days (median)
- Extreme cases can go up to 371 days (P95)
- Statistical threshold: 472 days
- 3 cases exceeded this → Process issue!

---

**2. Abnormal Gap Details**
```
Gap: Activity A → Activity B
Duration: 12.0 days
Threshold: 6.0 days
Severity: 2.00x
Case: 00000088
Time: 2005-04-01 → 2005-04-13
```

**Interpretation:**
- This specific transition took **twice as long** as statistically expected
- Possible causes:
  - Resource unavailability
  - Weekend/holiday delays
  - Process bottleneck
  - Data quality issue
  - Exceptional case circumstances

---

### Severity Levels

```
Severity = Duration / Threshold
```

**Classification:**
- **1.0x - 1.5x:** Minor deviation (borderline)
- **1.5x - 2.0x:** Moderate anomaly (worth investigating)
- **2.0x - 5.0x:** Significant anomaly (likely process issue)
- **> 5.0x:** Extreme anomaly (data quality or major incident)

---

## 🎨 Visualization

### What You See

**Red Rectangles:**
- **X-span:** Duration of the gap (start to end time)
- **Y-position:** 
  - Categorical Y: Drawn at the FROM-resource/activity band
  - Numeric Y: Spans the entire Y-axis

**Color Intensity:**
- All abnormal gaps shown in red
- Darker overlays = multiple gaps overlapping

### Visual Analysis Tips

1. **Horizontal patterns:** Specific resources/activities consistently have delays
2. **Vertical patterns:** Certain time periods have widespread delays
3. **Isolated rectangles:** Case-specific issues
4. **Dense clusters:** Systemic process bottlenecks

---

## 🔬 Technical Details

### Algorithm

```python
1. Extract Gaps:
   FOR each case:
     FOR each consecutive event pair (i, i+1):
       gap = {
         transition: activity_i → activity_{i+1},
         duration: time_{i+1} - time_i,
         case_id: case,
         x_start: time_i,
         x_end: time_{i+1}
       }

2. Compute Normality:
   FOR each unique transition:
     IF samples >= MIN_SAMPLES:
       Q1, Q3 = quantile(durations, [0.25, 0.75])
       IQR = Q3 - Q1
       P95 = quantile(durations, 0.95)
       threshold = max(P95, Q3 + 1.5 × IQR)

3. Detect Abnormal:
   FOR each gap:
     IF gap.duration > threshold[gap.transition]:
       severity = gap.duration / threshold
       abnormal_gaps.append(gap + {severity})

4. Visualize:
   FOR each abnormal_gap:
     draw_rectangle(x_start, x_end, y_position)
```

### Statistical Robustness

**Why IQR + P95?**
- **IQR (Interquartile Range):** Robust to outliers
- **P95 (95th Percentile):** Captures extreme but valid cases
- **Combined:** `max(P95, Q3 + 1.5×IQR)` balances both approaches

**Why 1.5 × IQR?**
- Standard statistical outlier detection threshold
- Tukey's method for boxplot outliers
- Well-established in literature

**Minimum Samples:**
- Default: 5 samples per transition
- Rationale: Statistical reliability (avoid false positives from small samples)
- Configurable via `MIN_SAMPLES_FOR_NORMALITY`

---

## 🚀 Usage Example

### Basic Usage

```python
from core.detection.gap_pattern import GapPattern
import pandas as pd

# Load event log
df = pd.read_csv('event_log.csv')

# Configure
view_config = {
    'x': 'actual_time',
    'y': 'resource'
}

# Detect
detector = GapPattern(view_config, y_is_categorical=True)
detector.detect(df)

# Get results
if detector.detected:
    summary = detector.get_summary()
    print(f"Found {summary['count']} abnormal gaps")
    print(f"Analyzed {summary['details']['total_transitions']} transitions")
    
    # Visualize
    import plotly.graph_objects as go
    fig = go.Figure()  # Your plotly figure
    fig = detector.visualize(df, fig)
    fig.show()
```

### Advanced: Custom Threshold

```python
# More strict (fewer gaps detected)
detector.MIN_SAMPLES_FOR_NORMALITY = 10

# More lenient (more gaps detected)
detector.MIN_SAMPLES_FOR_NORMALITY = 3
```

---

## 📈 Output Format

### Summary Structure

```python
summary = {
    'pattern_type': 'gap',
    'detected': True,
    'count': 15,  # Number of abnormal gaps
    'details': {
        'total_gaps': 450,  # All gaps (including normal)
        'total_abnormal_gaps': 15,
        'total_transitions': 29,  # Unique transitions analyzed
        'transitions_with_anomalies': 8,  # Transitions with abnormal gaps
        'total_magnitude': 1065600.0,  # Total duration (seconds)
        'average_magnitude': 71040.0,  # Average duration (seconds)
        'abnormal_gaps': [...],  # List of gap dicts
        'transition_stats': {...}  # Per-transition statistics
    }
}
```

### Gap Object

```python
gap = {
    'case_id': '00000088',
    'transition': 'Order → Payment',
    'activity_from': 'Order',
    'activity_to': 'Payment',
    'duration': 1036800.0,  # seconds
    'threshold': 518400.0,  # seconds
    'severity': 2.0,  # duration / threshold
    'x_start': Timestamp('2005-04-01 01:00:00'),
    'x_end': Timestamp('2005-04-13 01:00:00'),
    'y_value_from': 'Resource A',
    'y_value_to': 'Resource A'
}
```

---

## 🐛 Troubleshooting

### "No abnormal gaps detected"

**Possible causes:**
1. **Not enough samples:** Transitions need ≥5 occurrences
   - Solution: Lower `MIN_SAMPLES_FOR_NORMALITY`
2. **No case_id column:** Gap detection requires cases
   - Solution: Ensure `case_id` column exists
3. **No activity column:** Can't identify transitions
   - Solution: Ensure `activity` column exists
4. **All gaps are normal:** No outliers in the data
   - Solution: This is actually good! Your process is stable

### Gaps look wrong in visualization

1. **Wrong Y-position:** Check `y_is_categorical` parameter
2. **Rectangles span multiple resources:** Expected for categorical Y (drawn at FROM-resource)
3. **Rectangles too small:** Zoom in on X-axis
4. **Too many rectangles:** Increase `MIN_SAMPLES_FOR_NORMALITY`

### Performance issues

**For large datasets (>100k events):**
```python
# Enable demo mode in UI (samples to 100 cases)
# Or manually sample:
from core.utils.demo_sampling import sample_small_eventlog
df_small = sample_small_eventlog(df, max_cases=100)
detector.detect(df_small)
```

---

## 📚 References

**Statistical Methods:**
- Tukey, J. W. (1977). Exploratory Data Analysis
- IQR-based outlier detection: Q3 + 1.5 × IQR

**Process Mining:**
- Rozinat, A., & van der Aalst, W. M. (2008). Conformance checking
- van der Aalst, W. M. (2016). Process Mining: Data Science in Action

---

## 🔄 Future Enhancements

**Possible improvements:**
1. **Multi-resource gaps:** Track resource handover delays
2. **Contextual factors:** Consider day-of-week, time-of-day
3. **Trend analysis:** Are gaps increasing over time?
4. **Root cause hints:** Suggest possible causes based on patterns
5. **Auto-threshold tuning:** ML-based threshold optimization

---

**Last updated:** November 2025  
**Maintained by:** @jennifoofoo  
**Implementation:** `core/detection/gap_pattern.py`  
**Status:** ✅ Production-ready

