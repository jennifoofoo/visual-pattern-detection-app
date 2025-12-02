# Outlier Detection Pattern Documentation

## Name
**Outlier Detection Pattern**

## Description
Automatically detects anomalous events, cases, and behaviors in process mining event logs across multiple dimensions. This pattern identifies unusual occurrences that deviate significantly from normal process execution patterns, helping analysts spot process deviations, data quality issues, and exceptional cases that require attention on the met level.


## Visual Representation
- **Markers**: Red circles with dark red borders overlaying the main dotted chart
- **Size**: Larger markers (size=10) to ensure visibility against background data
- **Color**: Semi-transparent red fill (`rgba(255, 0, 0, 0.2)`) with thick dark red borders
- **Legend**: Shows "Max Score Outliers (X)" where X is the confidence score
- **Annotation**: Statistics box displaying outlier counts and detection methods used
- **Hover Info**: Detailed explanations of why each event was flagged as an outlier, including:
  - Case ID and Activity name
  - Confidence score (e.g., "3/6" meaning detected by 3 out of 6 methods)
  - Specific detection reasons with user-friendly explanations

## Impossible Configurations + Explanation

❌ **None - This pattern works with any axis combination**

The outlier detection pattern is designed to be universally applicable. It adapts its detection methods based on available data columns and can identify anomalies regardless of the chosen X and Y axes. The pattern will highlight outliers within whatever view configuration is selected, making all axis combinations meaningful.

## Types of Outlier detection 
**View-Independent (always detected):**
- Case duration outliers
- Activity frequency outliers
- Resource workload outliers
- Case complexity outliers
- View-Dependent (change based on axes):

**Time-based outliers :**
- Position/sequence outliers (adapt to X-axis type)
- Workload visualization overlays (adapt to Y-axis type)

## Possible Configurations + Interpretation

### ✅ **All Axis Combinations Are Meaningful**

**Why**: Outliers represent deviations from normal patterns, which can be identified in any dimensional view:

| X-Axis | Y-Axis | Outlier Interpretation |
|--------|--------|----------------------|
| `actual_time` | `case_id` | Shows cases with unusual timing patterns or temporal anomalies |
| `actual_time` | `activity` | Reveals activities occurring at unusual times (off-hours, weekends) |
| `actual_time` | `resource` | Indicates resources working at atypical times |
| `relative_time` | `activity` | Shows activities happening at unusual points in case lifecycles |
| `relative_time` | `variant` | Highlights process variants with abnormal timing characteristics |
| `logical_time` | `case_id` | Reveals cases with unusual event sequences or frequencies |

**Key Principle**: The pattern detects multiple types of anomalies simultaneously and displays them in whatever view is currently selected, making all configurations useful for different analytical perspectives.

**Note**: Some patterns and configs are technically possible but do not make sense/give useful information - however, this does NOT apply to outlier detection, as anomalies can be meaningfully identified in any dimensional view of the data.

## Algorithm Explanation

The outlier detection employs **6 parallel detection methods** that run simultaneously and combine results using a multi-criteria approach:

### 1. **Time-Based Outliers** (O(n))
```python
# Identifies events occurring at unusual hours/days
df_time['hour'] = df_time[time_col].dt.hour
hour_counts = df_time['hour'].value_counts()
rare_threshold = max(1, hour_counts.quantile(0.05))  # Bottom 5%
rare_hours = hour_counts[hour_counts <= rare_threshold].index
```
- Extracts hour and day-of-week from timestamps
- Flags events occurring in the bottom 5% of temporal frequency distribution
- Detects off-hours work, weekend activities, or unusual scheduling patterns

### 2. **Case Duration Outliers** (O(n))
```python
# Uses extremely strict IQR method (3×IQR instead of standard 1.5×IQR)
Q1 = case_stats['duration_seconds'].quantile(0.25)
Q3 = case_stats['duration_seconds'].quantile(0.75)
IQR = Q3 - Q1
outliers = case_stats[
    (case_stats['duration_seconds'] < Q1 - 3.0 * IQR) |
    (case_stats['duration_seconds'] > Q3 + 3.0 * IQR)
]
```
- Calculates case start-to-end duration for each process instance
- Uses extremely strict IQR method (3×IQR) to find only the most extreme duration anomalies
- Identifies cases that complete too quickly or take exceptionally long

### 3. **Activity Frequency Outliers** (O(n))
```python
# Activities occurring less than 1% of total events
total_events = len(df)
rare_threshold = max(1, total_events * 0.01)
rare_activities = activity_counts[activity_counts < rare_threshold]
```
- Identifies activities that occur very infrequently (< 1% of total events)
- Flags rare or exceptional process steps that deviate from standard workflows

### 4. **Resource Behavior Outliers** (O(n))
```python
# Resource workload using 3×IQR method
resource_counts = df.groupby('resource').size()
Q1 = resource_counts.quantile(0.25)
Q3 = resource_counts.quantile(0.75)
IQR = Q3 - Q1
outliers = resource_counts[
    (resource_counts < Q1 - 3.0 * IQR) |
    (resource_counts > Q3 + 3.0 * IQR)
]
```
- Detects resources with extremely high or low workloads compared to peers
- Identifies overloaded resources or those with suspiciously low activity

### 5. **Sequence Outliers** (O(n²) worst case)
```python
# Finds rare activity transitions (bottom 1%)
transitions = {}
for case_id, activities in case_sequences.items():
    for i in range(len(activities) - 1):
        transition = (activities[i], activities[i+1])
        transitions[transition] = transitions.get(transition, 0) + 1

# Flag bottom 1% of all transition patterns
transition_counts = pd.Series(transitions)
rare_threshold = max(1, transition_counts.quantile(0.01))
```
- Builds activity transition frequency map across all cases
- Flags events involved in the bottom 1% of transition patterns
- Detects unusual workflow sequences that deviate from standard process flows

### 6. **Case Complexity Outliers** (O(n))
```python
# Multiple complexity metrics using IQR
for case_id in df['case_id'].unique():
    case_events = df[df['case_id'] == case_id]
    metrics = {
        'event_count': len(case_events),
        'unique_activities': len(case_events['activity'].unique()),
        'time_span_hours': (max_time - min_time).total_seconds() / 3600
    }
    # Apply 3×IQR filtering on each metric
```
- Analyzes multiple case complexity dimensions:
  - **Event Count**: Total number of events per case
  - **Activity Diversity**: Number of unique activities performed
  - **Time Span**: Total duration from case start to finish
- Uses 2-3×IQR thresholds depending on the metric
- Identifies cases that are unusually simple or complex

### How Does It Detect the Pattern?

**Multi-Criteria Scoring System:**
1. Each detection method runs independently and flags potential outliers
2. Events receive scores based on how many detection methods flagged them (1-6 points)
3. Only events with the **maximum score** are visualized to reduce noise
4. If more than 10% of events are flagged, additional filtering keeps only the most extreme outliers

**Adaptive Filtering:**
```python
# Safety mechanism - if too many outliers detected
if outlier_percentage > 10:
    self._filter_extreme_outliers()  # Keep only top 5% by score

# Only show maximum confidence outliers
max_score = max(self.outlier_scores.values())
max_score_indices = [idx for idx in all_outliers 
                    if self.outlier_scores.get(idx, 0) == max_score]
```

**Graceful Degradation:**
- Works with minimal columns (just case_id + activity)
- Automatically skips detection methods if required columns are missing
- Provides meaningful results even with incomplete data

**Output**: Events flagged by multiple detection methods with detailed explanations of why each event is considered anomalous, helping analysts focus on the most significant process deviations.