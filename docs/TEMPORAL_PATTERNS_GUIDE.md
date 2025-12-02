# Temporal Cluster Pattern Documentation

## Name
**Temporal Cluster Pattern**

## Description
Detects dense clusters of events occurring close together in time or space within process mining event logs. This pattern reveals periods of intense activity (temporal bursts), synchronized processes, resource shift patterns, and workflow bottlenecks. The detection system adapts to different axis combinations to identify meaningful temporal relationships in the data.

The pattern uses DBSCAN clustering to find regions where events are densely packed in temporal or spatial dimensions, helping analysts understand when and where process activities concentrate.

## Visual Representation
- **Markers**: Colored circles with distinct colors for each cluster
- **Size**: Medium markers (size=6-8) to show cluster membership clearly
- **Color Scheme**: Automatic color assignment using plotly's color sequence
- **Cluster Boundaries**: Optional convex hull outlines around cluster regions
- **Legend**: Shows "Temporal Cluster X" where X is the cluster number
- **Annotation**: Statistics showing cluster count, largest cluster size, and coverage percentage
- **Hover Info**: Cluster membership details and temporal characteristics

## Impossible Configurations + Explanation

❌ **Sequential/Ordinal Axes Are Not Meaningful**

| X-Axis | Y-Axis | Why Impossible | Explanation |
|--------|--------|---------------|-------------|
| `logical_time` | Non-Activity | Sequential counter | Logical time is just event ordering (1,2,3...) - clustering sequential numbers reveals no meaningful temporal patterns |
| `logical_relative` | Non-Activity | Position counter | Just event position within traces (0,1,2...) - no actual time information |
| `event_index` | Any | Index counter | Similar to logical_relative - shows sequence position, not timing |
| `relative_ratio` | `case_id` | Cross-case normalization | Normalized time [0,1] loses meaning across different cases with different durations |
| `relative_ratio` | `resource` | Cross-case normalization | Resource activities span multiple cases - [0,1] ratios are not comparable |
| `actual_time` | `activity` | Visually Meaningless | For activity bursts visually meaningless |
| `actual_time` | `resource` |Visually Meaningless | For resource bursts visually meaningless |

**Core Issue**: These axes represent ordinal sequences or normalized values that don't preserve actual temporal relationships needed for meaningful clustering.

## Possible Configurations + Interpretation

### ✅ **Temporal Dimensions (Actual Time-based)**

| X-Axis | Y-Axis | Pattern Detected | Interpretation |
|--------|--------|-----------------|----------------|
| `actual_time` | `activity` | **Activity-Time Clustering** | Activities that consistently occur at specific times (daily patterns, shift changes) |
| `actual_time` | `resource` | **Resource Time Patterns** | Work shift detection, resource availability periods, overtime patterns |
| `actual_time` | `case_id` | **Temporal Bursts** | Periods of intense case activity, batch processing, system load peaks |
| `actual_time` | `variant` | **Variant Time Clustering** | Process variants that execute during specific time periods |

### ✅ **Relative Time Dimensions (Within-case)**

| X-Axis | Y-Axis | Pattern Detected | Interpretation |
|--------|--------|-----------------|----------------|
| `relative_time` | `activity` | **Activity-Time Clustering** | Activities that occur at similar points in case lifecycles |
| `relative_time` | `resource` | **Resource Involvement Timing** | When specific resources typically engage in cases |
| `relative_time` | `case_id` | **Case Parallelism** | Concurrent case execution patterns and overlaps |
| `relative_time` | `variant` | **Variant Duration Patterns** | Process variants with similar timing characteristics |

### ✅ **Within-Case Analysis (Normalized Time)**

| X-Axis | Y-Axis | Pattern Detected | Interpretation |
|--------|--------|-----------------|----------------|
| `relative_ratio` | `activity` | **Activity Position Clustering** | Activities that consistently occur at similar case completion percentages |
| `relative_ratio` | `variant` | **Variant Timing Patterns** | Process variants with similar relative timing structures |

**Key Principle**: Only use time axes that preserve meaningful temporal relationships - either absolute timestamps or relative durations within cases.

## Algorithm Explanation

The temporal cluster detection employs **adaptive DBSCAN clustering** with automatic parameter tuning:

### 1. **Temporal Burst Detection** (O(n log n) to O(n²))
```python
# Automatically calculate optimal epsilon based on data distribution
def _auto_calculate_temporal_eps(self):
    time_diffs = np.diff(sorted_timestamps)
    # Use 95th percentile of time differences
    return np.percentile(time_diffs, 95)

# Apply DBSCAN clustering
clustering = DBSCAN(eps=temporal_eps, min_samples=min_cluster_size)
clusters = clustering.fit_predict(X.reshape(-1, 1))
```
- Converts timestamps to numerical format (seconds since epoch)
- Calculates optimal epsilon as 95th percentile of adjacent time differences
- Groups events that occur within `temporal_eps` seconds of each other
- Filters clusters smaller than `min_cluster_size` events

### 2. **Activity-Time Clustering** (O(k × n log n))
```python
# Cluster each activity type separately
for activity in df['activity'].unique():
    activity_events = df[df['activity'] == activity]
    activity_clusters = DBSCAN(eps=eps, min_samples=min_size)
    cluster_labels = activity_clusters.fit_predict(
        activity_events[time_column].values.reshape(-1, 1)
    )
```
- Groups events by activity type first
- Applies temporal clustering within each activity group
- Identifies when specific activities tend to occur together in time

### 3. **Resource Pattern Detection** (O(r × n log n))
```python
# Find distinct work periods for each resource
for resource in df['resource'].unique():
    resource_events = df[df['resource'] == resource]
    # Cluster resource activities to find work shifts/periods
    work_periods = DBSCAN(eps=shift_eps, min_samples=min_shift_size)
```
- Analyzes each resource's activity patterns separately  
- Detects distinct work periods, shifts, or availability windows
- Uses larger epsilon values to capture shift-level patterns

### 4. **Adaptive Parameter Calculation**
```python
def _calculate_optimal_eps(self, time_values):
    # Sort time values and calculate differences
    sorted_times = np.sort(time_values)
    time_diffs = np.diff(sorted_times)
    
    # Remove outliers and calculate percentile
    filtered_diffs = time_diffs[time_diffs < np.percentile(time_diffs, 99)]
    return np.percentile(filtered_diffs, 95)
```
- Dynamically calculates clustering parameters based on data characteristics
- Uses percentile-based approach to handle outliers
- Adapts to different data densities and time scales

### How Does It Detect the Pattern?

**Multi-Stage Process:**
1. **Axis Validation**: Checks if the current X/Y combination is meaningful for temporal clustering
2. **Parameter Auto-tuning**: Calculates optimal DBSCAN parameters based on data distribution
3. **Pattern-Specific Clustering**: Applies different clustering strategies based on the axis combination:
   - **Temporal Bursts**: 1D clustering on time axis
   - **Activity Patterns**: Per-activity temporal clustering
   - **Resource Patterns**: Per-resource temporal clustering with shift detection
4. **Cluster Validation**: Filters out noise and validates cluster quality
5. **Visualization**: Colors clusters and provides interpretive statistics

**Output**: Groups of events that occur close together in time, with automatic parameter tuning ensuring meaningful cluster detection across different data scales and patterns.



