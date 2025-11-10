# Temporal Cluster Pattern Detection Guide

This guide explains which temporal cluster patterns are **meaningful** for each combination of X and Y axes in your dotted chart visualization.

## Available Axes

### X-Axis (Time Dimensions)
- **actual_time**: Real timestamps from the event log
- **relative_time**: Seconds elapsed since case started
- **relative_ratio**: Normalized time [0,1] within each case
- **logical_time**: Global sequential order (event #1, #2, #3... across all cases)
- **logical_relative**: Event position within trace (0, 1, 2...)

### Y-Axis (Grouping Dimensions)
- **case_id**: Individual case identifier
- **activity**: Activity/task name
- **event_index**: Position/sequence number of event
- **resource**: Person/department/system performing the activity
- **variant**: Process variant (activity sequence pattern)

---

## Meaningful Temporal Patterns by Axis Combination

### 1. **Temporal Bursts** 📊
**Meaningful for**: `actual_time × {activity, resource, case_id}`

**What it detects**: Periods where many events happen in a short time window

**Business Examples**:
- **actual_time × activity**: "Approval" activities all clustered on Friday afternoons
- **actual_time × resource**: "Radiology Lab" processes 50 events between 9-10 AM
- **actual_time × case_id**: 20 new cases started simultaneously (batch intake)

**Use Cases**:
- Identify batch processing patterns
- Detect shift changes
- Find system load peaks
- Discover scheduling patterns

**How it works**: Uses DBSCAN clustering on the time axis to find dense temporal regions

---

### 2. **Activity-Time Clustering** 🎯
**Meaningful for**: `{actual_time, relative_time, relative_ratio} × activity`

**What it detects**: When specific activities consistently occur at certain times

**Business Examples**:
- **actual_time × activity**: "Lab Tests" always happen 8-11 AM (hospital schedules)
- **relative_time × activity**: "Quality Check" occurs 2-4 hours after case start
- **relative_ratio × activity**: "Final Approval" always at 0.8-0.9 of case duration

**Use Cases**:
- Understand activity timing patterns
- Identify process bottlenecks
- Validate SLA compliance (activities at expected times)
- Detect scheduling rules

**How it works**: For each activity type, clusters events along the time axis

---

### 3. **Case Parallelism** ⏱️
**Meaningful for**: `{actual_time, relative_time} × case_id`

**What it detects**: How many cases run simultaneously (concurrent execution)

**Business Examples**:
- **actual_time × case_id**: Max 15 cases running at same time (capacity limits)
- **relative_time × case_id**: High overlap indicates parallel processing capability

**Use Cases**:
- Measure process capacity
- Identify resource constraints
- Understand workload distribution
- Detect batching vs continuous flow

**How it works**: Calculates case start/end times and tracks overlaps

---

### 4. **Resource Time Patterns** 
**Meaningful for**: `{actual_time, relative_time} × resource`

**What it detects**: When resources work (shift patterns, availability)

**Business Examples**:
- **actual_time × resource**: "Dr. Smith" works 9-17h, "Night Nurse" works 22-6h
- **relative_time × resource**: "Senior Analyst" only handles cases after 3 days

**Use Cases**:
- Identify shift schedules
- Detect resource availability patterns
- Find understaffing periods
- Understand handover patterns

**How it works**: Clusters each resource's activities along time to find distinct work periods

---

### 5. **Variant Timing Patterns** 🔄
**Meaningful for**: `{relative_time, relative_ratio} × variant`

**What it detects**: If different process paths have different timing characteristics

**Business Examples**:
- **relative_ratio × variant**: "Fast-track" variant completes at 0.2, "Complex" at 1.0
- **relative_time × variant**: Variant A takes 2 hours, Variant B takes 20 hours

**Use Cases**:
- Compare process variant performance
- Identify fast vs slow paths
- Optimize process routing
- Detect variant-specific bottlenecks

**How it works**: Compares timing distributions across different process variants

---

## Non-Meaningful Combinations ❌

### Why some combinations don't make sense:

**logical_time × anything**:
- Logical time is just a sequential counter - clustering it doesn't reveal meaningful patterns
- Already shows pure sequential order

**logical_relative × anything**:
- Just event position in trace (0, 1, 2...) - no temporal meaning
- Better for sequence analysis, not temporal clustering

**event_index × anything**:
- Similar to logical_relative - shows position, not timing
- Use for control-flow analysis instead

**relative_ratio × {case_id, resource}**:
- Normalized time [0,1] isn't meaningful across different cases
- Each case has its own [0,1] scale


Combinations with 
- logical_time, 
- logical_relative, 
- event_index 
are NOT meaningful for temporal clustering because they're sequential counters, not actual time measurements.
---


## Recommended Axis Combinations

### For **Temporal Analysis** (when did things happen?):
1. `actual_time × activity` → Activity-Time Clustering
2. `actual_time × resource` → Resource Time Patterns + Temporal Bursts
3. `actual_time × case_id` → Case Parallelism + Temporal Bursts

### For **Within-Case Analysis** (how does each case unfold?):
1. `relative_time × activity` → Activity-Time Clustering
2. `relative_ratio × activity` → Activity-Time Clustering
3. `relative_ratio × variant` → Variant Timing Patterns

### For **Resource Analysis**:
1. `actual_time × resource` → Resource Time Patterns
2. `relative_time × resource` → Resource involvement timing

### For **Process Variant Analysis**:
1. `relative_ratio × variant` → Variant Timing Patterns
2. `relative_time × variant` → Variant duration comparison

---

## Implementation Notes

### Auto-Detection Logic
The `TemporalClusterPattern` class automatically:
- Checks if the current axis combination is meaningful
- Selects appropriate detection algorithms
- Only runs relevant pattern detections
- Skips meaningless combinations

### Performance Considerations
- DBSCAN complexity: O(n log n) to O(n²) depending on data
- Recommended dataset size: < 100,000 events for real-time detection
- Use sampling for larger datasets

### Parameter Tuning
- `min_cluster_size`: Minimum events to form a cluster (default: 5)
  - Increase for larger datasets
  - Decrease for sparse processes
  
- `temporal_eps`: Distance threshold for clustering
  - Auto-calculated based on data range
  - Manual override for domain-specific requirements

---

