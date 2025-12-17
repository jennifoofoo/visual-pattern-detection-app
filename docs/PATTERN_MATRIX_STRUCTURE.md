# Pattern Matrix Structure

## Overview

The Extended Pattern Matrix defines which patterns are detectable and meaningful for each view configuration in the Dotted Chart visualization.

## Structure

### Key Format: Tuple-based (X, Y, Color)

The matrix uses **tuple keys** to explicitly represent all three dimensions of a Dotted Chart view:

```python
EXTENDED_PATTERN_MATRIX = {
    ("actual_time", "resource", "case_id"): {  # (X-axis, Y-axis, Color)
        "gap": {...},
        "temporal_cluster_x": {...},
        "outlier": {...}
    }
}
```

### Dimensions

1. **X-Axis** (5 options):
   - `actual_time`: Real timestamps
   - `relative_time`: Time in seconds (relative to case start)
   - `relative_ratio`: Normalized time [0,1] (relative to case duration)
   - `logical_time`: Global event counter
   - `logical_relative`: Event index within case

2. **Y-Axis** (4 options):
   - `resource`: Resource performing the activity
   - `activity`: Activity name
   - `case_id`: Case identifier
   - `event_index`: Event position within case

3. **Color/Dot Color** (5 options):
   - `case_id`: Color by case
   - `activity`: Color by activity
   - `resource`: Color by resource
   - `event_index_in_trace`: Color by event position
   - `timestamp_logical_global`: Color by global timestamp

### Pattern Types

Each view configuration can have 3 pattern types:

1. **gap**: Detects abnormal delays between consecutive activities
2. **temporal_cluster_x**: Detects time periods with high/low event concentration
3. **outlier**: Detects unusual events based on multiple anomaly types

## Pattern Metadata

Each pattern entry contains:

```python
{
    "can_be_found": bool,        # Technically detectable?
    "makes_sense": bool,         # Semantically meaningful?
    "visual": str,               # Visual representation (including color impact)
    "interpretation": str,       # What does it mean?
    "use_case": str,            # When to use it?
    "output": str               # What is returned?
}
```

## API Functions

### 1. `get_pattern_info(x_axis, y_axis, color, pattern_name)`

Get detailed information about a pattern for a specific view configuration.

```python
from config.extended_pattern_matrix import get_pattern_info

info = get_pattern_info('actual_time', 'resource', 'case_id', 'gap')
print(info['interpretation'])
print(info['visual'])
```

### 2. `is_pattern_meaningful(x_axis, y_axis, color, pattern_name)`

Check if a pattern is both technically possible AND semantically meaningful.

```python
from config.extended_pattern_matrix import is_pattern_meaningful

if is_pattern_meaningful('actual_time', 'resource', 'case_id', 'gap'):
    # Enable gap detection button
    pass
```

### 3. `get_meaningful_patterns(x_axis, y_axis, color)`

Get list of all meaningful patterns for a view configuration.

```python
from config.extended_pattern_matrix import get_meaningful_patterns

patterns = get_meaningful_patterns('actual_time', 'resource', 'case_id')
# Returns: ['gap', 'temporal_cluster_x', 'outlier']
```

### 4. `get_all_view_combinations()`

Get all defined view combinations in the matrix.

```python
from config.extended_pattern_matrix import get_all_view_combinations

views = get_all_view_combinations()
# Returns: [('actual_time', 'resource', 'case_id'), ...]
```

## Usage in Application

### In `app_handler.py`:

```python
# Get current view configuration
x_col = plot_config.get('x_col')
y_col = plot_config.get('y_col')
color_col = plot_config.get('dots_config_col', 'case_id')

# Check which patterns are meaningful
temporal_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'temporal_cluster_x')
outlier_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'outlier')
gap_meaningful = is_pattern_meaningful(x_col, y_col, color_col, 'gap')

# Get pattern info for tooltips
temporal_info = get_pattern_info(x_col, y_col, color_col, 'temporal_cluster_x')
outlier_info = get_pattern_info(x_col, y_col, color_col, 'outlier')
gap_info = get_pattern_info(x_col, y_col, color_col, 'gap')

# Use info to enable/disable buttons and show tooltips
if gap_meaningful:
    st.button("Detect Gaps", disabled=False)
    st.caption(gap_info['use_case'])
else:
    st.button("Detect Gaps", disabled=True)
    st.caption(f"Not available: {gap_info['interpretation']}")
```

### In Pattern Detectors (e.g., `outlier_detection.py`):

```python
def detect(self) -> bool:
    # Check if outlier detection is meaningful for this view
    x_axis = self.view_config.get('x', '')
    y_axis = self.view_config.get('y', '')
    color = self.view_config.get('color', 'case_id')
    
    if not is_pattern_meaningful(x_axis, y_axis, color, 'outlier'):
        return False
    
    # Proceed with detection...
```

## Current Coverage

As of now, the matrix defines **11 view combinations**:

1. `actual_time × resource × case_id`
2. `actual_time × resource × activity`
3. `actual_time × resource × resource`
4. `actual_time × activity × case_id`
5. `actual_time × activity × activity`
6. `actual_time × activity × resource`
7. `actual_time × case_id × case_id`
8. `actual_time × case_id × activity`
9. `actual_time × case_id × resource`
10. `logical_time × resource × case_id` (not meaningful)
11. `relative_ratio × resource × case_id` (partially meaningful)

**Total possible combinations:** 5 X-axes × 4 Y-axes × 5 Colors = **100 combinations**

**To be completed:** ~89 combinations

## Benefits of Tuple-based Structure

### Advantages:

1. **Explicit 3D Configuration**: Clear separation of X, Y, and Color dimensions
2. **Type-safe Keys**: Tuples prevent typos and enable IDE autocomplete
3. **Extensible**: Easy to add 4th dimension (e.g., size, shape) later
4. **No String Concatenation**: Avoids ambiguity (e.g., `"actual_time_resource_case_id"` vs `"actual_time_resource_case"` + `"_id"`)
5. **Color Impact in Visual**: The `visual` field now includes how color affects interpretation

### Previous Structure (String Keys):

```python
# OLD: String concatenation
"actual_time_resource": {  # Missing color dimension!
    "gap": {...}
}
```

**Problems:**
- Color dimension ignored
- Ambiguous keys (e.g., `"actual_time_event_index"` could be `actual_time_event` + `_index`)
- Hard to extend to 4th dimension

## Migration Notes

### Breaking Changes:

1. **Function Signatures Changed:**
   - `get_pattern_info(x, y, pattern)` → `get_pattern_info(x, y, color, pattern)`
   - `is_pattern_meaningful(x, y, pattern)` → `is_pattern_meaningful(x, y, color, pattern)`
   - `get_meaningful_patterns(x, y)` → `get_meaningful_patterns(x, y, color)`

2. **View Config Structure Changed:**
   ```python
   # OLD
   view_config = {'x_axis': 'actual_time', 'y_axis': 'resource'}
   
   # NEW
   view_config = {'x': 'actual_time', 'y': 'resource', 'color': 'case_id'}
   ```

3. **Matrix Keys Changed:**
   ```python
   # OLD
   EXTENDED_PATTERN_MATRIX["actual_time_resource"]
   
   # NEW
   EXTENDED_PATTERN_MATRIX[("actual_time", "resource", "case_id")]
   ```

## Future Work

### To Complete:

1. **Add remaining view combinations** (~89 combinations)
   - Focus on meaningful combinations first
   - Mark non-meaningful combinations with `can_be_found: False`

2. **Consider 4th dimension**
   - Size (e.g., case duration, event count)
   - Shape (e.g., event type, lifecycle transition)

## Examples

### Example 1: Gap Detection with Different Colors

```python
# Same X/Y, different colors → different interpretations

# Case coloring: See which cases have delays
get_pattern_info('actual_time', 'resource', 'case_id', 'gap')
# visual: "Red rectangles showing time spans of abnormal gaps between activities. 
#          Case coloring helps identify which specific cases experience delays at each resource."

# Activity coloring: See which activities cause delays
get_pattern_info('actual_time', 'resource', 'activity', 'gap')
# visual: "Red rectangles showing time spans of abnormal gaps between activities. 
#          Activity coloring reveals which specific activities are delayed at each resource."

# Resource coloring: Clear visual separation
get_pattern_info('actual_time', 'resource', 'resource', 'gap')
# visual: "Red rectangles showing time spans of abnormal gaps, colored by resource. 
#          Resource coloring provides redundant but clear visual separation of resources."
```

### Example 2: Non-meaningful Combinations

```python
# Logical time is not temporal → gap detection doesn't make sense
is_pattern_meaningful('logical_time', 'resource', 'case_id', 'gap')
# Returns: False

info = get_pattern_info('logical_time', 'resource', 'case_id', 'gap')
print(info['interpretation'])
# "Not meaningful: logical_time is a sequential counter, not actual time. Gap detection requires temporal data."
```

## References

- **Matrix File:** `config/extended_pattern_matrix.py`
- **Usage in App:** `core/app_utils/app_handler.py`
- **Usage in Patterns:** `core/detection/outlier_detection.py`, `core/detection/gap_pattern.py`
