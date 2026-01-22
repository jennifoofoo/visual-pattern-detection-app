# Outlier Detection Pattern Documentation

## Name
**Outlier Detection Pattern**

## Description
Automatically detects anomalous events in process mining event logs using machine learning (Isolation Forest algorithm). This pattern identifies unusual occurrences that deviate significantly from normal process execution patterns by analyzing multiple features simultaneously. It helps analysts spot process deviations, data quality issues, and exceptional cases that require attention, with human-readable explanations for each detected outlier.


## Visual Representation
- **Markers**: Red 'X' symbols overlaying the main visualization
- **Size**: 12pt markers with 2pt borders for visibility
- **Color**: Red with dark red borders
- **Legend**: Shows "Outliers (X)" where X is the number of outliers detected
- **Hover Info**: Detailed explanations for each outlier, including:
  - Case ID and Activity name
  - Anomaly Score (lower = more anomalous)
  - **Why it's an outlier**: Human-readable explanation highlighting which features are extreme (e.g., "Case has unusually many events (45) • Very late in timeline (127h from start)")
  - Specific reasons based on feature deviations (timing, case complexity, rare activities, etc.)

## Impossible Configurations + Explanation

❌ **None - This pattern works with any axis combination**

The outlier detection pattern is designed to be universally applicable. It adapts its detection methods based on available data columns and can identify anomalies regardless of the chosen X and Y axes. The pattern will highlight outliers within whatever view configuration is selected, making all axis combinations meaningful.

## Types of Outlier Detection

**Automatic Feature-Based Detection:**

The pattern automatically builds features from available data columns and uses Isolation Forest to detect anomalies across multiple dimensions simultaneously:

1. **Case-Level Patterns**
   - Events per case (case complexity)
   - Position within case (early/late event anomalies)

2. **Temporal Patterns**
   - Hour of day (off-hours work)
   - Day of week (weekend activities)
   - Time offset from process start

3. **Activity Patterns**
   - Activity frequency (rare vs. common activities)

4. **Resource Patterns**
   - Resource workload (over/under-utilized resources)

**Adaptive Detection:**
- Works with any subset of available columns
- Requires at least 2 features for detection
- Gracefully handles missing data

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

The outlier detection uses **Isolation Forest**, a state-of-the-art machine learning algorithm designed specifically for anomaly detection. It automatically analyzes multiple features simultaneously to identify events that deviate from normal patterns.

### Isolation Forest Algorithm

**Core Principle:** 
Anomalies are rare and different, making them easier to isolate than normal points. The algorithm builds random decision trees that partition the data, and outliers require fewer splits to be isolated.

**Algorithm Parameters:**
```python
IsolationForest(
    contamination=0.05,    # Expect ~5% outliers
    random_state=42,       # Reproducible results
    n_estimators=100,      # Number of trees
    max_samples='auto'     # Sample size per tree
)
```

### Feature Engineering Process

The algorithm automatically builds features from available columns:

#### 1. **Case-Level Features** (when case_id available)
```python
# Events per case - identifies unusually simple/complex cases
case_counts = df.groupby(case_col).size()
features.append(df[case_col].map(case_counts).values)

# Position in case - detects early/late event anomalies  
case_position = df.groupby(case_col).cumcount()
features.append(case_position.values)
```
- **Events per case**: Total events in each case (detects simple/complex cases)
- **Position in case**: Event sequence number within case (detects unusual positions)

#### 2. **Temporal Features** (when timestamp available)
```python
# Extract time components
hours = df[time_col].dt.hour.fillna(12).values
day_of_week = df[time_col].dt.dayofweek.fillna(2).values

# Calculate offset from process start
min_time = df[time_col].min()
hours_from_start = (df[time_col] - min_time).dt.total_seconds() / 3600
```
- **Hour**: Hour of day (0-23) - detects off-hours work
- **Day of week**: 0=Monday, 6=Sunday - detects weekend activities
- **Time offset**: Hours from process start - detects temporal outliers

#### 3. **Activity Features** (when activity column available)
```python
# Activity frequency - how common is this activity?
activity_freq = df[activity_col].value_counts()
features.append(df[activity_col].map(activity_freq).values)
```
- **Activity frequency**: Number of times this activity appears - detects rare activities

#### 4. **Resource Features** (when resource column available)
```python
# Resource workload - how busy is this resource?
resource_freq = df[resource_col].value_counts()
features.append(df[resource_col].map(resource_freq).values)
```
- **Resource workload**: Total events handled by resource - detects over/under-utilized resources

### Feature Normalization

```python
# Standardize features to zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- All features are standardized to ensure equal weight in anomaly detection
- Prevents features with larger scales from dominating the algorithm

### Outlier Detection Process

1. **Build Feature Matrix**: Construct features from available columns (minimum 2 required)
2. **Normalize Features**: Standardize to zero mean and unit variance
3. **Train Isolation Forest**: Build 100 random isolation trees
4. **Score Events**: Calculate anomaly scores for each event (lower = more anomalous)
5. **Identify Outliers**: Events with prediction = -1 are flagged as outliers
6. **Generate Explanations**: Analyze which features caused the anomaly

### Explanation Generation

For each outlier, the algorithm explains **why** it's anomalous:

```python
# Check each feature - if >2 std devs from mean, it's extreme
for feat_name, scaled_val, raw_val in zip(feature_names, scaled_features, raw_features):
    if abs(scaled_val) > 2.0:  # More than 2 standard deviations
        # Generate human-readable explanation
        if feat_name == 'events_per_case':
            reasons.append(f"Case has unusually many events ({int(raw_val)})")
```

**Explanation Categories:**
- **Case complexity**: "Case has unusually many events (45)" or "Case has very few events (2)"
- **Position anomalies**: "Very late position in case (#34)" or "Very early in case (#1)"
- **Timing issues**: "Off-hours timing (23:00)" or "Weekend activity (Sat)"
- **Rare activities**: "Rare activity (only 3 occurrences)"
- **Resource anomalies**: "Resource handles many events (234)" or "Resource handles few events (1)"
- **Timeline anomalies**: "Very late in timeline (127h from start)"

If no feature exceeds 2 standard deviations, the top 3 contributing features are combined:
- Example: "Combined pattern: 45 events in case • Position #34 in case • At 23:00"

### How Does It Detect the Pattern?

**Single Robust Method:**
- Uses Isolation Forest, a proven machine learning algorithm for anomaly detection
- No arbitrary thresholds or rules - learns what's normal from the data itself
- Handles multi-dimensional patterns that traditional statistical methods miss

**Adaptive to Data:**
```python
# Works with any subset of features
if case_col: 
    # Add case features
if time_col:
    # Add temporal features
if activity_col:
    # Add activity features
# ... gracefully degrades based on available columns
```

**Performance Characteristics:**
- **Time Complexity**: O(n log n) for training and scoring
- **Space Complexity**: O(n × f) where f = number of features (typically 3-7)
- **Scalability**: Efficiently handles thousands of events

**Safety Mechanisms:**
- Requires minimum 2 features to avoid false positives
- Limits visualization to top 500 outliers for performance
- Provides meaningful results even with incomplete data

**Output:** 
Events that deviate from normal patterns with:
- Anomaly scores (lower = more anomalous)
- Human-readable explanations of which features are extreme
- Visual highlighting in any view configuration