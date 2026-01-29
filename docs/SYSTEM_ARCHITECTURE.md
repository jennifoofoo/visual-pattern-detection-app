# System Architecture - Visual Pattern Detection App

## Overview
This is a **Streamlit-based web application** for detecting visual patterns in Process Mining event logs. The system uses a modular architecture with clear separation between the UI layer, business logic, pattern detection algorithms, and data processing.

---

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND LAYER (Streamlit)                         │
│                                  app.py                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────┐    ┌────────────────────┐    ┌────────────────┐  │
│  │  Main UI Components  │    │  Sidebar Controls  │    │ Chart Display  │  │
│  │  - File Input        │    │  - Layer Toggles   │    │ - Plotly Chart │  │
│  │  - Config Selectors  │    │  - AI Description  │    │ - Overlays     │  │
│  │  - Pattern Tabs      │    │  - Pattern Filters │    │ - Interactive  │  │
│  └──────────┬───────────┘    └────────┬───────────┘    └────────┬───────┘  │
│             │                         │                          │           │
│             └────────────┬────────────┴──────────────────────────┘           │
│                          │                                                   │
└──────────────────────────┼───────────────────────────────────────────────────┘
                           │
                           │ Function Calls
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    APPLICATION HANDLERS LAYER                                │
│                         core/app_utils/                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     app_handler.py                                  │    │
│  │  - init_state()              : Initialize session state             │    │
│  │  - load_data_button()        : Load XES/CSV files                  │    │
│  │  - plot_chart_button()       : Generate Plotly charts              │    │
│  │  - display_chart()           : Render chart with overlays          │    │
│  │  - auto_detect_patterns()    : Trigger all pattern detections      │    │
│  │  - sidebar_pattern_layer_controls() : Manage visibility toggles    │    │
│  └───────────────────────────┬────────────────────────────────────────┘    │
│                               │                                              │
│  ┌───────────────────────────┴────────────────────────────────────────┐    │
│  │          app_handler_pattern_detection.py                           │    │
│  │  - _detect_temporal_clusters() : Temporal cluster detection         │    │
│  │  - _detect_outliers()          : Outlier detection logic           │    │
│  │  - _detect_gaps()              : Gap detection logic               │    │
│  │  - _detect_sequences()         : Sequence detection logic          │    │
│  │  - _detect_clusters()          : DBSCAN/OPTICS clustering          │    │
│  │  - display_*_tab()             : Tab rendering for each pattern    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │         app_handler_pattern_filtering.py                            │    │
│  │  - list_to_multicheckbox()     : Filter UI components              │    │
│  │  - dict_to_multicheckbox()     : Multi-selection widgets           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    │ Calls
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CORE BUSINESS LOGIC LAYER                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │               DATA PROCESSING (core/data_processing/)                 │   │
│  │  ┌─────────────────────┐     ┌──────────────────────────────────┐   │   │
│  │  │   loader.py         │     │   preprocessor.py                 │   │   │
│  │  │  - load_xes_log()   │     │  - DataPreprocessor class         │   │   │
│  │  │  - load_csv_log()   │     │  - _encode_categoricals()        │   │   │
│  │  │  Time conversions:  │     │  - _normalize_numericals()       │   │   │
│  │  │    • actual_time    │     │  - fit_transform()               │   │   │
│  │  │    • relative_time  │     │  - transform()                   │   │   │
│  │  │    • relative_ratio │     │  Encoders: MinMaxScaler, etc.    │   │   │
│  │  │    • logical_time   │     └──────────────────────────────────┘   │   │
│  │  │    • logical_rel.   │                                             │   │
│  │  └─────────────────────┘                                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │          PATTERN DETECTION ENGINE (core/detection/)                   │   │
│  │                                                                        │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  pattern_base.py - Abstract Base Class                        │   │   │
│  │  │  class Pattern(ABC):                                          │   │   │
│  │  │    - __init__(name, view_config)                              │   │   │
│  │  │    - @abstractmethod detect(df) -> None                       │   │   │
│  │  │    - @abstractmethod visualize(df, fig) -> go.Figure          │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                │                                      │   │
│  │                                │ Inherits from Pattern                │   │
│  │                                ▼                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐  │   │
│  │  │ gap_pattern.py  │  │ outlier_detect  │  │ temporal_cluster   │  │   │
│  │  │                 │  │     ion.py      │  │       .py          │  │   │
│  │  │ GapPattern      │  │ OutlierPattern  │  │ TemporalCluster    │  │   │
│  │  │ - Transition-   │  │ - IQR Method    │  │ Pattern            │  │   │
│  │  │   aware gaps    │  │ - Z-score       │  │ - DBSCAN time-     │  │   │
│  │  │ - 1D & 2D gaps  │  │ - Isolation     │  │   based clusters   │  │   │
│  │  │ - Severity      │  │   Forest        │  │ - Density-based    │  │   │
│  │  │   scoring       │  │                 │  │   detection        │  │   │
│  │  └─────────────────┘  └─────────────────┘  └────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐  │   │
│  │  │cluster_pattern  │  │sequence_detection│ │  (Future patterns) │  │   │
│  │  │      .py        │  │       .py        │  │                    │  │   │
│  │  │ ClusterPattern  │  │HorizontalSeq...  │  │ - TrendPattern     │  │   │
│  │  │ - OPTICS        │  │ - Horizontal     │  │ - CorrelationPtn   │  │   │
│  │  │ - DBSCAN        │  │   sequences      │  │ - etc.             │  │   │
│  │  │ - Spatial       │  │ - Pattern        │  │                    │  │   │
│  │  │   clustering    │  │   recognition    │  │                    │  │   │
│  │  └─────────────────┘  └─────────────────┘  └────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │               VISUALIZATION (core/visualization/)                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  visualizer.py                                                 │  │   │
│  │  │  - plot_dotted_chart(df, x, y, color, title, labels)          │  │   │
│  │  │    Returns: Plotly go.Figure                                  │  │   │
│  │  │  - Uses Plotly Express for scatter plots                      │  │   │
│  │  │  - Integrates with Pattern.visualize() for overlays           │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │            SUMMARY GENERATION                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  summary_generator.py                                         │  │   │
│  │  │  - summarize_event_log(df) : Generate log statistics          │  │   │
│  │  │    • Number of cases/events                                   │  │   │
│  │  │    • Start/end activities                                     │  │   │
│  │  │    • Average duration                                         │  │   │
│  │  │    • Log date range                                           │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  UTILITIES (core/utils/)                            │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  demo_sampling.py                                             │  │   │
│  │  │  - sample_small_eventlog() : Sample for demo mode             │  │   │
│  │  │    • Limits to 100 cases                                      │  │   │
│  │  │    • Max events per case                                      │  │   │
│  │  │    • Fast gap detection                                       │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION & KNOWLEDGE BASE                               │
│                           config/                                               │
├────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │  extended_pattern_matrix.py                                           │     │
│  │                                                                        │     │
│  │  EXTENDED_PATTERN_MATRIX: Dict[(x, y, color), Dict[pattern, info]]   │     │
│  │                                                                        │     │
│  │  - is_pattern_meaningful(x, y, color, pattern_type) -> bool          │     │
│  │  - get_pattern_info(x, y, color, pattern_type) -> Dict               │     │
│  │                                                                        │     │
│  │  Matrix Structure:                                                    │     │
│  │    Key: (x_axis, y_axis, color)                                      │     │
│  │    Value: {                                                           │     │
│  │      "gap": {can_be_found, makes_sense, visual, interpretation...}   │     │
│  │      "temporal_cluster_x": {...}                                     │     │
│  │      "outlier": {...}                                                │     │
│  │      "cluster": {...}                                                │     │
│  │      "horizontal_sequence": {...}                                    │     │
│  │    }                                                                  │     │
│  │                                                                        │     │
│  │  Purpose: Determine which patterns make sense for each view config   │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │  mappings.py (in core/app_utils/)                                     │     │
│  │  - X_AXIS_COLUMN_MAP   : UI labels → column names                    │     │
│  │  - Y_AXIS_COLUMN_MAP   : UI labels → column names                    │     │
│  │  - DOTS_COLOR_MAP      : UI labels → column names                    │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                             DATA STORAGE LAYER                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────────────┐ │
│  │  data/              │  │  Streamlit Session  │  │  tests/               │ │
│  │  - Hospital_log.xes │  │  State (st.state)   │  │  - Test datasets      │ │
│  │  - Sepsis_Cases.xes │  │  - df               │  │  - Synthetic logs     │ │
│  │  - *.xes, *.csv     │  │  - fig              │  │  - Mock data          │ │
│  │                     │  │  - detectors        │  │                       │ │
│  │  PM4Py format       │  │  - view_config      │  │  Pytest fixtures      │ │
│  └─────────────────────┘  └─────────────────────┘  └───────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Breakdown

### 1. **Frontend Layer (Streamlit UI)**
**File**: [app.py](app.py)

**Responsibilities**:
- Entry point of the application
- Streamlit UI components (buttons, inputs, charts, tabs)
- Session state initialization
- User interaction handling

**Key Components**:
- **File Input**: XES/CSV file path input
- **Sampling Mode**: Toggle for variant-aware sampling (FULL, MINIMAL, SQRT, OPTIMIZED)
- **Chart Configuration**: X-axis, Y-axis, Color selectors with predefined view configs
- **Pattern Detection Section**: Tabs for each detected pattern
- **Sidebar**: Pattern layer visibility toggles, matrix viewer, focus view controls

**User Flow**:
1. User enters file path → Click "Load Data"
2. Select X/Y/Color axes → Click "Plot Chart"
3. Patterns auto-detected → Displayed in tabs
4. Toggle pattern overlays in sidebar

---

### 2. **Application Handlers Layer**
**Directory**: [core/app_utils/](core/app_utils/)

#### **app_handler.py**
Central orchestrator for all UI operations:
- `init_state()`: Initialize session state variables
- `load_data_button()`: Load XES/CSV using cached loader with sampling
- `get_chart_config_with_selectboxes()`: Generate axis selectors
- `plot_chart_button()`: Create Plotly chart and trigger auto-detection
- `display_chart()`: Persistent chart display with pattern overlays
- `sidebar_pattern_layer_controls()`: Manage visibility toggles

**Caching**:
- `@st.cache_data` for `load_xes_log()` (1 hour TTL)
- `@st.cache_data` for `generate_summary()` (1 hour TTL)

#### **pattern_detection.py**
Pattern detection orchestration:
- `auto_detect_patterns()`: Orchestrate all meaningful pattern detections
- `_detect_temporal_clusters()`: Temporal cluster detection
- `_detect_clusters()`: OPTICS/DBSCAN clustering
- `_detect_outliers()`: Outlier detection logic
- `_detect_gaps()`: Gap detection logic
- `_detect_case_arrival_trend()`: Case arrival trend analysis
- `_detect_sequences()`: Sequence detection (PrefixSpan)

#### **pattern_ui.py**
Pattern UI rendering:
- `display_pattern_detection_section()`: Main pattern UI orchestrator
- `_display_gap_tab()`: Gap pattern tab rendering
- `_display_temporal_cluster_tab()`: Temporal cluster tab
- `_display_outlier_tab()`: Outlier detection tab
- `_display_cluster_tab()`: Clustering tab
- `_display_sequence_tab()`: Sequence detection tab
- `_display_case_arrival_trend_tab()`: Trend analysis tab

#### **matrix_viewer.py**
- `display_matrix_viewer()`: Interactive pattern matrix UI
- View and explore pattern applicability for view configurations

---

### 3. **Core Business Logic Layer**

#### **Data Processing** ([core/data_processing/](core/data_processing/))

**loader.py**:
- `load_xes_log(xes_path)`: Load XES file using PM4Py
- Computes time representations:
  - `actual_time`: Original timestamps
  - `relative_time`: First event at time 0
  - `relative_ratio`: Normalized 0-1 range
  - `logical_time`: Sequential event numbering
  - `logical_relative`: Logical time with first event at 0
- Extracts: `case_id`, `activity`, `resource`

**preprocessor.py**:
- `DataPreprocessor` class for encoding and normalization
- `_encode_categoricals()`: Convert categorical → integer codes
- `_normalize_numericals()`: MinMaxScaler or StandardScaler
- `fit_transform()` / `transform()`: Scikit-learn style API

#### **Pattern Detection Engine** ([core/detection/](core/detection/))

**pattern_base.py**:
```python
class Pattern(ABC):
    def __init__(self, name: str, view_config: Dict[str, str])
    
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> None:
        """Analyze df and store results in self.detected"""
        
    @abstractmethod
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """Add pattern overlays to Plotly figure"""
```

**Concrete Pattern Detectors**:

1. **gap_pattern.py** - `GapPattern`
   - Detects abnormal gaps between consecutive events
   - **Two modes**:
     - Transition mode: Process-aware (Activity A → B)
     - Resource mode: Resource inactivity detection
   - Statistical thresholds: Q3 + 1.5*IQR, median + 2*MAD
   - Severity scoring: duration / threshold
   - Configurable min_samples for statistical validity

2. **outlier_detection.py** - `OutlierDetectionPattern`
   - Combined approach: Isolation Forest + Statistical methods
   - Feature-based detection (timing, complexity, workload)
   - Human-readable explanations for each outlier
   - Adaptive feature engineering based on available columns
   - Multi-dimensional outlier detection

3. **temporal_cluster.py** - `TemporalClusterPattern`
   - DBSCAN-based temporal clustering
   - Detects time periods with high event density
   - Dynamic hyperparameters based on data characteristics
   - X-axis temporal bursts and parallel case detection

4. **cluster_pattern.py** - `ClusterPattern`
   - OPTICS and DBSCAN spatial clustering
   - 2D clustering in X-Y space
   - Categorical encoding via TF-IDF + PCA
   - Dynamic min_samples calculation
   - Color-coded cluster visualization with noise handling

5. **sequence_detector.py** - `HorizontalSequencePatternDetector`
   - PrefixSpan algorithm for frequent subsequence mining
   - Strict vs. non-strict matching modes
   - Top-k filtering by support count
   - Pattern visualization with event highlighting
   - Configurable min_support and max_patterns

6. **case_arrival_trend_pattern.py** - `CaseArrivalTrendPattern`
   - Mann-Kendall test for trend significance
   - Sen's slope estimator for trend magnitude
   - Optional Prophet integration:
     - Weekly seasonality detection
     - Changepoint identification
     - Weekend effect analysis
   - Aggregates by case start time (not event time)

7. **trend_pattern.py** - `TrendPattern`
   - General event-level trend analysis
   - Analyzes all events over time (vs. case arrivals)

#### **Visualization** ([core/visualization/](core/visualization/))

**visualizer.py**:
- `plot_dotted_chart()`: Create Plotly scatter plot
- Returns `go.Figure` object
- Integrates with `Pattern.visualize()` for overlays
- Interactive hover tooltips

#### **AI Evaluation** ([core/evaluation/](core/evaluation/))


**summary_generator.py**:
- `summarize_event_log(df)`: Compute statistics
  - Number of cases/events
  - Most frequent activity
  - Start/end activities
  - Average case duration
  - Log date range
  - Variant distribution

#### **Utilities** ([core/utils/](core/utils/))

**demo_sampling.py**:
- `VariantAwareSampler` class: Intelligent trace-preserving sampling
- `sample_eventlog_variant_aware()`: Main sampling entry point
- **Sampling Strategies**:
  - **FULL**: No sampling (complete dataset)
  - **MINIMAL**: Aggressive sampling (1-2 traces/variant, max 5K events)
  - **SQRT**: Balanced sampling (√n traces for frequent variants)
  - **OPTIMIZED**: Gentle reduction (~70% data retained)
  - **LEGACY**: First-N cases (backward compatible)
- **Features**:
  - Trace-level sampling (preserves complete traces)
  - Variant-stratified (keeps rare variants)
  - Configurable caps (max events, traces, variants)
  - Statistical tracking (retention rates, reduction metrics)
- **Use Cases**: Fast demo mode, large log analysis, benchmarking

---

### 4. **Configuration & Knowledge Base**

**extended_pattern_matrix.py** ([config/](config/)):
- **Purpose**: Define which patterns are meaningful for each view configuration
- **Structure**: `Dict[(x_axis, y_axis, color), Dict[pattern_type, metadata]]`
- **Functions**:
  - `is_pattern_meaningful(x, y, color, pattern)`: Boolean check
  - `get_pattern_info(x, y, color, pattern)`: Metadata (visual, interpretation, use_case)
- **Pattern Types**: gap, temporal_cluster_x, outlier, cluster, horizontal_sequence

**mappings.py** ([core/app_utils/](core/app_utils/)):
- `X_AXIS_COLUMN_MAP`: UI labels → DataFrame columns
- `Y_AXIS_COLUMN_MAP`: UI labels → DataFrame columns
- `DOTS_COLOR_MAP`: UI labels → DataFrame columns
- `VIEW_CONFIGS`: Predefined view configurations (Resource Timeline, Case Progression, Activity Overview)

---

### 5. **Data Storage Layer**

**Event Logs** ([data/](data/)):
- XES format (PM4Py standard)
- CSV format (custom)
- Example datasets: Hospital_log.xes, Sepsis_Cases.xes

**Session State** (Streamlit):
- `st.session_state.df`: Loaded DataFrame
- `st.session_state.fig`: Current Plotly figure
- `st.session_state.view_config`: Current view configuration
- `st.session_state.*_detector`: Pattern detector instances
- `st.session_state.visible_*`: Pattern visibility flags

**Test Data** ([tests/](tests/)):
- Synthetic logs for unit tests
- Fixtures: `conftest.py`
- Test datasets for gap, outlier, sequence detection

---

## Data Flow Diagram

```
┌──────────────┐
│  User Input  │
│ (File Path)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  load_xes_log()      │ ← PM4Py
│  - Parse XES         │
│  - Compute times     │
│  - Extract columns   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Session State       │
│  st.session_state.df │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  User Selects Axes   │
│  (X, Y, Color)       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  plot_chart_button() │
│  - Create Plotly fig │
│  - Store config      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────┐
│  auto_detect_patterns()  │
│  - Check matrix          │
│  - Run detectors         │
└──────┬───────────────────┘
       │
       ├──────┬─────────┬──────────┬─────────┐
       ▼      ▼         ▼          ▼         ▼
   ┌─────┐ ┌────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │ Gap │ │Out │ │Temporal│ │Cluster │ │Sequence│
   │     │ │lier│ │Cluster │ │        │ │        │
   └──┬──┘ └──┬─┘ └───┬────┘ └───┬────┘ └───┬────┘
      │       │       │          │          │
      └───────┴───────┴──────────┴──────────┘
                      │
                      ▼
              ┌───────────────┐
              │  self.detected│ ← Stored in detector
              │  self.visualize() methods
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ display_chart()│
              │ - Recreate fig│
              │ - Add overlays│
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  st.plotly()  │ ← Rendered to user
              │  Pattern Tabs │
              └───────────────┘
```

---

## Pattern Detection Workflow

```
Chart Plotted
     │
     ▼
auto_detect_patterns(x, y, color, df)
     │
     ├─ Check extended_pattern_matrix: is_pattern_meaningful(x, y, color, "gap")?
     │  │
     │  ├─ YES → _detect_gaps(x, y, df)
     │  │         │
     │  │         ├─ GapPattern(view_config, y_is_categorical)
     │  │         ├─ detector.detect(df)
     │  │         │   │
     │  │         │   ├─ _extract_transition_gaps()
     │  │         │   ├─ _compute_transition_stats()
     │  │         │   ├─ _identify_abnormal_gaps()
     │  │         │   └─ Store in self.detected
     │  │         │
     │  │         └─ st.session_state['gap_detector'] = detector
     │  │
     │  └─ NO → Skip gap detection
     │
     ├─ Check: is_pattern_meaningful(x, y, color, "temporal_cluster_x")?
     │  │
     │  ├─ YES → _detect_temporal_clusters(x, y, df)
     │  │         │
     │  │         ├─ TemporalClusterPattern(df, x, y, min_cluster_size=10)
     │  │         ├─ detector.detect()
     │  │         │   │
     │  │         │   ├─ DBSCAN clustering on X-axis
     │  │         │   ├─ Identify temporal bursts
     │  │         │   └─ Store in self.detected
     │  │         │
     │  │         └─ st.session_state['temporal_clusters'] = detector
     │  │
     │  └─ NO → Skip temporal cluster detection
     │
     ├─ [Similar for outlier, cluster, sequence...]
     │
     └─ All detections complete
         │
         ▼
display_chart()
     │
     ├─ Recreate base Plotly chart
     │
     ├─ IF visible_gap AND gap_detector.detected:
     │    fig = gap_detector.visualize(df, fig)
     │         │
     │         └─ Add red rectangles for gaps
     │
     ├─ IF visible_temporal_cluster AND temporal_detected:
     │    fig = temporal_clusters.visualize(df, fig)
     │         │
     │         └─ Add colored circles for clusters
     │
     ├─ [Similar for other patterns...]
     │
     └─ st.plotly_chart(fig)
```

---

## Key Design Patterns

### 1. **Strategy Pattern** (Pattern Detection)
- Abstract base class: `Pattern`
- Concrete strategies: `GapPattern`, `OutlierPattern`, etc.
- Polymorphic `detect()` and `visualize()` methods

### 2. **Template Method Pattern** (Pattern Detection Flow)
- Base class defines algorithm skeleton
- Subclasses implement specific steps

### 3. **Facade Pattern** (App Handlers)
- `app_handler.py` provides simplified interface to complex subsystems
- Hides complexity of detection, visualization, and state management

### 4. **Observer Pattern** (Streamlit Session State)
- Session state acts as observable
- UI components react to state changes
- Pattern visibility toggles trigger reruns

### 5. **Factory Pattern** (Pattern Creation)
- Auto-detect logic creates appropriate pattern detectors
- Based on view configuration and matrix

---

## Technology Stack

| Layer                | Technology                          |
|----------------------|-------------------------------------|
| **Frontend**         | Streamlit 1.x                       |
| **Visualization**    | Plotly (plotly.express, go.Figure)  |
| **Data Processing**  | pandas, numpy                       |
| **Event Log Parsing**| PM4Py                               |
| **Clustering**       | scikit-learn (DBSCAN, OPTICS)       |
| **Outlier Detection**| scipy.stats, sklearn.ensemble       |
| **Testing**          | pytest                              |
| **Language**         | Python 3.9+                         |

---

## Session State Management

Streamlit uses session state (`st.session_state`) to persist data across reruns:

| Key                          | Type                  | Purpose                                    |
|------------------------------|-----------------------|--------------------------------------------|
| `df`                         | DataFrame             | Loaded event log data                      |
| `fig`                        | go.Figure             | Current Plotly chart                       |
| `view_config`                | Dict                  | Current view: {x, y, color}                |
| `current_plot_config`        | Dict                  | Plot metadata (axes, labels, df_selected)  |
| `data_loaded`                | bool                  | Whether data has been loaded               |
| `chart_plotted`              | bool                  | Whether chart has been plotted             |
| `gap_detector`               | GapPattern            | Gap detection instance                     |
| `temporal_clusters`          | TemporalClusterPtn    | Temporal cluster instance                  |
| `outlier_detector`           | OutlierPattern        | Outlier detection instance                 |
| `cluster_detector`           | ClusterPattern        | Cluster detection instance                 |
| `temporal_detected`          | bool                  | Whether temporal clusters found            |
| `outlier_detected`           | bool                  | Whether outliers found                     |
| `case_arrival_trend_detected`| bool                  | Whether case arrival trend detected        |
| `cluster_detected`           | bool                  | Whether clusters found                     |
| `sequence_detected`          | bool                  | Whether sequences detected                 |
| `visible_gap`                | bool                  | Gap layer visibility toggle                |
| `visible_outlier`            | bool                  | Outlier layer visibility toggle            |
| `visible_case_arrival_trend` | bool                  | Case arrival trend visibility              |
| `visible_sequence`           | bool                  | Sequence pattern visibility                |
| `visible_temporal_cluster`   | bool                  | Temporal cluster layer visibility          |
| `visible_cluster`            | bool                  | Cluster layer visibility                   |

---

## Performance Optimizations
Variant-Aware Sampling**: Four sampling strategies (FULL, MINIMAL, SQRT, OPTIMIZED) for scalable analysis
3. **Lazy Detection**: Patterns only detected when meaningful (matrix check)
4. **Detection Caching**: Cache key based on (x, y, color, df_len) prevents redundant detection
5. **Incremental Visualization**: Overlays added conditionally based on visibility flags
6. **Session State**: Avoid redundant computations across reruns
7. **Dynamic Hyperparameters**: Pattern detectors auto-tune based on dataset size
8. **PrefixSpan Guards**: Recursion limits and sequence truncation for large datasetmatrix check)
4. **Incremental Visualization**: Overlays added conditionally based on visibility flags
5. **Session State**: Avoid redundant computations across reruns

---

## Extension Points

To add a new pattern detector:

1. **Create detector class** in [core/detection/](core/detection/):
   ```python
   class MyNewPattern(Pattern):
       def detect(self, df):
           # Detection logic
           self.detected = {...}
       
       def visualize(self, df, fig):
           # Add overlays to fig
           return fig
   ```

2. **Update pattern matrix** in [config/extended_pattern_matrix.py](config/extended_pattern_matrix.py):
   ```python
   ("actual_time", "resource", "case_id"): {
       "my_new_pattern": {
           "can_be_found": True,
           "makes_sense": True,
           "visual": "...",
           "interpretation": "...",
           "use_case": "...",
           "output": "..."pattern_detection.py](core/app_utils/pattern_detection.py):
   ```python
   def _detect_my_new_pattern(x_col, y_col, df_selected):
       detector = MyNewPattern(view_config=...)
       detector.detect(df_selected)
       st.session_state.my_new_pattern_detector = detector
       st.session_state.my_new_pattern_detected = True
   ```

4. **Update auto-detect** in [core/app_utils/pattern_detection.py](core/app_utils/pattern_detection.py):
   ```python
   def auto_detect_patterns(...):
       if is_pattern_meaningful(x, y, color, 'my_new_pattern'):
           _detect_my_new_pattern(x_col, y_col, df_selected)
   ```

5. **Add UI tab** in [core/app_utils/pattern_ui.py](core/app_utils/pattern_ui.py):
   ```python
   def _display_my_new_pattern_tab():
       st.write("My New Pattern Results")
       # Render results with metrics and visualizations
   ```

6. **Register in pattern UI** in [core/app_utils/pattern_ui.py](core/app_utils/pattern_ui.py):
   ```python
   def display_pattern_detection_section():
       # Add tab to tab list
       if st.session_state.get('my_new_pattern_detected', False):
           with tabs[n]:
               _display_my_new_pattern_tab()
5. **Add UI tab** in [core/app_utils/app_handler_pattern_detection.py](core/app_utils/app_handler_pattern_detection.py):
   ```python
   def display_my_new_pattern_tab():
       st.write("My New Pattern Results")
       # Render results
   ```

---

## Testing Strategy

- **Unit Tests**: Individual pattern detectors ([tests/detection_tests/](tests/detection_tests/))
- **Integration Tests**: Data pipeline ([tests/data_processing_tests/](tests/data_processing_tests/))
- **Fixtures**: Synthetic logs for reproducible testing
- **Coverage**: Gap, outlier, sequence, temporal cluster detection

---

## Deployment

**Startup**:
```bash
# Windows
startApp.bat

# Manual
streamlit run app.py
```

**Requirements**: [requirements.txt](requirements.txt)
- streamlit
- pandas
- plotly
- pm4py
- scikit-learn
- scipy

---

## Summary

This system follows a **modular, extensible architecture** with clear separation of concerns:

- **Frontend (Streamlit)** handles UI and user interaction
- **App Handlers** orchestrate business logic and state management
- **Core modules** implement pattern detection, data processing, and visualization
- **Configuration** (pattern matrix) drives intelligent pattern detection
- **Session state** provides persistence across Streamlit reruns

The **Strategy pattern** for pattern detection allows easy addition of new detectors without modifying existing code. The **pattern matrix** acts as a knowledge base, ensuring only meaningful patterns are detected for each view configuration.
