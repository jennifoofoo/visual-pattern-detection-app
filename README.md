# Visual Pattern Detection for Process Mining

A Streamlit-based web application for automated detection and visualization of patterns in process mining event logs using dotted charts.

## 📋 What It Does

Analyzes event logs (XES/CSV) and automatically detects:
- **Gap Patterns** - Abnormal delays in process execution
- **Temporal Clusters** - Time-based event groupings
- **Outliers** - Anomalous events and cases
- **Sequence Patterns** - Frequent activity sequences (PrefixSpan)
- **Cluster Patterns** - Spatial groupings (OPTICS/DBSCAN)
- **Case Arrival Trends** - Workload patterns over time

## 🚀 Quick Start

### Installation

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate environment:**
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Run Application

```bash
streamlit run app.py
```

Or simply double-click `startApp.bat` or run `./startApp.bat` from repository root (Windows)

The app will open in your browser at `http://localhost:8501`

## 📊 Usage

1. Load an event log (XES or CSV format)
2. Select sampling strategy: Full, Minimal (about one trace per variant), Balanced (√n traces per variant; `SQRT` in the code), or Optimized (about 70% of the traces)
3. Choose view configuration (X-axis, Y-axis, Color)
4. Patterns are automatically detected and displayed in tabs
5. Toggle pattern overlays in the sidebar

## ⚙️ Detection Parameters

Default parameters of the detection methods and the rationale behind them. Their sensitivity is analyzed in the SoftwareX paper (Section 3.2); scripts and results are in [`benchmark/revision`](benchmark/revision).

### Case arrival trend analysis
- Case starts are aggregated into weekly intervals; trends are tested with Mann–Kendall and their magnitude estimated with Sen's slope.
- A trend is reported only if the test is significant (`p ≤ 0.05`) **and** the slope, expressed as a percentage of the mean interval count (`β%`), satisfies `|β%| ≥ 0.5%`.
- The effect-size threshold is heuristic: on large event logs the test flags even negligible drifts as significant, so significant trends below 0.5% are classified as *stable*, reducing false alarms due to small random oscillations.
- No correction for seasonality or autocorrelation is applied, and the test assumes independent observations: detected trends are descriptive indicators, not formal inference.

### Temporal clustering (DBSCAN)
- `ε` is expressed in seconds and recomputed on the events currently in view, so it follows the analyzed time window when zooming or filtering.
- **Absolute-time views:** `ε = 0.02 · (t_max − t_min)`, i.e., a burst is a concentration of events occupying at most a few percent of the visible window.
- **Relative-time views:** `ε` is the median of three estimates, capped at `max(10% of the time range, 2 h)` to prevent excessive cluster merging:
  - dispersion-based: `0.5 · σ` of the timestamps;
  - gap-based: `5 ×` the 75th percentile of consecutive time differences;
  - range-based: `3%` of the time range.
- `m_min = max(5, min(⌊√n / 3⌋, max(100, ⌊0.01 n⌋)))`, with `n` the number of events.

### Outlier detection (Isolation Forest)
- The contamination parameter `α = 0.05` only sets the decision threshold on the anomaly-score distribution. It is not an estimate of the anomaly rate but a deliberately permissive upper bound on the share of candidate anomalies, consistent with anomalies being rare in event logs, chosen to limit false negatives in this first screening stage.
- Only outliers explained by multiple features (typically at least two features deviating by more than two standard deviations) are reported, capped at the 500 highest-scoring ones.

## 📚 Documentation

Comprehensive guides available in [`docs/`](docs/):
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Gap Detection Guide](docs/GAP_DETECTION_GUIDE.md)
- [Gap and Trend Pattern Methods](docs/GAP_AND_TREND_PATTERN_METHODS.md)
- [Temporal Patterns Guide](docs/TEMPORAL_PATTERNS_GUIDE.md)
- [Clustering Guide](docs/CLUSTERING_GUIDE.md)
- [Outlier Detection Guide](docs/OUTLIER_DETECTION_GUIDE.md)
- [Sequence Detection Guide](docs/SEQUENCE_DETECTION_GUIDE.md)
- [Pattern Matrix Structure](docs/PATTERN_MATRIX_STRUCTURE.md) (full matrix of the 45 view configurations)
- [Quantitative Evaluation](benchmark/revision/README.md) (benchmark and sensitivity analysis of the SoftwareX paper)

## 📄 License

Released under the Apache License 2.0 — see [`LICENCE.txt`](LICENCE.txt).

## 🧪 Testing

Run test suite:
```bash
pytest tests/
```

## 📦 Sample Data

Example event logs included in `data/`:
- `Hospital_log.xes` - Hospital process data
- `Sepsis_Cases.xes` - Sepsis treatment process

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly
- **Event Log Processing:** PM4Py
- **Pattern Detection:** scikit-learn, PrefixSpan, Prophet (optional)
- **Data Processing:** pandas, numpy


