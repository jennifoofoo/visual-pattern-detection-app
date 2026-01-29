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

Or simply double-click `startApp.bat` (Windows)

The app will open in your browser at `http://localhost:8501`

## 📊 Usage

1. Load an event log (XES or CSV format)
2. Select sampling strategy (FULL, MINIMAL, SQRT, OPTIMIZED)
3. Choose view configuration (X-axis, Y-axis, Color)
4. Patterns are automatically detected and displayed in tabs
5. Toggle pattern overlays in the sidebar

## 📚 Documentation

Comprehensive guides available in [`docs/`](docs/):
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Gap Detection Guide](docs/GAP_DETECTION_GUIDE.md)
- [Temporal Patterns Guide](docs/TEMPORAL_PATTERNS_GUIDE.md)
- [Outlier Detection Guide](docs/OUTLIER_DETECTION_GUIDE.md)
- [Sequence Detection Guide](docs/SEQUENCE_DETECTION_GUIDE.md)
- [Pattern Matrix Structure](docs/PATTERN_MATRIX_STRUCTURE.md)

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

---

**Developed for Process Mining Praktikum**

