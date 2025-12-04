# 🎨 Pattern-Implementierung für Tai
## Wie du dein Pattern wie Gap, Outlier und Temporal Cluster implementierst

---

## ✅ Einheitliche API

Alle Patterns (Gap, Outlier, Temporal Cluster) verwenden die **gleiche API**:

```python
class MyPattern(Pattern):
    def __init__(self, view_config: Dict[str, str], **kwargs):
        super().__init__("Pattern Name", view_config)
        self.detected = None
    
    def detect(self, df: pd.DataFrame) -> None:
        # Detection-Logik
        self.detected = {...}
    
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        # Visualisierungs-Logik
        return fig
```

**Wichtig:** Die Signatur von `visualize()` ist **immer gleich**:
- Parameter: `df: pd.DataFrame, fig: go.Figure`
- Return: `go.Figure`

---

## 📋 Schritt-für-Schritt Anleitung

### Schritt 1: Erstelle deine Pattern-Klasse

**Datei:** `core/detection/tai_pattern.py` (oder wie du sie nennst)

```python
"""
Tai's Pattern Detection for Dotted Charts.
"""

from .pattern_base import Pattern
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

class TaiPattern(Pattern):
    """
    Beschreibung deines Patterns.
    """
    
    def __init__(self, view_config: Dict[str, str], **kwargs):
        """
        Initialize pattern detector.
        
        Parameters
        ----------
        view_config : dict
            Configuration with "x" and "y" keys for chart dimensions
        **kwargs : dict
            Weitere Parameter für dein Pattern
        """
        super().__init__("Tai Pattern", view_config)
        self.detected = None
        # Weitere Initialisierung hier
    
    def detect(self, df: pd.DataFrame) -> None:
        """
        Detect patterns in the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        """
        if df.empty:
            self.detected = None
            return
        
        # Deine Detection-Logik hier
        # Beispiel:
        # patterns = []
        # for ... in df.iterrows():
        #     if ...:
        #         patterns.append({...})
        
        # Speichere Ergebnisse in self.detected
        self.detected = {
            'patterns': patterns,  # Deine erkannten Patterns
            'count': len(patterns),
            # Weitere Metadaten...
        }
    
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """
        Add visualization to the chart.
        
        ⚠️ WICHTIG: Modifiziere das ÜBERGEBENE fig, erstelle KEIN neues!
        - NIE: new_fig = go.Figure()  ❌
        - IMMER: fig.add_shape(...) oder fig.add_trace(...)  ✅
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log dataframe
        fig : go.Figure
            Plotly figure to annotate (DAS IST BEREITS DAS DOTTED CHART!)
            
        Returns
        -------
        go.Figure
            Figure with pattern visualization added (DAS GLEICHE fig!)
        """
        if self.detected is None:
            return fig  # ← Gleiche Figure zurückgeben
        
        # ✅ RICHTIG: Elemente zum bestehenden Figure hinzufügen
        # ❌ FALSCH: new_fig = go.Figure() erstellen!
        
        # Beispiel mit Shapes:
        for pattern in self.detected.get('patterns', []):
            fig.add_shape(  # ← Hinzufügen, nicht neu erstellen!
                type="rect",  # oder "line", "circle", etc.
                x0=pattern['x_start'],
                y0=pattern['y_low'],
                x1=pattern['x_end'],
                y1=pattern['y_high'],
                fillcolor="rgba(255, 0, 0, 0.25)",
                line=dict(color="red", width=2),
                layer="below"
            )
        
        # Oder mit Traces:
        # fig.add_trace(go.Scatter(...))  # ← Hinzufügen zum bestehenden Figure
        
        # Oder mit Annotations:
        # fig.add_annotation(...)  # ← Hinzufügen zum bestehenden Figure
        
        return fig  # ← IMMER das gleiche fig zurückgeben!
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected patterns.
        
        Returns
        -------
        dict
            Summary dictionary (wird in UI angezeigt)
        """
        if self.detected is None:
            return {
                'pattern_type': 'tai_pattern',
                'detected': False,
                'count': 0,
                'details': {}
            }
        
        return {
            'pattern_type': 'tai_pattern',
            'detected': True,
            'count': self.detected.get('count', 0),
            'details': {
                # Weitere Details...
            }
        }
```

### Schritt 2: Exportiere dein Pattern

**Datei:** `core/detection/__init__.py`

```python
from .tai_pattern import TaiPattern

__all__ = [
    'Pattern',
    'GapPattern',
    'OutlierDetectionPattern',
    'TemporalClusterPattern',
    'TaiPattern'  # ← Hinzufügen
]
```

### Schritt 3: Integriere in app_handler.py

**Datei:** `core/app_utils/app_handler.py`

**1. Import hinzufügen:**
```python
from core.detection import OutlierDetectionPattern, TemporalClusterPattern, TaiPattern
```

**2. Detection-Logik hinzufügen:**
```python
def handle_tai_pattern_detection_logic(x_col, y_col, x_axis_label, y_axis_label, df_selected):
    """Execute Tai pattern detection logic."""
    try:
        detector = TaiPattern(
            view_config={'x': x_col, 'y': y_col}
        )
        
        if detector.detect(df_selected):
            st.session_state.tai_pattern = detector
            st.session_state.tai_detected = True
            st.rerun()
        else:
            st.session_state.tai_detected = False
            st.info("No patterns detected!")
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
```

**3. UI-Button hinzufügen:**
In `handle_pattern_detection()`:
```python
# Prüfe ob Pattern für diese View sinnvoll ist
tai_meaningful = is_pattern_meaningful(x_col, y_col, 'tai_pattern')

# In der Pattern Detection Sektion:
with col4:  # Oder neue Spalte
    st.markdown("### 🎨 Tai Pattern")
    st.write("Beschreibung deines Patterns...")
    
    if st.button("Detect Tai Pattern", 
                 type="primary", 
                 disabled=not tai_meaningful,
                 help="..." if not tai_meaningful else "Detect Tai patterns"):
        handle_tai_pattern_detection_logic(x_col, y_col, x_axis_label, y_axis_label, df_selected)
```

**4. Visualisierung hinzufügen:**
In `display_chart()`:
```python
# Add Tai pattern visualization if detected AND layer is visible
if layer_visibility.get('tai_pattern', True):
    if st.session_state.get('tai_detected', False) and 'tai_pattern' in st.session_state:
        fig = st.session_state.tai_pattern.visualize(df_selected, fig)
```

**5. Summary hinzufügen:**
In der Pattern Summary Sektion:
```python
# === TAI PATTERN SUMMARY ===
with sum_col4:  # Oder neue Spalte
    if st.session_state.get('tai_detected', False) and 'tai_pattern' in st.session_state:
        detector = st.session_state.tai_pattern
        summary = detector.get_summary()
        
        with st.container(border=True):
            # Header mit layer toggle
            header_col1, header_col2 = st.columns([0.8, 0.2])
            with header_col1:
                st.markdown("### 🎨 Tai Pattern")
            with header_col2:
                # Layer visibility toggle (wie bei anderen Patterns)
                ...
            
            # Metrics und Details
            ...
```

---

## 💡 Beispiele aus bestehenden Patterns

### Gap Pattern (Beispiel)
```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    if self.detected is None:
        return fig
    
    for gap in self.detected['abnormal_gaps']:
        fig.add_shape(
            type="rect",
            x0=gap['x_start'],
            y0=gap['y_low'],
            x1=gap['x_end'],
            y1=gap['y_high'],
            fillcolor="rgba(255, 0, 0, 0.25)",
            line=dict(color="rgba(255, 0, 0, 0.6)", width=2),
            layer="below"
        )
    return fig
```

### Outlier Detection (Beispiel)
```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    if not self.detected:
        return fig
    
    outlier_data = self.df.loc[outlier_indices]
    
    fig.add_trace(go.Scatter(
        x=outlier_data['actual_time'],
        y=outlier_data['resource'],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Outliers',
        hovertemplate='...'
    ))
    return fig
```

### Temporal Cluster (Beispiel)
```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    if 'temporal_bursts' in self.clusters:
        for burst in self.clusters['temporal_bursts']:
            fig.add_shape(
                type="rect",
                x0=burst['start_time'],
                y0=y_min,
                x1=burst['end_time'],
                y1=y_max,
                fillcolor="rgba(0, 255, 0, 0.2)",
                layer="below"
            )
    return fig
```

---

## ✅ Checkliste

- [ ] Pattern-Klasse erstellt (erbt von `Pattern`)
- [ ] `__init__()` mit `super().__init__(name, view_config)`
- [ ] `detect(df)` implementiert → speichert in `self.detected`
- [ ] `visualize(df, fig)` implementiert → returnt `fig`
- [ ] `get_summary()` implementiert → returnt Dict
- [ ] In `core/detection/__init__.py` exportiert
- [ ] In `app_handler.py` importiert
- [ ] Detection-Logik in `handle_*_detection_logic()` Funktion
- [ ] UI-Button in `handle_pattern_detection()`
- [ ] Visualisierung in `display_chart()` hinzugefügt
- [ ] Summary in Pattern Summary Sektion hinzugefügt
- [ ] Layer Visibility Toggle implementiert
- [ ] Pattern Matrix Eintrag (wenn nötig)

---

## 📚 Verfügbare Plotly-Funktionen

Du kannst alles verwenden, was Plotly unterstützt:

**Shapes:**
```python
fig.add_shape(type="rect", x0=..., y0=..., x1=..., y1=..., ...)
fig.add_shape(type="line", x0=..., y0=..., x1=..., y1=..., ...)
fig.add_shape(type="circle", x0=..., y0=..., x1=..., y1=..., ...)
```

**Traces:**
```python
fig.add_trace(go.Scatter(...))
fig.add_trace(go.Bar(...))
```

**Annotations:**
```python
fig.add_annotation(x=..., y=..., text=..., ...)
```

**Layout:**
```python
fig.update_layout(...)
fig.update_traces(...)
```

---

## 🎯 Wichtig: Einheitliche API

**Alle Patterns müssen diese Signatur haben:**
```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    # df: Event Log DataFrame (bereits gefiltert)
    # fig: Plotly Figure (Dotted Chart)
    # Return: Modifizierte Figure
    return fig
```

**So werden sie aufgerufen:**
```python
fig = pattern.visualize(df_selected, fig)
```

---

## 💬 Fragen?

Schaue dir die bestehenden Patterns an:
- `core/detection/gap_pattern.py` - Process-aware Gap Detection
- `core/detection/outlier_detection.py` - Multi-dimensional Outliers
- `core/detection/temporal_cluster.py` - Temporal Clustering

Alle verwenden die **gleiche API**! 🚀

