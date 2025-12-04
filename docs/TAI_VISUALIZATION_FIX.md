# ❌ Was Tai falsch macht (und wie es richtig geht)

---

## 🚨 Das Problem

Tai erstellt wahrscheinlich ein **neues Figure** in seiner `visualize()` Methode, anstatt das **übergebene Figure zu modifizieren**.

---

## ❌ FALSCH (was Tai vermutlich macht):

```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    # ❌ FALSCH: Neues Figure erstellen
    new_fig = go.Figure()  # ← Das ist falsch!
    
    # ❌ FALSCH: Neuen Plot erstellen
    new_fig.add_trace(go.Scatter(
        x=df['actual_time'],
        y=df['resource'],
        mode='markers'
    ))
    
    return new_fig  # ← Gibt neues Figure zurück, nicht das übergebene!
```

**Problem:** 
- Erstellt einen komplett neuen, leeren Plot
- Das ursprüngliche Dotted Chart geht verloren
- Andere Patterns (Gap, Outlier) werden nicht angezeigt

---

## ✅ RICHTIG (wie alle anderen Patterns es machen):

```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    # ✅ RICHTIG: Das übergebene Figure verwenden
    # KEIN neues Figure erstellen!
    
    if self.detected is None:
        return fig  # ← Gleiche Figure zurückgeben
    
    # ✅ RICHTIG: Elemente zum bestehenden Figure hinzufügen
    fig.add_shape(
        type="rect",
        x0=...,
        y0=...,
        x1=...,
        y1=...,
        fillcolor="rgba(255, 0, 0, 0.25)",
        layer="below"
    )
    
    # Oder Traces hinzufügen:
    fig.add_trace(go.Scatter(
        x=...,
        y=...,
        mode='markers',
        name='My Pattern'
    ))
    
    return fig  # ← DAS GLEICHE Figure zurückgeben!
```

**Wichtig:** 
- **NIE** `go.Figure()` aufrufen
- **IMMER** das übergebene `fig` verwenden
- **IMMER** das gleiche `fig` zurückgeben

---

## 📚 Beispiele aus bestehenden Patterns

### Gap Pattern (richtig):
```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    if self.detected is None:
        return fig  # ← Gleiche Figure zurückgeben
    
    # Elemente zum bestehenden Figure hinzufügen
    for gap in self.detected['abnormal_gaps']:
        fig.add_shape(  # ← Hinzufügen, nicht neu erstellen!
            type="rect",
            x0=gap['x_start'],
            y0=gap['y_low'],
            x1=gap['x_end'],
            y1=gap['y_high'],
            fillcolor="rgba(255, 0, 0, 0.25)",
            layer="below"
        )
    
    return fig  # ← Gleiche Figure zurückgeben
```

### Outlier Detection (richtig):
```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    if not self.detected:
        return fig  # ← Gleiche Figure zurückgeben
    
    # Trace zum bestehenden Figure hinzufügen
    fig.add_trace(go.Scatter(  # ← Hinzufügen, nicht neu erstellen!
        x=outlier_data['actual_time'],
        y=outlier_data['resource'],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Outliers'
    ))
    
    return fig  # ← Gleiche Figure zurückgeben
```

---

## 🔍 Wie prüfen ob es richtig ist?

**Richtig:**
```python
def visualize(self, df, fig):
    # ✅ Kein "go.Figure()" Aufruf
    # ✅ Verwendet das übergebene "fig"
    fig.add_shape(...)  # oder fig.add_trace(...)
    return fig  # ← Gleiche Variable zurückgeben
```

**Falsch:**
```python
def visualize(self, df, fig):
    # ❌ "go.Figure()" wird aufgerufen
    new_fig = go.Figure()  # ← FALSCH!
    new_fig.add_trace(...)
    return new_fig  # ← Andere Variable zurückgeben
```

---

## 💡 Zusammenfassung

**Die Regel:**
1. **NIE** `go.Figure()` in `visualize()` aufrufen
2. **IMMER** das übergebene `fig` verwenden
3. **IMMER** `fig.add_shape()`, `fig.add_trace()`, etc. verwenden
4. **IMMER** das gleiche `fig` zurückgeben

**Das übergebene `fig` ist bereits das vollständige Dotted Chart!**
Du musst nur deine Visualisierung **darauf** hinzufügen, nicht ein neues Chart erstellen!

---

## 🎯 Korrekte Template für Tai:

```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    """
    Add visualization to EXISTING figure.
    
    WICHTIG: Modifiziere das übergebene fig, erstelle KEIN neues!
    """
    if self.detected is None:
        return fig  # ← Gleiche Figure zurückgeben
    
    # Deine Visualisierung zum bestehenden Figure hinzufügen:
    
    # Option 1: Shapes hinzufügen
    fig.add_shape(
        type="rect",  # oder "line", "circle"
        x0=...,
        y0=...,
        x1=...,
        y1=...,
        fillcolor="rgba(255, 0, 0, 0.25)",
        layer="below"
    )
    
    # Option 2: Traces hinzufügen
    fig.add_trace(go.Scatter(
        x=...,
        y=...,
        mode='markers',
        name='My Pattern'
    ))
    
    # Option 3: Annotations hinzufügen
    fig.add_annotation(
        x=...,
        y=...,
        text="My Annotation"
    )
    
    return fig  # ← IMMER das gleiche fig zurückgeben!
```

---

**Das ist der häufigste Fehler!** 🎯

