# Visualization API Guide

## 📋 Overview

The visualization system supports two ways to add custom visualizations:

1. **Pattern API** (Full Pattern Classes) - For complete pattern detection + visualization
2. **Registry API** (Simple Callbacks) - For quick custom visualizations without full Pattern classes

---

## 🎯 Quick Start: Registry API (Recommended for Simple Visualizations)

### Simple Example

```python
from core.visualization.registry import register_visualization
import plotly.graph_objects as go

def my_custom_viz(df, fig):
    """Add custom visualization to chart."""
    # Your visualization code here
    fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3], name="Custom"))
    return fig

# Register it
register_visualization("my_pattern", my_custom_viz)
```

That's it! Your visualization will automatically appear on all charts.

---

## 📚 Full Pattern API

### Standard Pattern Class

All patterns must implement:

```python
from core.detection.pattern_base import Pattern
import pandas as pd
import plotly.graph_objects as go

class MyPattern(Pattern):
    def __init__(self, view_config: Dict[str, str], **kwargs):
        super().__init__("My Pattern", view_config)
        self.detected = None
    
    def detect(self, df: pd.DataFrame) -> None:
        """Detect patterns in data."""
        # Your detection logic
        self.detected = {...}
    
    def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """Add visualization to chart."""
        if self.detected is None:
            return fig
        
        # Your visualization code
        fig.add_trace(...)
        return fig
```

### Standard Signature

**All `visualize()` methods must use:**
```python
def visualize(self, df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    # df: Event log dataframe
    # fig: Plotly figure to annotate
    # Returns: Modified figure
    return fig
```

---

## 🔧 Registry API Reference

### Register a Visualization

```python
from core.visualization.registry import register_visualization

def my_viz(df: pd.DataFrame, fig: go.Figure) -> go.Figure:
    # Your code
    return fig

register_visualization("my_pattern", my_viz, visible=True)
```

### Unregister a Visualization

```python
from core.visualization.registry import unregister_visualization

unregister_visualization("my_pattern")
```

### Control Visibility

```python
from core.visualization.registry import VisualizationRegistry

# Hide a visualization
VisualizationRegistry.set_visibility("my_pattern", False)

# Show it again
VisualizationRegistry.set_visibility("my_pattern", True)
```

### List Registered Visualizations

```python
from core.visualization.registry import VisualizationRegistry

print(VisualizationRegistry.list_registered())
# ['my_pattern', 'another_pattern', ...]
```

---

## 📝 Examples

### Example 1: Weekend Highlighting

```python
import pandas as pd
import plotly.graph_objects as go
from core.visualization.registry import register_visualization

def highlight_weekends(df, fig):
    if 'actual_time' not in df.columns:
        return fig
    
    df['is_weekend'] = pd.to_datetime(df['actual_time']).dt.weekday >= 5
    weekend_dates = df[df['is_weekend']]['actual_time'].unique()
    
    for date in weekend_dates[:10]:
        fig.add_vline(
            x=date,
            line_dash="dash",
            line_color="gray",
            opacity=0.3
        )
    
    return fig

register_visualization("weekend_highlights", highlight_weekends)
```

### Example 2: Custom Annotations

```python
def add_case_annotation(df, fig):
    if 'case_id' not in df.columns:
        return fig
    
    # Highlight first case
    first_case = df['case_id'].iloc[0]
    case_df = df[df['case_id'] == first_case]
    
    fig.add_annotation(
        x=case_df['actual_time'].iloc[0],
        y=first_case,
        text=f"Case: {first_case}",
        showarrow=True,
        bgcolor="yellow"
    )
    
    return fig

register_visualization("case_annotations", add_case_annotation)
```

### Example 3: Custom Traces

```python
def add_trend_line(df, fig):
    if 'actual_time' not in df.columns:
        return fig
    
    # Count events per day
    df['date'] = pd.to_datetime(df['actual_time']).dt.date
    daily_counts = df.groupby('date').size()
    
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(daily_counts.index),
        y=daily_counts.values,
        mode='lines',
        name='Daily Count',
        yaxis='y2'
    ))
    
    fig.update_layout(yaxis2=dict(
        title="Count",
        overlaying='y',
        side='right'
    ))
    
    return fig

register_visualization("trend_line", add_trend_line)
```

---

## 🎨 How It Works

### Automatic Application

Registered visualizations are automatically applied in `display_chart()`:

```python
# In display_chart():
# 1. Pattern-based visualizations (Gap, Outlier, Temporal Cluster)
fig = pattern.visualize(df, fig)

# 2. Registry-based visualizations (your custom ones)
fig = VisualizationRegistry.apply_all(df, fig)
```

### Execution Order

1. Base chart is created
2. Pattern visualizations are applied (Gap, Outlier, Temporal Cluster)
3. Registry visualizations are applied (in registration order)

---

## ⚠️ Important Notes

1. **Function Signature**: Must be `(df: pd.DataFrame, fig: go.Figure) -> go.Figure`
2. **Return Value**: Always return the modified `fig`
3. **Error Handling**: Errors in one visualization won't break others
4. **Performance**: Keep visualizations fast (they run on every chart update)

---

## 🔍 See Also

- `examples/custom_visualization_example.py` - Full working examples
- `core/visualization/registry.py` - Registry implementation
- `core/detection/pattern_base.py` - Pattern base class

