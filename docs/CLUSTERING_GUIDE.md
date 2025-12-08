# Clustering Pattern Documentation

## Name
**Clustering Pattern**

## Description
Automatically detects clusters and groupings in process mining event logs visualized as dotted charts. This pattern identifies groups of similar events, cases, or activities based on their spatial or categorical proximity, helping analysts discover process variants, bottlenecks, and behavioral patterns.

## Supported Algorithms
- **OPTICS**: Hierarchical density-based clustering, robust to varying densities and noise.
- **DBSCAN**: Density-based clustering, effective for spatial patterns and noise detection.

## Data Handling
- **Preprocessing**: Uses `DataPreprocessor` for consistent data cleaning and transformation.
- **Categorical Encoding**: Converts categorical columns to numeric embeddings using TF-IDF vectorization and PCA dimensionality reduction, enabling clustering on non-numeric data.
- **Feature Scaling**: Applies standard scaling to ensure all features contribute equally to clustering.

## Visual Representation
- **Markers**: Each cluster is shown with a distinct color on the dotted chart.
- **Noise**: Points not assigned to any cluster (noise) are shown in light gray with 'x' symbols.
- **Legend**: Displays cluster labels and noise count.
- **Annotation**: Shows algorithm, number of clusters, and noise points.
- **Hover Info**: Displays cluster label and event details for each point.

## Configuration & Parameters
- **Algorithm Selection**: Choose between 'optics', 'dbscan' via the `algorithm` parameter.
- **Dynamic Parameters**: Embedding dimension and clustering parameters are auto-calculated based on data size and complexity, but can be overridden.
- **Hierarchical Clustering**: If a 'color' column is specified, clustering is performed within each color group, supporting multi-level analysis.

## Impossible Configurations + Explanation
❌ **None - This pattern works with any axis combination**

The clustering pattern adapts to the selected axes and data types. It can cluster on any combination of numeric or categorical axes, and will automatically encode and scale features as needed.

## Types of Clustering
- **Spatial Clustering**: Groups events based on their position in the chart (e.g., time vs. activity).
- **Categorical Clustering**: Groups events based on encoded categorical features (e.g., activity, resource).
- **Hierarchical Clustering**: Groups by a higher-level category (e.g., resource) and clusters within each group.

## Interpretation
Clusters represent groups of events or cases that share similar characteristics in the selected view. Large, dense clusters may indicate common process paths or bottlenecks, while small or isolated clusters may reveal rare behaviors or process variants. Noise points are events that do not fit well into any cluster, potentially indicating outliers or unique cases.

## Example Usage
```python
from core.detection.cluster_pattern import ClusterPattern

# Initialize with view configuration and algorithm
pattern = ClusterPattern(view_config={'x': 'actual_time', 'y': 'activity'}, algorithm='optics')

# Detect clusters in a DataFrame
pattern.detect(df)

# Visualize clusters on a Plotly figure
fig = pattern.visualize(df, fig)
```
