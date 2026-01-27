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
- **Algorithm Selection**: Choose between 'optics' or 'dbscan' via the `algorithm` parameter.
- **Dynamic Hyperparameters**: All clustering parameters are automatically calculated based on data characteristics:
  - **min_samples**: Dynamically calculated as `min(20, max(3, int(sqrt(n_points) / 3)))` (value is floored to an integer) - adapts to dataset size
  - **OPTICS-specific**:
    - **max_eps**: Auto-calculated based on standard deviation of coordinates (0.3 × avg_std, capped between 0.1 and 2.0)
    - **min_cluster_size**: Set equal to dynamic min_samples
    - **xi**: Fixed at 0.01 (reachability threshold)
  - **DBSCAN-specific**:
    - **eps**: Auto-calculated based on standard deviation of coordinates (0.2 × avg_std, capped between 0.05 and 1.0)
- **Parameter Override**: All auto-calculated parameters can be manually overridden by passing them as kwargs during initialization.
- **Hierarchical Clustering**: If a 'color' column is specified, clustering is performed within each color group, supporting multi-level analysis.

## Impossible Configurations + Explanation
❌ **None - This pattern works with any axis combination**

The clustering pattern adapts to the selected axes and data types. It can cluster on any combination of numeric or categorical axes, and will automatically encode and scale features as needed.

## Types of Clustering
- **Spatial Clustering**: Groups events based on their position in the chart (e.g., time vs. activity).
- **Categorical Clustering**: Groups events based on encoded categorical features (e.g., activity, resource).
- **Hierarchical Clustering**: Groups by a higher-level category (e.g., resource) and clusters within each group.

## Interpretation
## How Categorical Encoding and Visualization Work Together

Clustering algorithms require numeric input, so categorical columns (like Activity, Resource, or Case ID) are encoded numerically (e.g., label encoding, one-hot encoding, or embeddings) before clustering. The clustering is performed in this numeric space.

For visualization, Plotly and other plotting libraries can display categorical axes using the original string values (e.g., "Activity A", "Resource X"). When plotting, the original category names are used for axes and shapes, not the encoded numbers.

**How clusters are mapped to the plot:**
- After clustering, each data point has a cluster label (from the numeric space).
- To visualize clusters, the original data is grouped by cluster label.
- For each cluster, the min/max (or unique set) of the original categorical values is found (not the encoded numbers).
- Rectangles or highlights are drawn on the plot using the original category names, so the rectangles align with the visible axis labels.

**Summary:**
- Encoding is only for clustering.
- Visualization always uses the original category names.
- The mapping from cluster to plot is done by grouping the original data by cluster label, then using the original values for drawing.

This ensures that clusters found in the encoded numeric space are correctly and intuitively visualized using the original, human-readable category names.
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
