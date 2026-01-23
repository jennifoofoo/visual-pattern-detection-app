# Benchmark Analysis: bpi 2019 report

## 📄 Dataset: BPI_Challenge_2019.xes

> [!WARNING]
> **Baseline Missing**: The `full` sampling mode was not completed for this dataset. Retention percentages and Speedup factors relative to the full dataset cannot be calculated for the aggregated summary, but raw algorithm metrics are still available below.

### ⚙️ Configuration: Time x Case x Activity

#### 📊 Configuration Summary
Aggregated metrics across all algorithms for this configuration.

| Sampling Mode   |   Total Patterns |   Total Time (s) | Avg Retention (%)   | Avg Speedup (x)   |
|:----------------|-----------------:|-----------------:|:--------------------|:------------------|
| minimal         |             4009 |           3.5156 | nan%                | N/A               |
| sqrt            |            42497 |          18.9741 | nan%                | N/A               |
| optimized       |            16599 |          27.0592 | nan%                | N/A               |

#### 🔍 Algorithm: Case Arrival Trend
**Parameters:** `aggregation=W, mann_kendall`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |          1 |  0.183097  | nan%            | N/A           |
| sqrt            |          1 |  0.502384  | nan%            | N/A           |
| optimized       |          1 |  0.0291059 | nan%            | N/A           |

#### 🔍 Algorithm: Cluster (OPTICS)
**Parameters:** `algorithm=optics, min_samples=5`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        717 |    2.09433 | nan%            | N/A           |
| sqrt            |       2696 |   11.1699  | nan%            | N/A           |
| optimized       |          0 |   11.555   | nan%            | N/A           |

#### 🔍 Algorithm: Gap (Transition)
**Parameters:** `mode=transition, min_samples=5`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        147 |   0.638824 | nan%            | N/A           |
| sqrt            |        677 |   3.48094  | nan%            | N/A           |
| optimized       |       1315 |  10.803    | nan%            | N/A           |

#### 🔍 Algorithm: Outlier Detection
**Parameters:** `isolation_forest + statistical`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        250 |   0.168976 | nan%            | N/A           |
| sqrt            |       1000 |   0.374641 | nan%            | N/A           |
| optimized       |       2500 |   0.615873 | nan%            | N/A           |

#### 🔍 Algorithm: Sequence Detection
**Parameters:** `min_support=30, prefixspan`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |       2894 |   0.430351 | nan%            | N/A           |
| sqrt            |      38123 |   3.44618  | nan%            | N/A           |
| optimized       |      12783 |   4.05618  | nan%            | N/A           |

#### 🔍 Algorithm: Temporal Cluster
**Parameters:** `eps=auto, min_samples=auto`

| Sampling Mode   |   Patterns |    Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|------------:|:----------------|:--------------|
| minimal         |          0 | 4.62532e-05 | nan%            | N/A           |
| sqrt            |          0 | 4.17233e-05 | nan%            | N/A           |
| optimized       |          0 | 4.17233e-05 | nan%            | N/A           |

### ⚙️ Configuration: Time x Resource x Activity

#### 📊 Configuration Summary
Aggregated metrics across all algorithms for this configuration.

| Sampling Mode   |   Total Patterns |   Total Time (s) | Avg Retention (%)   | Avg Speedup (x)   |
|:----------------|-----------------:|-----------------:|:--------------------|:------------------|
| minimal         |              664 |           3.0076 | nan%                | N/A               |
| sqrt            |             2717 |          14.2269 | nan%                | N/A               |
| optimized       |            13961 |          45.0839 | nan%                | N/A               |

#### 🔍 Algorithm: Case Arrival Trend
**Parameters:** `aggregation=W, mann_kendall`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |          1 |  0.182868  | nan%            | N/A           |
| sqrt            |          1 |  0.497662  | nan%            | N/A           |
| optimized       |          1 |  0.0296683 | nan%            | N/A           |

#### 🔍 Algorithm: Cluster (OPTICS)
**Parameters:** `algorithm=optics, min_samples=5`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        516 |    2.16566 | nan%            | N/A           |
| sqrt            |       2039 |   11.1213  | nan%            | N/A           |
| optimized       |       4650 |   38.3854  | nan%            | N/A           |

#### 🔍 Algorithm: Gap (Transition)
**Parameters:** `mode=transition, min_samples=5`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        147 |   0.591266 | nan%            | N/A           |
| sqrt            |        677 |   2.4116   | nan%            | N/A           |
| optimized       |       1315 |   5.16162  | nan%            | N/A           |

#### 🔍 Algorithm: Outlier Detection
**Parameters:** `isolation_forest + statistical`

| Sampling Mode   |   Patterns |    Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|------------:|:----------------|:--------------|
| minimal         |          0 | 7.10487e-05 | nan%            | N/A           |
| sqrt            |          0 | 0.000108957 | nan%            | N/A           |
| optimized       |          0 | 7.65324e-05 | nan%            | N/A           |

#### 🔍 Algorithm: Sequence Detection
**Parameters:** `min_support=30, prefixspan`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |          0 |  0.0676863 | nan%            | N/A           |
| sqrt            |          0 |  0.196262  | nan%            | N/A           |
| optimized       |       7995 |  1.50706   | nan%            | N/A           |

#### 🔍 Algorithm: Temporal Cluster
**Parameters:** `eps=auto, min_samples=auto`

| Sampling Mode   |   Patterns |    Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|------------:|:----------------|:--------------|
| minimal         |          0 | 3.14713e-05 | nan%            | N/A           |
| sqrt            |          0 | 3.19481e-05 | nan%            | N/A           |
| optimized       |          0 | 2.36034e-05 | nan%            | N/A           |

---

