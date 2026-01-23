# Benchmark Analysis: hospital log report

## 📄 Dataset: Hospital_log.xes

### ⚙️ Configuration: Time x Case x Activity

#### 🔍 Algorithm: Case Arrival Trend
**Parameters:** `aggregation=W, mann_kendall`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |          1 |  0.0222642 | 100.0%          | 1.39x         |
| sqrt            |          1 |  0.0223742 | 100.0%          | 1.39x         |
| optimized       |          1 |  0.0233121 | 100.0%          | 1.33x         |
| full            |          1 |  0.0309896 | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Cluster (OPTICS)
**Parameters:** `algorithm=optics, min_samples=5`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        393 |    2.46916 | 4.8%            | 100.10x       |
| sqrt            |       1234 |   11.6617  | 14.9%           | 21.19x        |
| optimized       |       2871 |   39.3971  | 34.7%           | 6.27x         |
| full            |       8264 |  247.169   | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Gap (Transition)
**Parameters:** `mode=transition, min_samples=5`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |         29 |    2.20328 | 4.0%            | 33.03x        |
| sqrt            |         83 |    8.28947 | 11.5%           | 8.78x         |
| optimized       |        215 |   21.0055  | 29.7%           | 3.46x         |
| full            |        724 |   72.7727  | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Outlier Detection
**Parameters:** `isolation_forest + statistical`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        250 |   0.173542 | 3.3%            | 9.55x         |
| sqrt            |        999 |   0.324899 | 13.3%           | 5.10x         |
| optimized       |       2496 |   0.567163 | 33.2%           | 2.92x         |
| full            |       7515 |   1.65756  | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Sequence Detection
**Parameters:** `min_support=30, prefixspan`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        407 |   0.153327 | 52.7%           | 8.42x         |
| sqrt            |        272 |   0.271548 | 35.2%           | 4.75x         |
| optimized       |        445 |   0.521491 | 57.6%           | 2.48x         |
| full            |        773 |   1.29072  | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Temporal Cluster
**Parameters:** `eps=auto, min_samples=auto`

| Sampling Mode   |   Patterns |    Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|------------:|:----------------|:--------------|
| minimal         |          0 | 4.69685e-05 | 100.0%          | 0.90x         |
| sqrt            |          0 | 3.57628e-05 | 100.0%          | 1.19x         |
| optimized       |          0 | 3.50475e-05 | 100.0%          | 1.21x         |
| full            |          0 | 4.24385e-05 | 100.0%          | 1.00x         |

### ⚙️ Configuration: Time x Resource x Activity

#### 🔍 Algorithm: Case Arrival Trend
**Parameters:** `aggregation=W, mann_kendall`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |          1 |  0.0237801 | 100.0%          | 1.14x         |
| sqrt            |          1 |  0.021594  | 100.0%          | 1.25x         |
| optimized       |          1 |  0.0236931 | 100.0%          | 1.14x         |
| full            |          1 |  0.0270782 | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Cluster (OPTICS)
**Parameters:** `algorithm=optics, min_samples=5`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        439 |     2.612  | 10.0%           | 156.36x       |
| sqrt            |       1392 |    13.3214 | 31.6%           | 30.66x        |
| optimized       |       2768 |    53.6737 | 62.8%           | 7.61x         |
| full            |       4406 |   408.405  | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Gap (Transition)
**Parameters:** `mode=transition, min_samples=5`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |         29 |    2.23053 | 4.0%            | 27.77x        |
| sqrt            |         83 |    8.12602 | 11.5%           | 7.62x         |
| optimized       |        215 |   20.2141  | 29.7%           | 3.06x         |
| full            |        724 |   61.9395  | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Outlier Detection
**Parameters:** `isolation_forest + statistical`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |        250 |   0.17192  | 3.3%            | 9.48x         |
| sqrt            |        999 |   0.320218 | 13.3%           | 5.09x         |
| optimized       |       2496 |   0.597389 | 33.2%           | 2.73x         |
| full            |       7515 |   1.62913  | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Sequence Detection
**Parameters:** `min_support=30, prefixspan`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |          0 |  0.0318172 | 100.0%          | 7.03x         |
| sqrt            |          0 |  0.062602  | 100.0%          | 3.57x         |
| optimized       |          0 |  0.101744  | 100.0%          | 2.20x         |
| full            |          0 |  0.223691  | 100.0%          | 1.00x         |

#### 🔍 Algorithm: Temporal Cluster
**Parameters:** `eps=auto, min_samples=auto`

| Sampling Mode   |   Patterns |   Time (s) | Retention (%)   | Speedup (x)   |
|:----------------|-----------:|-----------:|:----------------|:--------------|
| minimal         |         75 |  0.0344687 | 7.0%            | 29.83x        |
| sqrt            |        376 |  0.113794  | 34.9%           | 9.04x         |
| optimized       |        742 |  0.30975   | 68.8%           | 3.32x         |
| full            |       1078 |  1.0282    | 100.0%          | 1.00x         |

---

