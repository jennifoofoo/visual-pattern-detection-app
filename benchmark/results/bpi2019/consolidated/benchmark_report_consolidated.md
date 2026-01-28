# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 10:32:29

> [!CAUTION]
> **5 timeout(s) detected** - Some detection methods exceeded the 10-minute limit.

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2019.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 1,595,923 / 1,595,923 | 100.0% | 23046.7 | 36.485s | 1.0x |
| **optimized** | 49,998 / 1,595,923 | 3.1% | 1475.3 | 11.527s | 3.2x |
| **sqrt** | 19,998 / 1,595,923 | 1.3% | 678.4 | 3.585s | 10.2x |
| **minimal** | 4,996 / 1,595,923 | 0.3% | 238.4 | 0.695s | 52.5x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2019.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 8.47s | 1.0x |
| minimal | 1 | 100.0% | 0.19s | 45.4x |
| sqrt | 1 | 100.0% | 0.51s | 16.6x |
| optimized | 1 | 100.0% | 0.03s | 333.9x |

**Cluster (OPTICS)**
*Parameters: `TIMEOUT after 600s`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | nan | 100% | N/A | 1.0x |
| minimal | 91 | N/A | 2.94s | nanx |
| sqrt | 190 | N/A | 15.83s | nanx |
| optimized | 299 | N/A | 1m 2.3s | nanx |

**Gap (Transition)**
*Parameters: `TIMEOUT after 600s`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | nan | 100% | N/A | 1.0x |
| minimal | 147 | N/A | 0.84s | nanx |
| sqrt | 677 | N/A | 4.73s | nanx |
| optimized | 1315 | N/A | 13.85s | nanx |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 19.18s | 1.0x |
| minimal | 250 | 0.3% | 0.24s | 79.8x |
| sqrt | 1000 | 1.3% | 0.37s | 52.0x |
| optimized | 2500 | 3.1% | 0.83s | 23.1x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 4700 | 100% | 20.66s | 1.0x |
| minimal | 0 | 0.0% | 0.06s | 318.4x |
| sqrt | 0 | 0.0% | 0.27s | 76.4x |
| optimized | 0 | 0.0% | 0.53s | 39.1x |

**Temporal Cluster**
*Parameters: `min_cluster_size=421, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 443 | 100% | 28.19s | 1.0x |
| minimal | 1 | 0.2% | 0.06s | 448.2x |
| sqrt | 4 | 0.9% | 0.13s | 215.5x |
| optimized | 27 | 6.1% | 0.47s | 59.7x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 8.62s | 1.0x |
| minimal | 1 | 100.0% | 0.19s | 45.8x |
| sqrt | 1 | 100.0% | 0.53s | 16.3x |
| optimized | 1 | 100.0% | 0.03s | 339.0x |

**Cluster (OPTICS)**
*Parameters: `TIMEOUT after 600s`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | nan | 100% | N/A | 1.0x |
| minimal | 26 | N/A | 2.50s | nanx |
| sqrt | 94 | N/A | 15.25s | nanx |
| optimized | 181 | N/A | 49.04s | nanx |

**Gap (Transition)**
*Parameters: `TIMEOUT after 600s`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | nan | 100% | N/A | 1.0x |
| minimal | 147 | N/A | 0.90s | nanx |
| sqrt | 677 | N/A | 4.69s | nanx |
| optimized | 1315 | N/A | 13.32s | nanx |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 19.17s | 1.0x |
| minimal | 250 | 0.3% | 0.25s | 77.7x |
| sqrt | 1000 | 1.3% | 0.48s | 39.7x |
| optimized | 2500 | 3.1% | 0.85s | 22.6x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9966 | 100% | 1m 43.6s | 1.0x |
| minimal | 2894 | 29.0% | 0.55s | 187.0x |
| sqrt | 6708 | 67.3% | 2.02s | 51.2x |
| optimized | 10000 | 100.3% | 5.37s | 19.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=421, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 0.9x |
| sqrt | 0 | 100% | 0.00s | 0.8x |
| optimized | 0 | 100% | 0.00s | 0.9x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 8.52s | 1.0x |
| minimal | 1 | 100.0% | 0.19s | 45.2x |
| sqrt | 1 | 100.0% | 0.51s | 16.6x |
| optimized | 1 | 100.0% | 0.03s | 336.0x |

**Cluster (OPTICS)**
*Parameters: `TIMEOUT after 600s`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | nan | 100% | N/A | 1.0x |
| minimal | 86 | N/A | 2.48s | nanx |
| sqrt | 179 | N/A | 15.47s | nanx |
| optimized | 278 | N/A | 51.70s | nanx |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 43269 | 100% | 3m 21.0s | 1.0x |
| minimal | 147 | 0.3% | 0.82s | 244.1x |
| sqrt | 677 | 1.6% | 2.98s | 67.5x |
| optimized | 1315 | 3.0% | 6.23s | 32.3x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 18.64s | 1.0x |
| minimal | 250 | 0.3% | 0.21s | 89.0x |
| sqrt | 1000 | 1.3% | 0.40s | 46.9x |
| optimized | 2500 | 3.1% | 0.82s | 22.8x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1399 | 100% | 23.73s | 1.0x |
| minimal | 0 | 0.0% | 0.07s | 328.6x |
| sqrt | 0 | 0.0% | 0.29s | 80.6x |
| optimized | 4303 | 307.6% | 1.93s | 12.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=421, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 436 | 100% | 14.55s | 1.0x |
| minimal | 0 | 0.0% | 0.02s | 585.9x |
| sqrt | 2 | 0.5% | 0.06s | 246.0x |
| optimized | 20 | 4.6% | 0.19s | 76.5x |
