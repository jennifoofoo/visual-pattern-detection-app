# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 22:50:37

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 4875.1 | 200.682s | 1.0x |
| **optimized** | 19.1% | 929.8 | 11.018s | 18.2x |
| **sqrt** | 7.6% | 369.1 | 3.205s | 62.6x |
| **minimal** | 1.9% | 100.7 | 0.678s | 296.1x |

### DomesticDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 1007.8 | 13.245s | 1.0x |
| **optimized** | 70.6% | 719.1 | 8.415s | 1.6x |
| **sqrt** | 5.1% | 64.9 | 0.368s | 36.0x |
| **minimal** | 1.5% | 24.7 | 0.270s | 49.1x |

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 1549.5 | 62.065s | 1.0x |
| **optimized** | 33.2% | 551.2 | 11.594s | 5.4x |
| **sqrt** | 13.3% | 239.1 | 3.637s | 17.1x |
| **minimal** | 3.3% | 92.3 | 0.803s | 77.3x |

### InternationalDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 1310.6 | 19.989s | 1.0x |
| **optimized** | 69.3% | 909.0 | 11.775s | 1.7x |
| **sqrt** | 14.5% | 206.7 | 1.565s | 12.8x |
| **minimal** | 6.9% | 99.5 | 0.658s | 30.4x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.02s | 1.0x |
| minimal | 0 | 100% | 0.00s | 6.3x |
| sqrt | 0 | 100% | 0.00s | 5.1x |
| optimized | 0 | 100% | 0.01s | 2.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=170, max_eps=1.66`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 750 | 100% | 17m 7.5s | 1.0x |
| minimal | 95 | 12.7% | 2.57s | 399.9x |
| sqrt | 175 | 23.3% | 12.04s | 85.4x |
| optimized | 286 | 38.1% | 43.68s | 23.5x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 3m 42.2s | 1.0x |
| minimal | 189 | 1.8% | 0.76s | 293.1x |
| sqrt | 722 | 6.8% | 3.62s | 61.4x |
| optimized | 1879 | 17.7% | 13.03s | 17.0x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.32s | 1.0x |
| minimal | 250 | 1.9% | 0.15s | 15.1x |
| sqrt | 999 | 7.6% | 0.28s | 8.2x |
| optimized | 2500 | 19.1% | 0.52s | 4.5x |

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 1.25s | 1.0x |
| minimal | 0 | 100% | 0.04s | 31.8x |
| sqrt | 0 | 100% | 0.10s | 12.4x |
| optimized | 0 | 100% | 0.23s | 5.5x |

#### DomesticDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.1x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=1.66`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 341 | 100% | 58.27s | 1.0x |
| minimal | 44 | 12.9% | 0.36s | 160.3x |
| sqrt | 71 | 20.8% | 1.31s | 44.5x |
| optimized | 297 | 87.1% | 37.25s | 1.6x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 14.93s | 1.0x |
| minimal | 32 | 1.7% | 0.12s | 128.5x |
| sqrt | 113 | 5.9% | 0.37s | 40.6x |
| optimized | 1335 | 70.2% | 8.47s | 1.8x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.59s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.3x |
| sqrt | 146 | 5.2% | 0.13s | 4.4x |
| optimized | 1994 | 70.7% | 0.47s | 1.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 50 | 100% | 0.31s | 1.0x |
| minimal | 0 | 0.0% | 2.20s | 0.1x |
| sqrt | 1 | 2.0% | 0.03s | 11.3x |
| optimized | 31 | 62.0% | 0.22s | 1.4x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.3x |
| sqrt | 1 | 100.0% | 0.02s | 1.3x |
| optimized | 1 | 100.0% | 0.03s | 0.8x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=48.52`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 399 | 100% | 4m 20.8s | 1.0x |
| minimal | 80 | 20.1% | 2.40s | 108.5x |
| sqrt | 165 | 41.4% | 12.16s | 21.4x |
| optimized | 269 | 67.4% | 43.37s | 6.0x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 12.4s | 1.0x |
| minimal | 29 | 4.0% | 2.11s | 34.4x |
| sqrt | 83 | 11.5% | 8.38s | 8.6x |
| optimized | 215 | 29.7% | 21.42s | 3.4x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.56s | 1.0x |
| minimal | 250 | 3.3% | 0.16s | 9.5x |
| sqrt | 999 | 13.3% | 0.34s | 4.6x |
| optimized | 2496 | 33.2% | 0.56s | 2.8x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.68s | 1.0x |
| minimal | 0 | 100% | 0.06s | 11.2x |
| sqrt | 0 | 100% | 0.19s | 3.5x |
| optimized | 0 | 100% | 0.37s | 1.9x |

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 1.37s | 1.0x |
| minimal | 63 | 11.7% | 0.09s | 15.0x |
| sqrt | 123 | 22.8% | 0.23s | 5.8x |
| optimized | 256 | 47.4% | 0.49s | 2.8x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.0x |
| sqrt | 1 | 100.0% | 0.01s | 1.9x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=2.59`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 375 | 100% | 1m 17.2s | 1.0x |
| minimal | 82 | 21.9% | 2.43s | 31.7x |
| sqrt | 117 | 31.2% | 5.78s | 13.3x |
| optimized | 290 | 77.3% | 43.00s | 1.8x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 22.57s | 1.0x |
| minimal | 178 | 6.7% | 0.70s | 32.1x |
| sqrt | 403 | 15.2% | 1.70s | 13.3x |
| optimized | 1812 | 68.6% | 12.81s | 1.8x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.77s | 1.0x |
| minimal | 250 | 6.9% | 0.16s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.3x |
| optimized | 2500 | 69.4% | 0.59s | 1.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=89, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7 | 100% | 0.40s | 1.0x |
| minimal | 4 | 57.1% | 0.05s | 8.1x |
| sqrt | 5 | 71.4% | 0.08s | 5.2x |
| optimized | 5 | 71.4% | 0.27s | 1.5x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.03s | 1.0x |
| minimal | 0 | 100% | 0.00s | 6.5x |
| sqrt | 0 | 100% | 0.00s | 5.2x |
| optimized | 0 | 100% | 0.01s | 2.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=170, max_eps=938.87`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 522 | 100% | 11m 29.0s | 1.0x |
| minimal | 2 | 0.4% | 2.58s | 267.3x |
| sqrt | 21 | 4.0% | 13.67s | 50.4x |
| optimized | 220 | 42.1% | 40.56s | 17.0x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 3m 33.2s | 1.0x |
| minimal | 189 | 1.8% | 0.72s | 295.6x |
| sqrt | 722 | 6.8% | 3.57s | 59.7x |
| optimized | 1879 | 17.7% | 12.47s | 17.1x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.34s | 1.0x |
| minimal | 250 | 1.9% | 0.15s | 15.3x |
| sqrt | 999 | 7.6% | 0.29s | 8.0x |
| optimized | 2500 | 19.1% | 0.51s | 4.6x |

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.0x |
| sqrt | 0 | 100% | 0.00s | 1.1x |
| optimized | 0 | 100% | 0.00s | 1.1x |

#### DomesticDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.2x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=749.87`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 152 | 100% | 41.62s | 1.0x |
| minimal | 51 | 33.6% | 0.38s | 109.2x |
| sqrt | 38 | 25.0% | 1.29s | 32.1x |
| optimized | 125 | 82.2% | 30.48s | 1.4x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 14.85s | 1.0x |
| minimal | 32 | 1.7% | 0.13s | 116.5x |
| sqrt | 113 | 5.9% | 0.37s | 40.4x |
| optimized | 1335 | 70.2% | 8.27s | 1.8x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.59s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.3x |
| sqrt | 146 | 5.2% | 0.13s | 4.4x |
| optimized | 1994 | 70.7% | 0.47s | 1.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.2x |
| sqrt | 0 | 100% | 0.00s | 1.3x |
| optimized | 0 | 100% | 0.00s | 1.2x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.3x |
| sqrt | 1 | 100.0% | 0.02s | 1.2x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=81.70`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 465 | 100% | 4m 33.1s | 1.0x |
| minimal | 99 | 21.3% | 2.39s | 114.4x |
| sqrt | 183 | 39.4% | 12.38s | 22.1x |
| optimized | 298 | 64.1% | 46.22s | 5.9x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 11.0s | 1.0x |
| minimal | 29 | 4.0% | 2.05s | 34.7x |
| sqrt | 83 | 11.5% | 8.41s | 8.4x |
| optimized | 215 | 29.7% | 21.45s | 3.3x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.53s | 1.0x |
| minimal | 250 | 3.3% | 0.16s | 9.3x |
| sqrt | 999 | 13.3% | 0.31s | 4.9x |
| optimized | 2496 | 33.2% | 0.55s | 2.8x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 773 | 100% | 1.14s | 1.0x |
| minimal | 407 | 52.7% | 0.15s | 7.8x |
| sqrt | 272 | 35.2% | 0.26s | 4.4x |
| optimized | 445 | 57.6% | 0.48s | 2.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.1x |
| sqrt | 0 | 100% | 0.00s | 1.2x |
| optimized | 0 | 100% | 0.00s | 1.3x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.0x |
| sqrt | 1 | 100.0% | 0.01s | 1.9x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=444.95`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 161 | 100% | 1m 1.6s | 1.0x |
| minimal | 23 | 14.3% | 2.30s | 26.7x |
| sqrt | 56 | 34.8% | 5.82s | 10.6x |
| optimized | 73 | 45.3% | 45.99s | 1.3x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 21.72s | 1.0x |
| minimal | 178 | 6.7% | 0.74s | 29.2x |
| sqrt | 403 | 15.2% | 1.62s | 13.4x |
| optimized | 1812 | 68.6% | 12.47s | 1.7x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.78s | 1.0x |
| minimal | 250 | 6.9% | 0.17s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.5x |
| optimized | 2500 | 69.4% | 0.59s | 1.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=89, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.2x |
| sqrt | 0 | 100% | 0.00s | 1.1x |
| optimized | 0 | 100% | 0.00s | 0.7x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.02s | 1.0x |
| minimal | 0 | 100% | 0.00s | 6.4x |
| sqrt | 0 | 100% | 0.00s | 5.0x |
| optimized | 0 | 100% | 0.01s | 2.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=164, max_eps=4.46`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 758 | 100% | 13m 34.0s | 1.0x |
| minimal | 97 | 12.8% | 2.36s | 345.0x |
| sqrt | 178 | 23.5% | 11.54s | 70.6x |
| optimized | 304 | 40.1% | 47.13s | 17.3x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 35.37s | 1.0x |
| minimal | 189 | 1.8% | 0.65s | 54.1x |
| sqrt | 722 | 6.8% | 2.63s | 13.5x |
| optimized | 1879 | 17.7% | 6.50s | 5.4x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.32s | 1.0x |
| minimal | 250 | 1.9% | 0.15s | 15.1x |
| sqrt | 999 | 7.6% | 0.28s | 8.2x |
| optimized | 2500 | 19.1% | 0.50s | 4.6x |

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.68s | 1.0x |
| minimal | 0 | 100% | 0.02s | 45.3x |
| sqrt | 0 | 100% | 0.05s | 14.3x |
| optimized | 0 | 100% | 0.11s | 6.0x |

#### DomesticDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 1.9x |
| sqrt | 1 | 100.0% | 0.01s | 2.1x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=0.17`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 362 | 100% | 59.96s | 1.0x |
| minimal | 48 | 13.3% | 0.38s | 158.4x |
| sqrt | 84 | 23.2% | 1.35s | 44.6x |
| optimized | 313 | 86.5% | 35.22s | 1.7x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 6.75s | 1.0x |
| minimal | 32 | 1.7% | 0.10s | 65.0x |
| sqrt | 113 | 5.9% | 0.36s | 18.7x |
| optimized | 1335 | 70.2% | 4.76s | 1.4x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.59s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.4x |
| sqrt | 146 | 5.2% | 0.14s | 4.4x |
| optimized | 1994 | 70.7% | 0.47s | 1.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 49 | 100% | 0.15s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 25.6x |
| sqrt | 0 | 0.0% | 0.01s | 14.8x |
| optimized | 30 | 61.2% | 0.10s | 1.4x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.3x |
| sqrt | 1 | 100.0% | 0.02s | 1.2x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=1.70`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 454 | 100% | 6m 7.2s | 1.0x |
| minimal | 109 | 24.0% | 2.52s | 146.0x |
| sqrt | 189 | 41.6% | 13.93s | 26.4x |
| optimized | 262 | 57.7% | 52.00s | 7.1x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 3.7s | 1.0x |
| minimal | 29 | 4.0% | 2.09s | 30.5x |
| sqrt | 83 | 11.5% | 8.33s | 7.6x |
| optimized | 215 | 29.7% | 20.82s | 3.1x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.54s | 1.0x |
| minimal | 250 | 3.3% | 0.17s | 9.3x |
| sqrt | 999 | 13.3% | 0.31s | 4.9x |
| optimized | 2496 | 33.2% | 0.56s | 2.8x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.17s | 1.0x |
| minimal | 0 | 100% | 0.03s | 6.8x |
| sqrt | 0 | 100% | 0.06s | 2.8x |
| optimized | 0 | 100% | 0.10s | 1.7x |

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 0.80s | 1.0x |
| minimal | 63 | 11.7% | 0.03s | 30.7x |
| sqrt | 123 | 22.8% | 0.08s | 9.5x |
| optimized | 256 | 47.4% | 0.23s | 3.5x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.0x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.02s | 1.1x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=0.13`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 371 | 100% | 1m 44.3s | 1.0x |
| minimal | 97 | 26.1% | 2.46s | 42.3x |
| sqrt | 134 | 36.1% | 6.39s | 16.3x |
| optimized | 326 | 87.9% | 53.66s | 1.9x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 9.46s | 1.0x |
| minimal | 178 | 6.7% | 0.64s | 14.9x |
| sqrt | 403 | 15.2% | 1.33s | 7.1x |
| optimized | 1812 | 68.6% | 6.49s | 1.5x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.77s | 1.0x |
| minimal | 250 | 6.9% | 0.16s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.3x |
| optimized | 2500 | 69.4% | 0.58s | 1.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=89, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 4 | 100% | 0.20s | 1.0x |
| minimal | 2 | 50.0% | 0.02s | 12.1x |
| sqrt | 2 | 50.0% | 0.03s | 6.6x |
| optimized | 2 | 50.0% | 0.13s | 1.5x |
