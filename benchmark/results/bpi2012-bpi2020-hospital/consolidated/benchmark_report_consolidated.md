# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 04:23:07

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 262,200 / 262,200 | 100.0% | 5157.2 | 206.283s | 1.0x |
| **optimized** | 49,988 / 262,200 | 19.1% | 1600.2 | 12.357s | 16.7x |
| **sqrt** | 19,976 / 262,200 | 7.6% | 1123.2 | 5.096s | 40.5x |
| **minimal** | 4,999 / 262,200 | 1.9% | 628.4 | 0.807s | 255.7x |

### DomesticDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 56,437 / 56,437 | 100.0% | 1949.8 | 14.595s | 1.0x |
| **optimized** | 39,870 / 56,437 | 70.6% | 1649.2 | 8.786s | 1.7x |
| **sqrt** | 2,904 / 56,437 | 5.1% | 358.6 | 0.396s | 36.9x |
| **minimal** | 874 / 56,437 | 1.5% | 150.7 | 0.271s | 53.8x |

### Hospital_log.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 150,291 / 150,291 | 100.0% | 1549.5 | 79.637s | 1.0x |
| **optimized** | 49,913 / 150,291 | 33.2% | 551.2 | 14.973s | 5.3x |
| **sqrt** | 19,967 / 150,291 | 13.3% | 239.1 | 4.740s | 16.8x |
| **minimal** | 4,991 / 150,291 | 3.3% | 92.3 | 1.028s | 77.5x |

### InternationalDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 72,151 / 72,151 | 100.0% | 1976.2 | 21.399s | 1.0x |
| **optimized** | 49,994 / 72,151 | 69.3% | 1677.3 | 13.148s | 1.6x |
| **sqrt** | 10,497 / 72,151 | 14.5% | 674.7 | 1.715s | 12.5x |
| **minimal** | 4,987 / 72,151 | 6.9% | 526.3 | 0.705s | 30.4x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.04s | 1.0x |
| minimal | 0 | 100% | 0.00s | 9.7x |
| sqrt | 0 | 100% | 0.01s | 5.4x |
| optimized | 0 | 100% | 0.02s | 2.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=170, max_eps=1.56`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 750 | 100% | 18m 28.8s | 1.0x |
| minimal | 95 | 12.7% | 3.06s | 362.8x |
| sqrt | 175 | 23.3% | 14.89s | 74.5x |
| optimized | 286 | 38.1% | 54.16s | 20.5x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 5m 36.2s | 1.0x |
| minimal | 189 | 1.8% | 0.99s | 340.1x |
| sqrt | 722 | 6.8% | 5.33s | 63.1x |
| optimized | 1879 | 17.7% | 18.64s | 18.0x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.37s | 1.0x |
| minimal | 250 | 1.9% | 0.15s | 15.4x |
| sqrt | 999 | 7.6% | 0.29s | 8.3x |
| optimized | 2500 | 19.1% | 0.68s | 3.5x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2930 | 100% | 15.08s | 1.0x |
| minimal | 2429 | 82.9% | 0.55s | 27.3x |
| sqrt | 4181 | 142.7% | 1.17s | 12.9x |
| optimized | 5780 | 197.3% | 3.26s | 4.6x |

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 1.31s | 1.0x |
| minimal | 0 | 100% | 0.04s | 33.2x |
| sqrt | 0 | 100% | 0.10s | 12.7x |
| optimized | 0 | 100% | 0.23s | 5.6x |

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
*Parameters: `algorithm=optics, min_samples=79, max_eps=1.67`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 341 | 100% | 1m 15.7s | 1.0x |
| minimal | 44 | 12.9% | 0.51s | 147.4x |
| sqrt | 71 | 20.8% | 1.54s | 49.3x |
| optimized | 297 | 87.1% | 44.96s | 1.7x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 20.80s | 1.0x |
| minimal | 32 | 1.7% | 0.11s | 196.2x |
| sqrt | 113 | 5.9% | 0.44s | 47.5x |
| optimized | 1335 | 70.2% | 11.71s | 1.8x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.74s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 6.7x |
| sqrt | 146 | 5.2% | 0.13s | 5.5x |
| optimized | 1994 | 70.7% | 0.47s | 1.6x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9979 | 100% | 2.28s | 1.0x |
| minimal | 369 | 3.7% | 0.09s | 24.4x |
| sqrt | 1105 | 11.1% | 0.24s | 9.3x |
| optimized | 9886 | 99.1% | 2.13s | 1.1x |

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 50 | 100% | 0.45s | 1.0x |
| minimal | 0 | 0.0% | 2.43s | 0.2x |
| sqrt | 1 | 2.0% | 0.03s | 16.8x |
| optimized | 31 | 62.0% | 0.22s | 2.0x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.04s | 1.0x |
| minimal | 1 | 100.0% | 0.04s | 0.9x |
| sqrt | 1 | 100.0% | 0.04s | 0.9x |
| optimized | 1 | 100.0% | 0.03s | 1.4x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=49.50`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 399 | 100% | 5m 29.5s | 1.0x |
| minimal | 80 | 20.1% | 2.97s | 111.0x |
| sqrt | 165 | 41.4% | 15.42s | 21.4x |
| optimized | 269 | 67.4% | 53.54s | 6.2x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 44.7s | 1.0x |
| minimal | 29 | 4.0% | 3.12s | 33.6x |
| sqrt | 83 | 11.5% | 12.16s | 8.6x |
| optimized | 215 | 29.7% | 29.08s | 3.6x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 2.04s | 1.0x |
| minimal | 250 | 3.3% | 0.17s | 12.3x |
| sqrt | 999 | 13.3% | 0.43s | 4.8x |
| optimized | 2496 | 33.2% | 0.81s | 2.5x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.94s | 1.0x |
| minimal | 0 | 100% | 0.06s | 15.2x |
| sqrt | 0 | 100% | 0.27s | 3.5x |
| optimized | 0 | 100% | 0.41s | 2.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 1.40s | 1.0x |
| minimal | 63 | 11.7% | 0.09s | 14.9x |
| sqrt | 123 | 22.8% | 0.22s | 6.3x |
| optimized | 256 | 47.4% | 0.47s | 3.0x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.0x |
| sqrt | 1 | 100.0% | 0.02s | 1.2x |
| optimized | 1 | 100.0% | 0.03s | 0.8x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=2.56`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 375 | 100% | 1m 39.0s | 1.0x |
| minimal | 82 | 21.9% | 2.73s | 36.2x |
| sqrt | 117 | 31.2% | 7.18s | 13.8x |
| optimized | 290 | 77.3% | 55.20s | 1.8x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 31.81s | 1.0x |
| minimal | 178 | 6.7% | 0.89s | 35.6x |
| sqrt | 403 | 15.2% | 2.24s | 14.2x |
| optimized | 1812 | 68.6% | 18.31s | 1.7x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.79s | 1.0x |
| minimal | 250 | 6.9% | 0.17s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.5x |
| optimized | 2500 | 69.4% | 0.60s | 1.3x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 8388 | 100% | 2.44s | 1.0x |
| minimal | 1871 | 22.3% | 0.33s | 7.4x |
| sqrt | 3987 | 47.5% | 0.70s | 3.5x |
| optimized | 9778 | 116.6% | 2.34s | 1.0x |

**Temporal Cluster**
*Parameters: `min_cluster_size=89, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7 | 100% | 0.40s | 1.0x |
| minimal | 4 | 57.1% | 0.05s | 8.5x |
| sqrt | 5 | 71.4% | 0.08s | 5.1x |
| optimized | 5 | 71.4% | 0.27s | 1.5x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.03s | 1.0x |
| minimal | 0 | 100% | 0.01s | 5.6x |
| sqrt | 0 | 100% | 0.00s | 6.7x |
| optimized | 0 | 100% | 0.02s | 1.9x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=170, max_eps=955.11`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 522 | 100% | 14m 3.9s | 1.0x |
| minimal | 2 | 0.4% | 2.76s | 306.2x |
| sqrt | 21 | 4.0% | 17.67s | 47.8x |
| optimized | 220 | 42.1% | 50.67s | 16.7x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 5m 25.2s | 1.0x |
| minimal | 189 | 1.8% | 0.98s | 331.4x |
| sqrt | 722 | 6.8% | 5.05s | 64.4x |
| optimized | 1879 | 17.7% | 18.19s | 17.9x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.32s | 1.0x |
| minimal | 250 | 1.9% | 0.17s | 14.0x |
| sqrt | 999 | 7.6% | 0.28s | 8.2x |
| optimized | 2500 | 19.1% | 0.51s | 4.6x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7557 | 100% | 18.98s | 1.0x |
| minimal | 952 | 12.6% | 0.45s | 42.3x |
| sqrt | 4326 | 57.2% | 1.55s | 12.3x |
| optimized | 2038 | 27.0% | 3.79s | 5.0x |

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 0.9x |
| sqrt | 0 | 100% | 0.00s | 0.8x |
| optimized | 0 | 100% | 0.00s | 1.0x |

#### DomesticDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.1x |
| sqrt | 1 | 100.0% | 0.01s | 2.1x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=782.96`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 152 | 100% | 54.13s | 1.0x |
| minimal | 51 | 33.6% | 0.45s | 119.2x |
| sqrt | 38 | 25.0% | 1.56s | 34.6x |
| optimized | 125 | 82.2% | 32.75s | 1.7x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 19.15s | 1.0x |
| minimal | 32 | 1.7% | 0.16s | 120.4x |
| sqrt | 113 | 5.9% | 0.42s | 45.4x |
| optimized | 1335 | 70.2% | 11.01s | 1.7x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.62s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.5x |
| sqrt | 146 | 5.2% | 0.14s | 4.6x |
| optimized | 1994 | 70.7% | 0.47s | 1.3x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10000 | 100% | 3.70s | 1.0x |
| minimal | 1972 | 19.7% | 0.19s | 20.0x |
| sqrt | 4375 | 43.8% | 0.57s | 6.4x |
| optimized | 9014 | 90.1% | 3.91s | 0.9x |

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.3x |
| sqrt | 0 | 100% | 0.00s | 1.2x |
| optimized | 0 | 100% | 0.00s | 1.2x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.05s | 1.0x |
| minimal | 1 | 100.0% | 0.04s | 1.3x |
| sqrt | 1 | 100.0% | 0.03s | 1.4x |
| optimized | 1 | 100.0% | 0.04s | 1.1x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=80.01`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 465 | 100% | 5m 29.3s | 1.0x |
| minimal | 99 | 21.3% | 3.02s | 109.1x |
| sqrt | 183 | 39.4% | 14.98s | 22.0x |
| optimized | 298 | 64.1% | 55.96s | 5.9x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 41.9s | 1.0x |
| minimal | 29 | 4.0% | 2.91s | 35.0x |
| sqrt | 83 | 11.5% | 11.43s | 8.9x |
| optimized | 215 | 29.7% | 30.95s | 3.3x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.57s | 1.0x |
| minimal | 250 | 3.3% | 0.23s | 6.9x |
| sqrt | 999 | 13.3% | 0.37s | 4.2x |
| optimized | 2496 | 33.2% | 0.75s | 2.1x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 773 | 100% | 1.43s | 1.0x |
| minimal | 407 | 52.7% | 0.20s | 7.2x |
| sqrt | 272 | 35.2% | 0.35s | 4.1x |
| optimized | 445 | 57.6% | 0.67s | 2.1x |

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 0.9x |
| sqrt | 0 | 100% | 0.00s | 1.0x |
| optimized | 0 | 100% | 0.00s | 1.0x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.1x |
| sqrt | 1 | 100.0% | 0.01s | 2.1x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=456.72`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 161 | 100% | 1m 14.9s | 1.0x |
| minimal | 23 | 14.3% | 2.68s | 28.0x |
| sqrt | 56 | 34.8% | 7.09s | 10.6x |
| optimized | 73 | 45.3% | 58.05s | 1.3x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 30.90s | 1.0x |
| minimal | 178 | 6.7% | 0.84s | 36.7x |
| sqrt | 403 | 15.2% | 2.26s | 13.7x |
| optimized | 1812 | 68.6% | 17.67s | 1.7x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.77s | 1.0x |
| minimal | 250 | 6.9% | 0.16s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.3x |
| optimized | 2500 | 69.4% | 0.59s | 1.3x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7524 | 100% | 7.14s | 1.0x |
| minimal | 6111 | 81.2% | 0.94s | 7.6x |
| sqrt | 5056 | 67.2% | 1.30s | 5.5x |
| optimized | 6779 | 90.1% | 5.45s | 1.3x |

**Temporal Cluster**
*Parameters: `min_cluster_size=89, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.2x |
| sqrt | 0 | 100% | 0.00s | 1.2x |
| optimized | 0 | 100% | 0.00s | 1.2x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.04s | 1.0x |
| minimal | 0 | 100% | 0.01s | 7.2x |
| sqrt | 0 | 100% | 0.01s | 5.8x |
| optimized | 0 | 100% | 0.02s | 2.7x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=164, max_eps=4.40`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 758 | 100% | 16m 37.0s | 1.0x |
| minimal | 97 | 12.8% | 2.72s | 366.7x |
| sqrt | 178 | 23.5% | 14.43s | 69.1x |
| optimized | 304 | 40.1% | 59.52s | 16.8x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 43.40s | 1.0x |
| minimal | 189 | 1.8% | 0.85s | 51.0x |
| sqrt | 722 | 6.8% | 3.27s | 13.3x |
| optimized | 1879 | 17.7% | 8.23s | 5.3x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.34s | 1.0x |
| minimal | 250 | 1.9% | 0.16s | 14.9x |
| sqrt | 999 | 7.6% | 0.29s | 8.2x |
| optimized | 2500 | 19.1% | 0.51s | 4.6x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9216 | 100% | 15.47s | 1.0x |
| minimal | 6419 | 69.7% | 1.62s | 9.5x |
| sqrt | 6174 | 67.0% | 27.23s | 0.6x |
| optimized | 7039 | 76.4% | 3.83s | 4.0x |

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.70s | 1.0x |
| minimal | 0 | 100% | 0.02s | 43.9x |
| sqrt | 0 | 100% | 0.16s | 4.3x |
| optimized | 0 | 100% | 0.11s | 6.1x |

#### DomesticDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.1x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.03s | 0.8x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=0.17`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 362 | 100% | 1m 15.5s | 1.0x |
| minimal | 48 | 13.3% | 0.41s | 185.9x |
| sqrt | 84 | 23.2% | 1.54s | 49.2x |
| optimized | 313 | 86.5% | 44.09s | 1.7x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 8.68s | 1.0x |
| minimal | 32 | 1.7% | 0.15s | 56.5x |
| sqrt | 113 | 5.9% | 0.33s | 26.2x |
| optimized | 1335 | 70.2% | 5.78s | 1.5x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.61s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.4x |
| sqrt | 146 | 5.2% | 0.14s | 4.5x |
| optimized | 1994 | 70.7% | 0.48s | 1.3x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.01s | 1.0x |
| minimal | 0 | 100% | 0.00s | 5.1x |
| sqrt | 0 | 100% | 0.00s | 4.4x |
| optimized | 0 | 100% | 0.01s | 1.4x |

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 49 | 100% | 0.22s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 31.5x |
| sqrt | 0 | 0.0% | 0.01s | 22.1x |
| optimized | 30 | 61.2% | 0.10s | 2.2x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.05s | 1.0x |
| minimal | 1 | 100.0% | 0.04s | 1.3x |
| sqrt | 1 | 100.0% | 0.02s | 2.0x |
| optimized | 1 | 100.0% | 0.04s | 1.1x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=1.70`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 454 | 100% | 7m 48.3s | 1.0x |
| minimal | 109 | 24.0% | 3.03s | 154.4x |
| sqrt | 189 | 41.6% | 17.43s | 26.9x |
| optimized | 261 | 57.5% | 1m 6.2s | 7.1x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 29.2s | 1.0x |
| minimal | 29 | 4.0% | 2.35s | 37.9x |
| sqrt | 83 | 11.5% | 11.61s | 7.7x |
| optimized | 215 | 29.7% | 29.40s | 3.0x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 2.07s | 1.0x |
| minimal | 250 | 3.3% | 0.17s | 12.3x |
| sqrt | 999 | 13.3% | 0.39s | 5.3x |
| optimized | 2496 | 33.2% | 0.79s | 2.6x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.29s | 1.0x |
| minimal | 0 | 100% | 0.04s | 6.7x |
| sqrt | 0 | 100% | 0.09s | 3.4x |
| optimized | 0 | 100% | 0.15s | 2.0x |

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 0.82s | 1.0x |
| minimal | 63 | 11.7% | 0.03s | 30.5x |
| sqrt | 123 | 22.8% | 0.09s | 9.6x |
| optimized | 256 | 47.4% | 0.24s | 3.4x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.04s | 1.0x |
| minimal | 0 | 0.0% | 0.02s | 2.0x |
| sqrt | 1 | 100.0% | 0.02s | 1.9x |
| optimized | 1 | 100.0% | 0.02s | 1.7x |

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=0.13`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 371 | 100% | 2m 3.9s | 1.0x |
| minimal | 97 | 26.1% | 2.86s | 43.4x |
| sqrt | 134 | 36.1% | 7.53s | 16.5x |
| optimized | 326 | 87.9% | 1m 9.0s | 1.8x |

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 12.06s | 1.0x |
| minimal | 178 | 6.7% | 0.81s | 14.9x |
| sqrt | 403 | 15.2% | 1.73s | 7.0x |
| optimized | 1812 | 68.6% | 8.38s | 1.4x |

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.77s | 1.0x |
| minimal | 250 | 6.9% | 0.17s | 4.6x |
| sqrt | 525 | 14.6% | 0.23s | 3.4x |
| optimized | 2500 | 69.4% | 0.59s | 1.3x |

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.01s | 1.0x |
| minimal | 0 | 100% | 0.00s | 5.1x |
| sqrt | 0 | 100% | 0.00s | 3.2x |
| optimized | 0 | 100% | 0.01s | 1.7x |

**Temporal Cluster**
*Parameters: `min_cluster_size=89, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 4 | 100% | 0.20s | 1.0x |
| minimal | 2 | 50.0% | 0.02s | 10.9x |
| sqrt | 2 | 50.0% | 0.03s | 5.8x |
| optimized | 2 | 50.0% | 0.14s | 1.4x |
