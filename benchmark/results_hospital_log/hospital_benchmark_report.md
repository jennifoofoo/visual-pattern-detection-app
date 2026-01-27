# Pattern Detection Benchmark Report

**Generated:** 2026-01-27 14:52:16

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| Hospital_log.xes | 83.0 | 150,291 | 10.99s |

## 📊 Sampling Performance Summary

| Sampling | Events Retained | Pattern Retention | Time Saved | Sampling Time |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 100.0% | 0.0% | 0.000s |
| **optimized** | 33.2% | 62.8% | -92.3% | 0.400s |
| **sqrt** | 13.3% | 48.8% | -67.2% | 0.400s |
| **minimal** | 3.3% | 42.4% | -60.6% | 0.400s |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### Case Arrival Trend
**Parameters:** `aggregation=W, mann_kendall`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 1 | 100.0% | 0.03s | 0.0% |
| optimized | 1 | 100.0% | 0.02s | -481.6% |
| sqrt | 1 | 100.0% | 0.02s | -436.9% |
| minimal | 1 | 100.0% | 0.02s | -438.2% |

#### Cluster (OPTICS)
**Parameters:** `algorithm=optics, min_samples=129, max_eps=49.02`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 399 | 100.0% | 4m 11.0s | 0.0% |
| optimized | 269 | 67.4% | 42.02s | 83.1% |
| sqrt | 165 | 41.4% | 12.09s | 95.0% |
| minimal | 80 | 20.1% | 2.34s | 98.9% |

#### Gap (Transition)
**Parameters:** `mode=transition, min_samples=5`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 724 | 100.0% | 1m 16.7s | 0.0% |
| optimized | 215 | 29.7% | 21.21s | 71.8% |
| sqrt | 83 | 11.5% | 8.42s | 88.5% |
| minimal | 29 | 4.0% | 2.01s | 96.9% |

#### Outlier Detection
**Parameters:** `isolation_forest + statistical`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 7515 | 100.0% | 1.66s | 0.0% |
| optimized | 2496 | 33.2% | 0.61s | 40.2% |
| sqrt | 999 | 13.3% | 0.33s | 58.5% |
| minimal | 250 | 3.3% | 0.17s | 68.2% |

#### Sequence Detection
**Parameters:** `min_support=30, prefixspan`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 0 | 100.0% | 0.72s | 0.0% |
| optimized | 0 | 100.0% | 0.37s | -1.2% |
| sqrt | 0 | 100.0% | 0.26s | 16.2% |
| minimal | 0 | 100.0% | 0.06s | 42.7% |

#### Temporal Cluster
**Parameters:** `min_cluster_size=129, temporal_eps=1.00`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 540 | 100.0% | 1.51s | 0.0% |
| optimized | 256 | 47.4% | 0.50s | 41.9% |
| sqrt | 123 | 22.8% | 0.22s | 61.9% |
| minimal | 63 | 11.7% | 0.10s | 69.7% |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### Case Arrival Trend
**Parameters:** `aggregation=W, mann_kendall`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 1 | 100.0% | 0.03s | 0.0% |
| optimized | 1 | 100.0% | 0.02s | -476.7% |
| sqrt | 1 | 100.0% | 0.02s | -433.1% |
| minimal | 1 | 100.0% | 0.02s | -434.8% |

#### Cluster (OPTICS)
**Parameters:** `algorithm=optics, min_samples=129, max_eps=79.50`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 465 | 100.0% | 4m 26.9s | 0.0% |
| optimized | 298 | 64.1% | 44.88s | 83.0% |
| sqrt | 183 | 39.4% | 12.09s | 95.3% |
| minimal | 99 | 21.3% | 2.34s | 99.0% |

#### Gap (Transition)
**Parameters:** `mode=transition, min_samples=5`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 724 | 100.0% | 1m 10.9s | 0.0% |
| optimized | 215 | 29.7% | 21.18s | 69.6% |
| sqrt | 83 | 11.5% | 8.35s | 87.7% |
| minimal | 29 | 4.0% | 2.04s | 96.6% |

#### Outlier Detection
**Parameters:** `isolation_forest + statistical`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 7515 | 100.0% | 1.89s | 0.0% |
| optimized | 2496 | 33.2% | 0.59s | 48.5% |
| sqrt | 999 | 13.3% | 0.36s | 62.0% |
| minimal | 250 | 3.3% | 0.17s | 71.7% |

#### Sequence Detection
**Parameters:** `min_support=30, prefixspan`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 773 | 100.0% | 1.17s | 0.0% |
| optimized | 445 | 57.6% | 0.54s | 22.3% |
| sqrt | 272 | 35.2% | 0.26s | 47.2% |
| minimal | 407 | 52.7% | 0.14s | 57.0% |

#### Temporal Cluster
**Parameters:** `min_cluster_size=129, temporal_eps=1.00`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 0 | 100.0% | 0.00s | 0.0% |
| optimized | 0 | 100.0% | 0.00s | -785.7% |
| sqrt | 0 | 100.0% | 0.00s | -717.5% |
| minimal | 0 | 100.0% | 0.00s | -721.9% |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### Case Arrival Trend
**Parameters:** `aggregation=W, mann_kendall`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 1 | 100.0% | 0.03s | 0.0% |
| optimized | 1 | 100.0% | 0.02s | -491.2% |
| sqrt | 1 | 100.0% | 0.02s | -446.7% |
| minimal | 1 | 100.0% | 0.02s | -450.6% |

#### Cluster (OPTICS)
**Parameters:** `algorithm=optics, min_samples=129, max_eps=1.73`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 454 | 100.0% | 5m 46.1s | 0.0% |
| optimized | 262 | 57.7% | 50.78s | 85.2% |
| sqrt | 189 | 41.6% | 13.60s | 96.0% |
| minimal | 109 | 24.0% | 2.48s | 99.2% |

#### Gap (Transition)
**Parameters:** `mode=transition, min_samples=5`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 724 | 100.0% | 1m 1.2s | 0.0% |
| optimized | 215 | 29.7% | 20.29s | 66.2% |
| sqrt | 83 | 11.5% | 8.23s | 85.9% |
| minimal | 29 | 4.0% | 2.11s | 95.9% |

#### Outlier Detection
**Parameters:** `isolation_forest + statistical`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 7515 | 100.0% | 1.59s | 0.0% |
| optimized | 2496 | 33.2% | 0.59s | 39.1% |
| sqrt | 999 | 13.3% | 0.32s | 57.5% |
| minimal | 250 | 3.3% | 0.17s | 66.7% |

#### Sequence Detection
**Parameters:** `min_support=30, prefixspan`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 0 | 100.0% | 0.21s | 0.0% |
| optimized | 0 | 100.0% | 0.10s | -102.1% |
| sqrt | 0 | 100.0% | 0.06s | -73.5% |
| minimal | 0 | 100.0% | 0.03s | -61.2% |

#### Temporal Cluster
**Parameters:** `min_cluster_size=129, temporal_eps=1.00`

| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |
|:---|---:|---:|---:|---:|
| full | 540 | 100.0% | 0.82s | 0.0% |
| optimized | 256 | 47.4% | 0.23s | 26.8% |
| sqrt | 123 | 22.8% | 0.08s | 47.3% |
| minimal | 63 | 11.7% | 0.03s | 53.6% |
