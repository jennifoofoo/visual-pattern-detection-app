# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 19:52:58

## 📈 Sampling Performance by Dataset

### BPI Challenge 2018.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 0.5 | 0.863s | 1.0x |

### BPI_Challenge_2019.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 424725.3 | 49.030s | 1.0x |
| **optimized** | 3.1% | 2383.9 | 0.765s | 64.1x |
| **sqrt** | 1.3% | 3461.9 | 0.594s | 82.5x |
| **minimal** | 0.3% | 308.6 | 0.161s | 303.9x |

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 2033.4 | 0.879s | 1.0x |
| **optimized** | 33.2% | 704.0 | 0.329s | 2.7x |
| **sqrt** | 13.3% | 293.2 | 0.173s | 5.1x |
| **minimal** | 3.3% | 107.2 | 0.084s | 10.5x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`

#### BPI Challenge 2018.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.19s | 1.0x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 1.50s | 1.0x |

#### BPI_Challenge_2019.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 8.95s | 1.0x |
| minimal | 1 | 100.0% | 0.20s | 44.9x |
| sqrt | 1 | 100.0% | 0.51s | 17.6x |
| optimized | 1 | 100.0% | 0.04s | 235.7x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 17.07s | 1.0x |
| minimal | 250 | 0.3% | 0.20s | 85.9x |
| sqrt | 1000 | 1.3% | 0.42s | 41.1x |
| optimized | 2500 | 3.1% | 0.66s | 26.0x |

**Sequence Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 17360 | 100% | 16.77s | 1.0x |
| minimal | 55 | 0.3% | 0.19s | 86.5x |
| sqrt | 411 | 2.4% | 0.61s | 27.4x |
| optimized | 279 | 1.6% | 1.08s | 15.6x |

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 443 | 100% | 31.60s | 1.0x |
| minimal | 1 | 0.2% | 0.07s | 460.0x |
| sqrt | 4 | 0.9% | 0.15s | 216.7x |
| optimized | 27 | 6.1% | 0.46s | 69.0x |

#### Hospital_log.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.4x |
| sqrt | 1 | 100.0% | 0.03s | 1.2x |
| optimized | 1 | 100.0% | 0.03s | 1.3x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.83s | 1.0x |
| minimal | 250 | 3.3% | 0.18s | 10.0x |
| sqrt | 999 | 13.3% | 0.34s | 5.4x |
| optimized | 2496 | 33.2% | 0.60s | 3.1x |

**Sequence Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.70s | 1.0x |
| minimal | 0 | 100% | 0.06s | 11.1x |
| sqrt | 0 | 100% | 0.20s | 3.5x |
| optimized | 0 | 100% | 0.38s | 1.9x |

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 1.83s | 1.0x |
| minimal | 63 | 11.7% | 0.13s | 14.1x |
| sqrt | 123 | 22.8% | 0.26s | 6.9x |
| optimized | 256 | 47.4% | 0.71s | 2.6x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`

#### BPI Challenge 2018.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.19s | 1.0x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 1.56s | 1.0x |

#### BPI_Challenge_2019.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 9.09s | 1.0x |
| minimal | 1 | 100.0% | 0.20s | 46.0x |
| sqrt | 1 | 100.0% | 0.60s | 15.1x |
| optimized | 1 | 100.0% | 0.03s | 289.9x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 17.58s | 1.0x |
| minimal | 250 | 0.3% | 0.18s | 98.8x |
| sqrt | 1000 | 1.3% | 0.38s | 45.8x |
| optimized | 2500 | 3.1% | 0.63s | 27.9x |

**Sequence Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 4545857 | 100% | 6m 41.9s | 1.0x |
| minimal | 2894 | 0.1% | 0.43s | 924.9x |
| sqrt | 38123 | 0.8% | 3.28s | 122.6x |
| optimized | 12783 | 0.3% | 3.96s | 101.5x |

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 0.8x |
| sqrt | 0 | 100% | 0.00s | 1.5x |
| optimized | 0 | 100% | 0.00s | 1.5x |

#### Hospital_log.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.5x |
| sqrt | 1 | 100.0% | 0.03s | 1.2x |
| optimized | 1 | 100.0% | 0.03s | 1.2x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.82s | 1.0x |
| minimal | 250 | 3.3% | 0.17s | 10.5x |
| sqrt | 999 | 13.3% | 0.33s | 5.6x |
| optimized | 2496 | 33.2% | 0.61s | 3.0x |

**Sequence Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 773 | 100% | 1.23s | 1.0x |
| minimal | 407 | 52.7% | 0.15s | 8.1x |
| sqrt | 272 | 35.2% | 0.26s | 4.7x |
| optimized | 445 | 57.6% | 0.51s | 2.4x |

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.2x |
| sqrt | 0 | 100% | 0.00s | 0.5x |
| optimized | 0 | 100% | 0.00s | 0.8x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`

#### BPI Challenge 2018.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.20s | 1.0x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 1.54s | 1.0x |

#### BPI_Challenge_2019.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 8.82s | 1.0x |
| minimal | 1 | 100.0% | 0.19s | 45.7x |
| sqrt | 1 | 100.0% | 0.52s | 16.9x |
| optimized | 1 | 100.0% | 0.03s | 278.3x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 17.59s | 1.0x |
| minimal | 250 | 0.3% | 0.18s | 99.9x |
| sqrt | 1000 | 1.3% | 0.38s | 46.6x |
| optimized | 2500 | 3.1% | 0.67s | 26.1x |

**Sequence Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 293214 | 100% | 41.05s | 1.0x |
| minimal | 0 | 0.0% | 0.07s | 627.7x |
| sqrt | 0 | 0.0% | 0.20s | 203.1x |
| optimized | 7995 | 2.7% | 1.45s | 28.3x |

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 436 | 100% | 17.96s | 1.0x |
| minimal | 0 | 0.0% | 0.03s | 595.3x |
| sqrt | 2 | 0.5% | 0.08s | 219.1x |
| optimized | 20 | 4.6% | 0.17s | 103.8x |

#### Hospital_log.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.6x |
| sqrt | 1 | 100.0% | 0.02s | 1.5x |
| optimized | 1 | 100.0% | 0.02s | 1.5x |

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.82s | 1.0x |
| minimal | 250 | 3.3% | 0.17s | 10.5x |
| sqrt | 999 | 13.3% | 0.40s | 4.5x |
| optimized | 2496 | 33.2% | 0.64s | 2.8x |

**Sequence Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.18s | 1.0x |
| minimal | 0 | 100% | 0.03s | 6.7x |
| sqrt | 0 | 100% | 0.06s | 2.8x |
| optimized | 0 | 100% | 0.10s | 1.8x |

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 1.04s | 1.0x |
| minimal | 63 | 11.7% | 0.04s | 26.9x |
| sqrt | 123 | 22.8% | 0.14s | 7.7x |
| optimized | 256 | 47.4% | 0.32s | 3.2x |
