# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 02:17:33

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.61s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 3.94s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 9.50s |
| Hospital_log.xes | 83.0 | 150,291 | 11.81s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 262,200 / 262,200 | 100.0% | 0.0 | 0.668s | 1.0x |
| **optimized** | 49,988 / 262,200 | 19.1% | 0.0 | 0.116s | 5.8x |
| **sqrt** | 19,976 / 262,200 | 7.6% | 0.0 | 0.088s | 7.6x |
| **minimal** | 4,999 / 262,200 | 1.9% | 0.0 | 0.018s | 36.3x |

### DomesticDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 56,437 / 56,437 | 100.0% | 33.0 | 0.226s | 1.0x |
| **optimized** | 39,870 / 56,437 | 70.6% | 20.3 | 0.109s | 2.1x |
| **sqrt** | 2,904 / 56,437 | 5.1% | 0.3 | 0.012s | 18.2x |
| **minimal** | 874 / 56,437 | 1.5% | 0.0 | 0.814s | 0.3x |

### Hospital_log.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 150,291 / 150,291 | 100.0% | 360.0 | 0.740s | 1.0x |
| **optimized** | 49,913 / 150,291 | 33.2% | 170.7 | 0.237s | 3.1x |
| **sqrt** | 19,967 / 150,291 | 13.3% | 82.0 | 0.103s | 7.2x |
| **minimal** | 4,991 / 150,291 | 3.3% | 42.0 | 0.040s | 18.3x |

### InternationalDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 72,151 / 72,151 | 100.0% | 3.7 | 0.200s | 1.0x |
| **optimized** | 49,994 / 72,151 | 69.3% | 2.3 | 0.139s | 1.4x |
| **sqrt** | 10,497 / 72,151 | 14.5% | 2.3 | 0.038s | 5.3x |
| **minimal** | 4,987 / 72,151 | 6.9% | 2.0 | 0.022s | 9.2x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 1.31s | 1.0x |
| minimal | 0 | 100% | 0.04s | 33.2x |
| sqrt | 0 | 100% | 0.10s | 12.7x |
| optimized | 0 | 100% | 0.23s | 5.6x |

#### DomesticDeclarations.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 50 | 100% | 0.45s | 1.0x |
| minimal | 0 | 0.0% | 2.43s | 0.2x |
| sqrt | 1 | 2.0% | 0.03s | 16.8x |
| optimized | 31 | 62.0% | 0.22s | 2.0x |

#### Hospital_log.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 1.40s | 1.0x |
| minimal | 63 | 11.7% | 0.09s | 14.9x |
| sqrt | 123 | 22.8% | 0.22s | 6.3x |
| optimized | 256 | 47.4% | 0.47s | 3.0x |

#### InternationalDeclarations.xes

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

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 0.9x |
| sqrt | 0 | 100% | 0.00s | 0.8x |
| optimized | 0 | 100% | 0.00s | 1.0x |

#### DomesticDeclarations.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.3x |
| sqrt | 0 | 100% | 0.00s | 1.2x |
| optimized | 0 | 100% | 0.00s | 1.2x |

#### Hospital_log.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 0.9x |
| sqrt | 0 | 100% | 0.00s | 1.0x |
| optimized | 0 | 100% | 0.00s | 1.0x |

#### InternationalDeclarations.xes

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

**Temporal Cluster**
*Parameters: `min_cluster_size=170, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.70s | 1.0x |
| minimal | 0 | 100% | 0.02s | 43.9x |
| sqrt | 0 | 100% | 0.16s | 4.3x |
| optimized | 0 | 100% | 0.11s | 6.1x |

#### DomesticDeclarations.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=79, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 49 | 100% | 0.22s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 31.5x |
| sqrt | 0 | 0.0% | 0.01s | 22.1x |
| optimized | 30 | 61.2% | 0.10s | 2.2x |

#### Hospital_log.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=129, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 0.82s | 1.0x |
| minimal | 63 | 11.7% | 0.03s | 30.5x |
| sqrt | 123 | 22.8% | 0.09s | 9.6x |
| optimized | 256 | 47.4% | 0.24s | 3.4x |

#### InternationalDeclarations.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=89, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 4 | 100% | 0.20s | 1.0x |
| minimal | 2 | 50.0% | 0.02s | 10.9x |
| sqrt | 2 | 50.0% | 0.03s | 5.8x |
| optimized | 2 | 50.0% | 0.14s | 1.4x |
