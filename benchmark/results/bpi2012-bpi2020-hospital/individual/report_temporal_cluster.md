# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 20:37:05

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 2.99s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 3.76s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 9.18s |
| Hospital_log.xes | 83.0 | 150,291 | 11.53s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 0.0 | 0.644s | 1.0x |
| **optimized** | 19.1% | 0.0 | 0.113s | 5.7x |
| **sqrt** | 7.6% | 0.0 | 0.049s | 13.0x |
| **minimal** | 1.9% | 0.0 | 0.018s | 35.5x |

### DomesticDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 33.0 | 0.152s | 1.0x |
| **optimized** | 70.6% | 20.3 | 0.107s | 1.4x |
| **sqrt** | 5.1% | 0.3 | 0.012s | 12.2x |
| **minimal** | 1.5% | 0.0 | 0.737s | 0.2x |

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 360.0 | 0.722s | 1.0x |
| **optimized** | 33.2% | 170.7 | 0.240s | 3.0x |
| **sqrt** | 13.3% | 82.0 | 0.106s | 6.8x |
| **minimal** | 3.3% | 42.0 | 0.039s | 18.5x |

### InternationalDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 3.7 | 0.198s | 1.0x |
| **optimized** | 69.3% | 2.3 | 0.133s | 1.5x |
| **sqrt** | 14.5% | 2.3 | 0.035s | 5.6x |
| **minimal** | 6.9% | 2.0 | 0.022s | 9.1x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 1.25s | 1.0x |
| minimal | 0 | 100% | 0.04s | 31.8x |
| sqrt | 0 | 100% | 0.10s | 12.4x |
| optimized | 0 | 100% | 0.23s | 5.5x |

#### DomesticDeclarations.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 50 | 100% | 0.31s | 1.0x |
| minimal | 0 | 0.0% | 2.20s | 0.1x |
| sqrt | 1 | 2.0% | 0.03s | 11.3x |
| optimized | 31 | 62.0% | 0.22s | 1.4x |

#### Hospital_log.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 1.37s | 1.0x |
| minimal | 63 | 11.7% | 0.09s | 15.0x |
| sqrt | 123 | 22.8% | 0.23s | 5.8x |
| optimized | 256 | 47.4% | 0.49s | 2.8x |

#### InternationalDeclarations.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7 | 100% | 0.40s | 1.0x |
| minimal | 4 | 57.1% | 0.05s | 8.1x |
| sqrt | 5 | 71.4% | 0.08s | 5.2x |
| optimized | 5 | 71.4% | 0.27s | 1.5x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.0x |
| sqrt | 0 | 100% | 0.00s | 1.1x |
| optimized | 0 | 100% | 0.00s | 1.1x |

#### DomesticDeclarations.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.2x |
| sqrt | 0 | 100% | 0.00s | 1.3x |
| optimized | 0 | 100% | 0.00s | 1.2x |

#### Hospital_log.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.1x |
| sqrt | 0 | 100% | 0.00s | 1.2x |
| optimized | 0 | 100% | 0.00s | 1.3x |

#### InternationalDeclarations.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 1.2x |
| sqrt | 0 | 100% | 0.00s | 1.1x |
| optimized | 0 | 100% | 0.00s | 0.7x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.68s | 1.0x |
| minimal | 0 | 100% | 0.02s | 45.3x |
| sqrt | 0 | 100% | 0.05s | 14.3x |
| optimized | 0 | 100% | 0.11s | 6.0x |

#### DomesticDeclarations.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 49 | 100% | 0.15s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 25.6x |
| sqrt | 0 | 0.0% | 0.01s | 14.8x |
| optimized | 30 | 61.2% | 0.10s | 1.4x |

#### Hospital_log.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 540 | 100% | 0.80s | 1.0x |
| minimal | 63 | 11.7% | 0.03s | 30.7x |
| sqrt | 123 | 22.8% | 0.08s | 9.5x |
| optimized | 256 | 47.4% | 0.23s | 3.5x |

#### InternationalDeclarations.xes

**Temporal Cluster**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 4 | 100% | 0.20s | 1.0x |
| minimal | 2 | 50.0% | 0.02s | 12.1x |
| sqrt | 2 | 50.0% | 0.03s | 6.6x |
| optimized | 2 | 50.0% | 0.13s | 1.5x |
