# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 02:10:59

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.19s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 4.99s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 11.77s |
| Hospital_log.xes | 83.0 | 150,291 | 14.97s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 262,200 / 262,200 | 100.0% | 0.0 | 0.038s | 1.0x |
| **optimized** | 49,988 / 262,200 | 19.1% | 0.0 | 0.016s | 2.3x |
| **sqrt** | 19,976 / 262,200 | 7.6% | 0.0 | 0.006s | 5.9x |
| **minimal** | 4,999 / 262,200 | 1.9% | 0.0 | 0.005s | 7.3x |

### DomesticDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 56,437 / 56,437 | 100.0% | 1.0 | 0.022s | 1.0x |
| **optimized** | 39,870 / 56,437 | 70.6% | 1.0 | 0.020s | 1.1x |
| **sqrt** | 2,904 / 56,437 | 5.1% | 1.0 | 0.011s | 2.0x |
| **minimal** | 874 / 56,437 | 1.5% | 0.0 | 0.010s | 2.1x |

### Hospital_log.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 150,291 / 150,291 | 100.0% | 1.0 | 0.043s | 1.0x |
| **optimized** | 49,913 / 150,291 | 33.2% | 1.0 | 0.036s | 1.2x |
| **sqrt** | 19,967 / 150,291 | 13.3% | 1.0 | 0.031s | 1.4x |
| **minimal** | 4,991 / 150,291 | 3.3% | 1.0 | 0.037s | 1.2x |

### InternationalDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 72,151 / 72,151 | 100.0% | 1.0 | 0.027s | 1.0x |
| **optimized** | 49,994 / 72,151 | 69.3% | 1.0 | 0.022s | 1.2x |
| **sqrt** | 10,497 / 72,151 | 14.5% | 1.0 | 0.016s | 1.7x |
| **minimal** | 4,987 / 72,151 | 6.9% | 0.0 | 0.013s | 2.0x |

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

#### DomesticDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.2x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.04s | 1.0x |
| minimal | 1 | 100.0% | 0.04s | 0.9x |
| sqrt | 1 | 100.0% | 0.04s | 0.9x |
| optimized | 1 | 100.0% | 0.03s | 1.4x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.0x |
| sqrt | 1 | 100.0% | 0.02s | 1.2x |
| optimized | 1 | 100.0% | 0.03s | 0.8x |

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

#### DomesticDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.1x |
| sqrt | 1 | 100.0% | 0.01s | 2.1x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.05s | 1.0x |
| minimal | 1 | 100.0% | 0.04s | 1.3x |
| sqrt | 1 | 100.0% | 0.03s | 1.4x |
| optimized | 1 | 100.0% | 0.04s | 1.1x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.1x |
| sqrt | 1 | 100.0% | 0.01s | 2.1x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

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

#### DomesticDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.1x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.03s | 0.8x |

#### Hospital_log.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.05s | 1.0x |
| minimal | 1 | 100.0% | 0.04s | 1.3x |
| sqrt | 1 | 100.0% | 0.02s | 2.0x |
| optimized | 1 | 100.0% | 0.04s | 1.1x |

#### InternationalDeclarations.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.04s | 1.0x |
| minimal | 0 | 0.0% | 0.02s | 2.0x |
| sqrt | 1 | 100.0% | 0.02s | 1.9x |
| optimized | 1 | 100.0% | 0.02s | 1.7x |
