# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 02:14:29

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.55s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 3.86s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 9.25s |
| Hospital_log.xes | 83.0 | 150,291 | 14.21s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 262,200 / 262,200 | 100.0% | 13110.0 | 2.346s | 1.0x |
| **optimized** | 49,988 / 262,200 | 19.1% | 2500.0 | 0.568s | 4.1x |
| **sqrt** | 19,976 / 262,200 | 7.6% | 999.0 | 0.285s | 8.2x |
| **minimal** | 4,999 / 262,200 | 1.9% | 250.0 | 0.159s | 14.8x |

### DomesticDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 56,437 / 56,437 | 100.0% | 2819.0 | 0.656s | 1.0x |
| **optimized** | 39,870 / 56,437 | 70.6% | 1994.0 | 0.475s | 1.4x |
| **sqrt** | 2,904 / 56,437 | 5.1% | 146.0 | 0.135s | 4.8x |
| **minimal** | 874 / 56,437 | 1.5% | 44.0 | 0.112s | 5.8x |

### Hospital_log.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 150,291 / 150,291 | 100.0% | 7515.0 | 1.891s | 1.0x |
| **optimized** | 49,913 / 150,291 | 33.2% | 2496.0 | 0.785s | 2.4x |
| **sqrt** | 19,967 / 150,291 | 13.3% | 999.0 | 0.397s | 4.8x |
| **minimal** | 4,991 / 150,291 | 3.3% | 250.0 | 0.187s | 10.1x |

### InternationalDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 72,151 / 72,151 | 100.0% | 3603.0 | 0.773s | 1.0x |
| **optimized** | 49,994 / 72,151 | 69.3% | 2500.0 | 0.592s | 1.3x |
| **sqrt** | 10,497 / 72,151 | 14.5% | 525.0 | 0.228s | 3.4x |
| **minimal** | 4,987 / 72,151 | 6.9% | 250.0 | 0.166s | 4.7x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.37s | 1.0x |
| minimal | 250 | 1.9% | 0.15s | 15.4x |
| sqrt | 999 | 7.6% | 0.29s | 8.3x |
| optimized | 2500 | 19.1% | 0.68s | 3.5x |

#### DomesticDeclarations.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.74s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 6.7x |
| sqrt | 146 | 5.2% | 0.13s | 5.5x |
| optimized | 1994 | 70.7% | 0.47s | 1.6x |

#### Hospital_log.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 2.04s | 1.0x |
| minimal | 250 | 3.3% | 0.17s | 12.3x |
| sqrt | 999 | 13.3% | 0.43s | 4.8x |
| optimized | 2496 | 33.2% | 0.81s | 2.5x |

#### InternationalDeclarations.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.79s | 1.0x |
| minimal | 250 | 6.9% | 0.17s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.5x |
| optimized | 2500 | 69.4% | 0.60s | 1.3x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.32s | 1.0x |
| minimal | 250 | 1.9% | 0.17s | 14.0x |
| sqrt | 999 | 7.6% | 0.28s | 8.2x |
| optimized | 2500 | 19.1% | 0.51s | 4.6x |

#### DomesticDeclarations.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.62s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.5x |
| sqrt | 146 | 5.2% | 0.14s | 4.6x |
| optimized | 1994 | 70.7% | 0.47s | 1.3x |

#### Hospital_log.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.57s | 1.0x |
| minimal | 250 | 3.3% | 0.23s | 6.9x |
| sqrt | 999 | 13.3% | 0.37s | 4.2x |
| optimized | 2496 | 33.2% | 0.75s | 2.1x |

#### InternationalDeclarations.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.77s | 1.0x |
| minimal | 250 | 6.9% | 0.16s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.3x |
| optimized | 2500 | 69.4% | 0.59s | 1.3x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.34s | 1.0x |
| minimal | 250 | 1.9% | 0.16s | 14.9x |
| sqrt | 999 | 7.6% | 0.29s | 8.2x |
| optimized | 2500 | 19.1% | 0.51s | 4.6x |

#### DomesticDeclarations.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.61s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.4x |
| sqrt | 146 | 5.2% | 0.14s | 4.5x |
| optimized | 1994 | 70.7% | 0.48s | 1.3x |

#### Hospital_log.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 2.07s | 1.0x |
| minimal | 250 | 3.3% | 0.17s | 12.3x |
| sqrt | 999 | 13.3% | 0.39s | 5.3x |
| optimized | 2496 | 33.2% | 0.79s | 2.6x |

#### InternationalDeclarations.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.77s | 1.0x |
| minimal | 250 | 6.9% | 0.17s | 4.6x |
| sqrt | 525 | 14.6% | 0.23s | 3.4x |
| optimized | 2500 | 69.4% | 0.59s | 1.3x |
