# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 02:50:03

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.95s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 5.36s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 11.79s |
| Hospital_log.xes | 83.0 | 150,291 | 16.10s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 262,200 / 262,200 | 100.0% | 10589.0 | 234.910s | 1.0x |
| **optimized** | 49,988 / 262,200 | 19.1% | 1879.0 | 15.023s | 15.6x |
| **sqrt** | 19,976 / 262,200 | 7.6% | 722.0 | 4.548s | 51.6x |
| **minimal** | 4,999 / 262,200 | 1.9% | 189.0 | 0.940s | 249.8x |

### DomesticDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 56,437 / 56,437 | 100.0% | 1901.0 | 16.211s | 1.0x |
| **optimized** | 39,870 / 56,437 | 70.6% | 1335.0 | 9.501s | 1.7x |
| **sqrt** | 2,904 / 56,437 | 5.1% | 113.0 | 0.397s | 40.8x |
| **minimal** | 874 / 56,437 | 1.5% | 32.0 | 0.140s | 116.2x |

### Hospital_log.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 150,291 / 150,291 | 100.0% | 724.0 | 98.581s | 1.0x |
| **optimized** | 49,913 / 150,291 | 33.2% | 215.0 | 29.812s | 3.3x |
| **sqrt** | 19,967 / 150,291 | 13.3% | 83.0 | 11.732s | 8.4x |
| **minimal** | 4,991 / 150,291 | 3.3% | 29.0 | 2.793s | 35.3x |

### InternationalDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 72,151 / 72,151 | 100.0% | 2643.0 | 24.921s | 1.0x |
| **optimized** | 49,994 / 72,151 | 69.3% | 1812.0 | 14.787s | 1.7x |
| **sqrt** | 10,497 / 72,151 | 14.5% | 403.0 | 2.076s | 12.0x |
| **minimal** | 4,987 / 72,151 | 6.9% | 178.0 | 0.849s | 29.4x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 5m 36.2s | 1.0x |
| minimal | 189 | 1.8% | 0.99s | 340.1x |
| sqrt | 722 | 6.8% | 5.33s | 63.1x |
| optimized | 1879 | 17.7% | 18.64s | 18.0x |

#### DomesticDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 20.80s | 1.0x |
| minimal | 32 | 1.7% | 0.11s | 196.2x |
| sqrt | 113 | 5.9% | 0.44s | 47.5x |
| optimized | 1335 | 70.2% | 11.71s | 1.8x |

#### Hospital_log.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 44.7s | 1.0x |
| minimal | 29 | 4.0% | 3.12s | 33.6x |
| sqrt | 83 | 11.5% | 12.16s | 8.6x |
| optimized | 215 | 29.7% | 29.08s | 3.6x |

#### InternationalDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 31.81s | 1.0x |
| minimal | 178 | 6.7% | 0.89s | 35.6x |
| sqrt | 403 | 15.2% | 2.24s | 14.2x |
| optimized | 1812 | 68.6% | 18.31s | 1.7x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 5m 25.2s | 1.0x |
| minimal | 189 | 1.8% | 0.98s | 331.4x |
| sqrt | 722 | 6.8% | 5.05s | 64.4x |
| optimized | 1879 | 17.7% | 18.19s | 17.9x |

#### DomesticDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 19.15s | 1.0x |
| minimal | 32 | 1.7% | 0.16s | 120.4x |
| sqrt | 113 | 5.9% | 0.42s | 45.4x |
| optimized | 1335 | 70.2% | 11.01s | 1.7x |

#### Hospital_log.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 41.9s | 1.0x |
| minimal | 29 | 4.0% | 2.91s | 35.0x |
| sqrt | 83 | 11.5% | 11.43s | 8.9x |
| optimized | 215 | 29.7% | 30.95s | 3.3x |

#### InternationalDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 30.90s | 1.0x |
| minimal | 178 | 6.7% | 0.84s | 36.7x |
| sqrt | 403 | 15.2% | 2.26s | 13.7x |
| optimized | 1812 | 68.6% | 17.67s | 1.7x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 43.40s | 1.0x |
| minimal | 189 | 1.8% | 0.85s | 51.0x |
| sqrt | 722 | 6.8% | 3.27s | 13.3x |
| optimized | 1879 | 17.7% | 8.23s | 5.3x |

#### DomesticDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 8.68s | 1.0x |
| minimal | 32 | 1.7% | 0.15s | 56.5x |
| sqrt | 113 | 5.9% | 0.33s | 26.2x |
| optimized | 1335 | 70.2% | 5.78s | 1.5x |

#### Hospital_log.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 29.2s | 1.0x |
| minimal | 29 | 4.0% | 2.35s | 37.9x |
| sqrt | 83 | 11.5% | 11.61s | 7.7x |
| optimized | 215 | 29.7% | 29.40s | 3.0x |

#### InternationalDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 12.06s | 1.0x |
| minimal | 178 | 6.7% | 0.81s | 14.9x |
| sqrt | 403 | 15.2% | 1.73s | 7.0x |
| optimized | 1812 | 68.6% | 8.38s | 1.4x |
