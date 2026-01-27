# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 21:23:09

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.37s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 3.82s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 10.19s |
| Hospital_log.xes | 83.0 | 150,291 | 12.22s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 10589.0 | 156.912s | 1.0x |
| **optimized** | 19.1% | 1879.0 | 10.668s | 14.7x |
| **sqrt** | 7.6% | 722.0 | 3.271s | 48.0x |
| **minimal** | 1.9% | 189.0 | 0.711s | 220.7x |

### DomesticDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 1901.0 | 12.179s | 1.0x |
| **optimized** | 70.6% | 1335.0 | 7.165s | 1.7x |
| **sqrt** | 5.1% | 113.0 | 0.366s | 33.3x |
| **minimal** | 1.5% | 32.0 | 0.116s | 105.1x |

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 724.0 | 69.045s | 1.0x |
| **optimized** | 33.2% | 215.0 | 21.230s | 3.3x |
| **sqrt** | 13.3% | 83.0 | 8.375s | 8.2x |
| **minimal** | 3.3% | 29.0 | 2.082s | 33.2x |

### InternationalDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 2643.0 | 17.918s | 1.0x |
| **optimized** | 69.3% | 1812.0 | 10.588s | 1.7x |
| **sqrt** | 14.5% | 403.0 | 1.552s | 11.5x |
| **minimal** | 6.9% | 178.0 | 0.695s | 25.8x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 3m 42.2s | 1.0x |
| minimal | 189 | 1.8% | 0.76s | 293.1x |
| sqrt | 722 | 6.8% | 3.62s | 61.4x |
| optimized | 1879 | 17.7% | 13.03s | 17.0x |

#### DomesticDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 14.93s | 1.0x |
| minimal | 32 | 1.7% | 0.12s | 128.5x |
| sqrt | 113 | 5.9% | 0.37s | 40.6x |
| optimized | 1335 | 70.2% | 8.47s | 1.8x |

#### Hospital_log.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 12.4s | 1.0x |
| minimal | 29 | 4.0% | 2.11s | 34.4x |
| sqrt | 83 | 11.5% | 8.38s | 8.6x |
| optimized | 215 | 29.7% | 21.42s | 3.4x |

#### InternationalDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 22.57s | 1.0x |
| minimal | 178 | 6.7% | 0.70s | 32.1x |
| sqrt | 403 | 15.2% | 1.70s | 13.3x |
| optimized | 1812 | 68.6% | 12.81s | 1.8x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 3m 33.2s | 1.0x |
| minimal | 189 | 1.8% | 0.72s | 295.6x |
| sqrt | 722 | 6.8% | 3.57s | 59.7x |
| optimized | 1879 | 17.7% | 12.47s | 17.1x |

#### DomesticDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 14.85s | 1.0x |
| minimal | 32 | 1.7% | 0.13s | 116.5x |
| sqrt | 113 | 5.9% | 0.37s | 40.4x |
| optimized | 1335 | 70.2% | 8.27s | 1.8x |

#### Hospital_log.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 11.0s | 1.0x |
| minimal | 29 | 4.0% | 2.05s | 34.7x |
| sqrt | 83 | 11.5% | 8.41s | 8.4x |
| optimized | 215 | 29.7% | 21.45s | 3.3x |

#### InternationalDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 21.72s | 1.0x |
| minimal | 178 | 6.7% | 0.74s | 29.2x |
| sqrt | 403 | 15.2% | 1.62s | 13.4x |
| optimized | 1812 | 68.6% | 12.47s | 1.7x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10589 | 100% | 35.37s | 1.0x |
| minimal | 189 | 1.8% | 0.65s | 54.1x |
| sqrt | 722 | 6.8% | 2.63s | 13.5x |
| optimized | 1879 | 17.7% | 6.50s | 5.4x |

#### DomesticDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1901 | 100% | 6.75s | 1.0x |
| minimal | 32 | 1.7% | 0.10s | 65.0x |
| sqrt | 113 | 5.9% | 0.36s | 18.7x |
| optimized | 1335 | 70.2% | 4.76s | 1.4x |

#### Hospital_log.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 724 | 100% | 1m 3.7s | 1.0x |
| minimal | 29 | 4.0% | 2.09s | 30.5x |
| sqrt | 83 | 11.5% | 8.33s | 7.6x |
| optimized | 215 | 29.7% | 20.82s | 3.1x |

#### InternationalDeclarations.xes

**Gap (Transition)**
*Parameters: `mode=transition, min_samples=5`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2643 | 100% | 9.46s | 1.0x |
| minimal | 178 | 6.7% | 0.64s | 14.9x |
| sqrt | 403 | 15.2% | 1.33s | 7.1x |
| optimized | 1812 | 68.6% | 6.49s | 1.5x |
