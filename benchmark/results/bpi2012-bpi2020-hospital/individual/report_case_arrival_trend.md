# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 20:30:53

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 2.98s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 3.84s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 9.22s |
| Hospital_log.xes | 83.0 | 150,291 | 11.52s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 0.0 | 0.025s | 1.0x |
| **optimized** | 19.1% | 0.0 | 0.011s | 2.3x |
| **sqrt** | 7.6% | 0.0 | 0.005s | 5.1x |
| **minimal** | 1.9% | 0.0 | 0.004s | 6.4x |

### DomesticDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 1.0 | 0.022s | 1.0x |
| **optimized** | 70.6% | 1.0 | 0.017s | 1.2x |
| **sqrt** | 5.1% | 1.0 | 0.011s | 2.0x |
| **minimal** | 1.5% | 0.0 | 0.010s | 2.1x |

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 1.0 | 0.027s | 1.0x |
| **optimized** | 33.2% | 1.0 | 0.026s | 1.0x |
| **sqrt** | 13.3% | 1.0 | 0.022s | 1.2x |
| **minimal** | 3.3% | 1.0 | 0.021s | 1.3x |

### InternationalDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 1.0 | 0.021s | 1.0x |
| **optimized** | 69.3% | 1.0 | 0.018s | 1.2x |
| **sqrt** | 14.5% | 1.0 | 0.011s | 1.9x |
| **minimal** | 6.9% | 0.0 | 0.011s | 2.0x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.02s | 1.0x |
| minimal | 0 | 100% | 0.00s | 6.3x |
| sqrt | 0 | 100% | 0.00s | 5.1x |
| optimized | 0 | 100% | 0.01s | 2.3x |

#### DomesticDeclarations.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.1x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

#### Hospital_log.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.3x |
| sqrt | 1 | 100.0% | 0.02s | 1.3x |
| optimized | 1 | 100.0% | 0.03s | 0.8x |

#### InternationalDeclarations.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.0x |
| sqrt | 1 | 100.0% | 0.01s | 1.9x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.03s | 1.0x |
| minimal | 0 | 100% | 0.00s | 6.5x |
| sqrt | 0 | 100% | 0.00s | 5.2x |
| optimized | 0 | 100% | 0.01s | 2.3x |

#### DomesticDeclarations.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.2x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

#### Hospital_log.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.3x |
| sqrt | 1 | 100.0% | 0.02s | 1.2x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

#### InternationalDeclarations.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.0x |
| sqrt | 1 | 100.0% | 0.01s | 1.9x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.02s | 1.0x |
| minimal | 0 | 100% | 0.00s | 6.4x |
| sqrt | 0 | 100% | 0.00s | 5.0x |
| optimized | 0 | 100% | 0.01s | 2.3x |

#### DomesticDeclarations.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 1.9x |
| sqrt | 1 | 100.0% | 0.01s | 2.1x |
| optimized | 1 | 100.0% | 0.02s | 1.3x |

#### Hospital_log.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.03s | 1.0x |
| minimal | 1 | 100.0% | 0.02s | 1.3x |
| sqrt | 1 | 100.0% | 0.02s | 1.2x |
| optimized | 1 | 100.0% | 0.02s | 1.2x |

#### InternationalDeclarations.xes

**Case Arrival Trend**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 0.02s | 1.0x |
| minimal | 0 | 0.0% | 0.01s | 2.0x |
| sqrt | 1 | 100.0% | 0.01s | 2.0x |
| optimized | 1 | 100.0% | 0.02s | 1.1x |
