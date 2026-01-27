# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 20:34:08

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 2.99s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 3.82s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 9.35s |
| Hospital_log.xes | 83.0 | 150,291 | 11.53s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 13110.0 | 2.329s | 1.0x |
| **optimized** | 19.1% | 2500.0 | 0.509s | 4.6x |
| **sqrt** | 7.6% | 999.0 | 0.286s | 8.1x |
| **minimal** | 1.9% | 250.0 | 0.154s | 15.1x |

### DomesticDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 2819.0 | 0.593s | 1.0x |
| **optimized** | 70.6% | 1994.0 | 0.469s | 1.3x |
| **sqrt** | 5.1% | 146.0 | 0.135s | 4.4x |
| **minimal** | 1.5% | 44.0 | 0.111s | 5.3x |

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 7515.0 | 1.543s | 1.0x |
| **optimized** | 33.2% | 2496.0 | 0.557s | 2.8x |
| **sqrt** | 13.3% | 999.0 | 0.321s | 4.8x |
| **minimal** | 3.3% | 250.0 | 0.165s | 9.4x |

### InternationalDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 3603.0 | 0.773s | 1.0x |
| **optimized** | 69.3% | 2500.0 | 0.586s | 1.3x |
| **sqrt** | 14.5% | 525.0 | 0.230s | 3.4x |
| **minimal** | 6.9% | 250.0 | 0.165s | 4.7x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.32s | 1.0x |
| minimal | 250 | 1.9% | 0.15s | 15.1x |
| sqrt | 999 | 7.6% | 0.28s | 8.2x |
| optimized | 2500 | 19.1% | 0.52s | 4.5x |

#### DomesticDeclarations.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.59s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.3x |
| sqrt | 146 | 5.2% | 0.13s | 4.4x |
| optimized | 1994 | 70.7% | 0.47s | 1.3x |

#### Hospital_log.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.56s | 1.0x |
| minimal | 250 | 3.3% | 0.16s | 9.5x |
| sqrt | 999 | 13.3% | 0.34s | 4.6x |
| optimized | 2496 | 33.2% | 0.56s | 2.8x |

#### InternationalDeclarations.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.77s | 1.0x |
| minimal | 250 | 6.9% | 0.16s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.3x |
| optimized | 2500 | 69.4% | 0.59s | 1.3x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.34s | 1.0x |
| minimal | 250 | 1.9% | 0.15s | 15.3x |
| sqrt | 999 | 7.6% | 0.29s | 8.0x |
| optimized | 2500 | 19.1% | 0.51s | 4.6x |

#### DomesticDeclarations.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.59s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.3x |
| sqrt | 146 | 5.2% | 0.13s | 4.4x |
| optimized | 1994 | 70.7% | 0.47s | 1.3x |

#### Hospital_log.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.53s | 1.0x |
| minimal | 250 | 3.3% | 0.16s | 9.3x |
| sqrt | 999 | 13.3% | 0.31s | 4.9x |
| optimized | 2496 | 33.2% | 0.55s | 2.8x |

#### InternationalDeclarations.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.78s | 1.0x |
| minimal | 250 | 6.9% | 0.17s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.5x |
| optimized | 2500 | 69.4% | 0.59s | 1.3x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 13110 | 100% | 2.32s | 1.0x |
| minimal | 250 | 1.9% | 0.15s | 15.1x |
| sqrt | 999 | 7.6% | 0.28s | 8.2x |
| optimized | 2500 | 19.1% | 0.50s | 4.6x |

#### DomesticDeclarations.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2819 | 100% | 0.59s | 1.0x |
| minimal | 44 | 1.6% | 0.11s | 5.4x |
| sqrt | 146 | 5.2% | 0.14s | 4.4x |
| optimized | 1994 | 70.7% | 0.47s | 1.3x |

#### Hospital_log.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7515 | 100% | 1.54s | 1.0x |
| minimal | 250 | 3.3% | 0.17s | 9.3x |
| sqrt | 999 | 13.3% | 0.31s | 4.9x |
| optimized | 2496 | 33.2% | 0.56s | 2.8x |

#### InternationalDeclarations.xes

**Outlier Detection**

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 3603 | 100% | 0.77s | 1.0x |
| minimal | 250 | 6.9% | 0.16s | 4.7x |
| sqrt | 525 | 14.6% | 0.23s | 3.3x |
| optimized | 2500 | 69.4% | 0.58s | 1.3x |
