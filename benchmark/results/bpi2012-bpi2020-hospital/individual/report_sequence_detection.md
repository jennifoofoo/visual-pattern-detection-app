# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 02:23:11

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.08s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 5.52s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 12.32s |
| Hospital_log.xes | 83.0 | 150,291 | 16.50s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 262,200 / 262,200 | 100.0% | 6567.7 | 16.511s | 1.0x |
| **optimized** | 49,988 / 262,200 | 19.1% | 4952.3 | 3.631s | 4.5x |
| **sqrt** | 19,976 / 262,200 | 7.6% | 4893.7 | 9.983s | 1.7x |
| **minimal** | 4,999 / 262,200 | 1.9% | 3266.7 | 0.874s | 18.9x |

### DomesticDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 56,437 / 56,437 | 100.0% | 6659.7 | 1.996s | 1.0x |
| **optimized** | 39,870 / 56,437 | 70.6% | 6300.0 | 2.016s | 1.0x |
| **sqrt** | 2,904 / 56,437 | 5.1% | 1826.7 | 0.274s | 7.3x |
| **minimal** | 874 / 56,437 | 1.5% | 780.3 | 0.094s | 21.3x |

### Hospital_log.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 150,291 / 150,291 | 100.0% | 257.7 | 0.888s | 1.0x |
| **optimized** | 49,913 / 150,291 | 33.2% | 148.3 | 0.409s | 2.2x |
| **sqrt** | 19,967 / 150,291 | 13.3% | 90.7 | 0.235s | 3.8x |
| **minimal** | 4,991 / 150,291 | 3.3% | 135.7 | 0.102s | 8.7x |

### InternationalDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 72,151 / 72,151 | 100.0% | 5304.0 | 3.199s | 1.0x |
| **optimized** | 49,994 / 72,151 | 69.3% | 5519.0 | 2.600s | 1.2x |
| **sqrt** | 10,497 / 72,151 | 14.5% | 3014.3 | 0.668s | 4.8x |
| **minimal** | 4,987 / 72,151 | 6.9% | 2660.7 | 0.423s | 7.6x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2930 | 100% | 15.08s | 1.0x |
| minimal | 2429 | 82.9% | 0.55s | 27.3x |
| sqrt | 4181 | 142.7% | 1.17s | 12.9x |
| optimized | 5780 | 197.3% | 3.26s | 4.6x |

#### DomesticDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9979 | 100% | 2.28s | 1.0x |
| minimal | 369 | 3.7% | 0.09s | 24.4x |
| sqrt | 1105 | 11.1% | 0.24s | 9.3x |
| optimized | 9886 | 99.1% | 2.13s | 1.1x |

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.94s | 1.0x |
| minimal | 0 | 100% | 0.06s | 15.2x |
| sqrt | 0 | 100% | 0.27s | 3.5x |
| optimized | 0 | 100% | 0.41s | 2.3x |

#### InternationalDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 8388 | 100% | 2.44s | 1.0x |
| minimal | 1871 | 22.3% | 0.33s | 7.4x |
| sqrt | 3987 | 47.5% | 0.70s | 3.5x |
| optimized | 9778 | 116.6% | 2.34s | 1.0x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7557 | 100% | 18.98s | 1.0x |
| minimal | 952 | 12.6% | 0.45s | 42.3x |
| sqrt | 4326 | 57.2% | 1.55s | 12.3x |
| optimized | 2038 | 27.0% | 3.79s | 5.0x |

#### DomesticDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10000 | 100% | 3.70s | 1.0x |
| minimal | 1972 | 19.7% | 0.19s | 20.0x |
| sqrt | 4375 | 43.8% | 0.57s | 6.4x |
| optimized | 9014 | 90.1% | 3.91s | 0.9x |

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 773 | 100% | 1.43s | 1.0x |
| minimal | 407 | 52.7% | 0.20s | 7.2x |
| sqrt | 272 | 35.2% | 0.35s | 4.1x |
| optimized | 445 | 57.6% | 0.67s | 2.1x |

#### InternationalDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7524 | 100% | 7.14s | 1.0x |
| minimal | 6111 | 81.2% | 0.94s | 7.6x |
| sqrt | 5056 | 67.2% | 1.30s | 5.5x |
| optimized | 6779 | 90.1% | 5.45s | 1.3x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9216 | 100% | 15.47s | 1.0x |
| minimal | 6419 | 69.7% | 1.62s | 9.5x |
| sqrt | 6174 | 67.0% | 27.23s | 0.6x |
| optimized | 7039 | 76.4% | 3.83s | 4.0x |

#### DomesticDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.01s | 1.0x |
| minimal | 0 | 100% | 0.00s | 5.1x |
| sqrt | 0 | 100% | 0.00s | 4.4x |
| optimized | 0 | 100% | 0.01s | 1.4x |

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.29s | 1.0x |
| minimal | 0 | 100% | 0.04s | 6.7x |
| sqrt | 0 | 100% | 0.09s | 3.4x |
| optimized | 0 | 100% | 0.15s | 2.0x |

#### InternationalDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.01s | 1.0x |
| minimal | 0 | 100% | 0.00s | 5.1x |
| sqrt | 0 | 100% | 0.00s | 3.2x |
| optimized | 0 | 100% | 0.01s | 1.7x |
