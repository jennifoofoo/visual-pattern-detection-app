# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 01:53:55

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.97s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 3.85s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 9.30s |
| Hospital_log.xes | 83.0 | 150,291 | 12.44s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 6567.7 | 11.259s | 1.0x |
| **optimized** | 19.1% | 4952.3 | 2.524s | 4.5x |
| **sqrt** | 7.6% | 4893.7 | 6.775s | 1.7x |
| **minimal** | 1.9% | 3266.7 | 0.579s | 19.4x |

### DomesticDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 6658.7 | 2.046s | 1.0x |
| **optimized** | 70.6% | 6320.0 | 1.470s | 1.4x |
| **sqrt** | 5.1% | 1826.7 | 0.208s | 9.9x |
| **minimal** | 1.5% | 780.3 | 0.083s | 24.6x |

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 257.7 | 0.704s | 1.0x |
| **optimized** | 33.2% | 148.3 | 0.332s | 2.1x |
| **sqrt** | 13.3% | 90.7 | 0.176s | 4.0x |
| **minimal** | 3.3% | 135.7 | 0.078s | 9.0x |

### InternationalDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 3852.0 | 2.479s | 1.0x |
| **optimized** | 69.3% | 4786.7 | 1.814s | 1.4x |
| **sqrt** | 14.5% | 3014.3 | 0.511s | 4.8x |
| **minimal** | 6.9% | 2660.7 | 0.302s | 8.2x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2930 | 100% | 10.77s | 1.0x |
| minimal | 2429 | 82.9% | 0.36s | 30.0x |
| sqrt | 4181 | 142.7% | 1.00s | 10.7x |
| optimized | 5780 | 197.3% | 2.35s | 4.6x |

#### DomesticDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9976 | 100% | 2.14s | 1.0x |
| minimal | 369 | 3.7% | 0.07s | 31.8x |
| sqrt | 1105 | 11.1% | 0.16s | 13.2x |
| optimized | 9946 | 99.7% | 1.68s | 1.3x |

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.72s | 1.0x |
| minimal | 0 | 100% | 0.06s | 11.6x |
| sqrt | 0 | 100% | 0.21s | 3.5x |
| optimized | 0 | 100% | 0.38s | 1.9x |

#### InternationalDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 4032 | 100% | 2.28s | 1.0x |
| minimal | 1871 | 46.4% | 0.25s | 9.0x |
| sqrt | 3987 | 98.9% | 0.55s | 4.2x |
| optimized | 7581 | 188.0% | 1.77s | 1.3x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7557 | 100% | 12.77s | 1.0x |
| minimal | 952 | 12.6% | 0.29s | 43.6x |
| sqrt | 4326 | 57.2% | 1.24s | 10.3x |
| optimized | 2038 | 27.0% | 2.63s | 4.9x |

#### DomesticDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 10000 | 100% | 3.98s | 1.0x |
| minimal | 1972 | 19.7% | 0.18s | 22.1x |
| sqrt | 4375 | 43.8% | 0.46s | 8.7x |
| optimized | 9014 | 90.1% | 2.72s | 1.5x |

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 773 | 100% | 1.20s | 1.0x |
| minimal | 407 | 52.7% | 0.14s | 8.3x |
| sqrt | 272 | 35.2% | 0.26s | 4.7x |
| optimized | 445 | 57.6% | 0.50s | 2.4x |

#### InternationalDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7524 | 100% | 5.14s | 1.0x |
| minimal | 6111 | 81.2% | 0.65s | 7.9x |
| sqrt | 5056 | 67.2% | 0.98s | 5.2x |
| optimized | 6779 | 90.1% | 3.66s | 1.4x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9216 | 100% | 10.23s | 1.0x |
| minimal | 6419 | 69.7% | 1.09s | 9.4x |
| sqrt | 6174 | 67.0% | 18.08s | 0.6x |
| optimized | 7039 | 76.4% | 2.60s | 3.9x |

#### DomesticDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.01s | 1.0x |
| minimal | 0 | 100% | 0.00s | 5.0x |
| sqrt | 0 | 100% | 0.00s | 4.6x |
| optimized | 0 | 100% | 0.01s | 1.4x |

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.19s | 1.0x |
| minimal | 0 | 100% | 0.03s | 7.0x |
| sqrt | 0 | 100% | 0.06s | 3.1x |
| optimized | 0 | 100% | 0.12s | 1.7x |

#### InternationalDeclarations.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.01s | 1.0x |
| minimal | 0 | 100% | 0.00s | 5.6x |
| sqrt | 0 | 100% | 0.00s | 4.4x |
| optimized | 0 | 100% | 0.01s | 1.1x |
