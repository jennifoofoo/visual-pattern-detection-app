# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 04:23:07

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.51s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 5.12s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 12.56s |
| Hospital_log.xes | 83.0 | 150,291 | 16.15s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 262,200 / 262,200 | 100.0% | 676.7 | 983.226s | 1.0x |
| **optimized** | 49,988 / 262,200 | 19.1% | 270.0 | 54.785s | 17.9x |
| **sqrt** | 19,976 / 262,200 | 7.6% | 124.7 | 15.663s | 62.8x |
| **minimal** | 4,999 / 262,200 | 1.9% | 64.7 | 2.844s | 345.8x |

### DomesticDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 56,437 / 56,437 | 100.0% | 285.0 | 68.457s | 1.0x |
| **optimized** | 39,870 / 56,437 | 70.6% | 245.0 | 40.597s | 1.7x |
| **sqrt** | 2,904 / 56,437 | 5.1% | 64.3 | 1.546s | 44.3x |
| **minimal** | 874 / 56,437 | 1.5% | 47.7 | 0.458s | 149.5x |

### Hospital_log.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 150,291 / 150,291 | 100.0% | 439.3 | 375.679s | 1.0x |
| **optimized** | 49,913 / 150,291 | 33.2% | 276.0 | 58.562s | 6.4x |
| **sqrt** | 19,967 / 150,291 | 13.3% | 179.0 | 15.939s | 23.6x |
| **minimal** | 4,991 / 150,291 | 3.3% | 96.0 | 3.006s | 125.0x |

### InternationalDeclarations.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 72,151 / 72,151 | 100.0% | 302.3 | 99.277s | 1.0x |
| **optimized** | 49,994 / 72,151 | 69.3% | 229.7 | 60.747s | 1.6x |
| **sqrt** | 10,497 / 72,151 | 14.5% | 102.3 | 7.266s | 13.7x |
| **minimal** | 4,987 / 72,151 | 6.9% | 67.3 | 2.756s | 36.0x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=170, max_eps=1.56`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 750 | 100% | 18m 28.8s | 1.0x |
| minimal | 95 | 12.7% | 3.06s | 362.8x |
| sqrt | 175 | 23.3% | 14.89s | 74.5x |
| optimized | 286 | 38.1% | 54.16s | 20.5x |

#### DomesticDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=1.67`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 341 | 100% | 1m 15.7s | 1.0x |
| minimal | 44 | 12.9% | 0.51s | 147.4x |
| sqrt | 71 | 20.8% | 1.54s | 49.3x |
| optimized | 297 | 87.1% | 44.96s | 1.7x |

#### Hospital_log.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=49.50`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 399 | 100% | 5m 29.5s | 1.0x |
| minimal | 80 | 20.1% | 2.97s | 111.0x |
| sqrt | 165 | 41.4% | 15.42s | 21.4x |
| optimized | 269 | 67.4% | 53.54s | 6.2x |

#### InternationalDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=2.56`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 375 | 100% | 1m 39.0s | 1.0x |
| minimal | 82 | 21.9% | 2.73s | 36.2x |
| sqrt | 117 | 31.2% | 7.18s | 13.8x |
| optimized | 290 | 77.3% | 55.20s | 1.8x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=170, max_eps=955.11`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 522 | 100% | 14m 3.9s | 1.0x |
| minimal | 2 | 0.4% | 2.76s | 306.2x |
| sqrt | 21 | 4.0% | 17.67s | 47.8x |
| optimized | 220 | 42.1% | 50.67s | 16.7x |

#### DomesticDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=782.96`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 152 | 100% | 54.13s | 1.0x |
| minimal | 51 | 33.6% | 0.45s | 119.2x |
| sqrt | 38 | 25.0% | 1.56s | 34.6x |
| optimized | 125 | 82.2% | 32.75s | 1.7x |

#### Hospital_log.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=80.01`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 465 | 100% | 5m 29.3s | 1.0x |
| minimal | 99 | 21.3% | 3.02s | 109.1x |
| sqrt | 183 | 39.4% | 14.98s | 22.0x |
| optimized | 298 | 64.1% | 55.96s | 5.9x |

#### InternationalDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=456.72`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 161 | 100% | 1m 14.9s | 1.0x |
| minimal | 23 | 14.3% | 2.68s | 28.0x |
| sqrt | 56 | 34.8% | 7.09s | 10.6x |
| optimized | 73 | 45.3% | 58.05s | 1.3x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=164, max_eps=4.40`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 758 | 100% | 16m 37.0s | 1.0x |
| minimal | 97 | 12.8% | 2.72s | 366.7x |
| sqrt | 178 | 23.5% | 14.43s | 69.1x |
| optimized | 304 | 40.1% | 59.52s | 16.8x |

#### DomesticDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=0.17`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 362 | 100% | 1m 15.5s | 1.0x |
| minimal | 48 | 13.3% | 0.41s | 185.9x |
| sqrt | 84 | 23.2% | 1.54s | 49.2x |
| optimized | 313 | 86.5% | 44.09s | 1.7x |

#### Hospital_log.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=1.70`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 454 | 100% | 7m 48.3s | 1.0x |
| minimal | 109 | 24.0% | 3.03s | 154.4x |
| sqrt | 189 | 41.6% | 17.43s | 26.9x |
| optimized | 261 | 57.5% | 1m 6.2s | 7.1x |

#### InternationalDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=0.13`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 371 | 100% | 2m 3.9s | 1.0x |
| minimal | 97 | 26.1% | 2.86s | 43.4x |
| sqrt | 134 | 36.1% | 7.53s | 16.5x |
| optimized | 326 | 87.9% | 1m 9.0s | 1.8x |
