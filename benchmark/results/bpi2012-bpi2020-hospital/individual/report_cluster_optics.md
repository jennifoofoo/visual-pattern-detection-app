# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 22:41:10

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| DomesticDeclarations.xes | 19.5 | 56,437 | 3.02s |
| InternationalDeclarations.xes | 27.8 | 72,151 | 4.29s |
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 9.21s |
| Hospital_log.xes | 83.0 | 150,291 | 11.85s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 676.7 | 843.503s | 1.0x |
| **optimized** | 19.1% | 270.0 | 43.790s | 19.3x |
| **sqrt** | 7.6% | 124.7 | 12.414s | 67.9x |
| **minimal** | 1.9% | 64.7 | 2.502s | 337.1x |

### DomesticDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 285.0 | 53.283s | 1.0x |
| **optimized** | 70.6% | 245.0 | 34.319s | 1.6x |
| **sqrt** | 5.1% | 64.3 | 1.317s | 40.5x |
| **minimal** | 1.5% | 47.7 | 0.374s | 142.4x |

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 439.3 | 300.390s | 1.0x |
| **optimized** | 33.2% | 276.3 | 47.195s | 6.4x |
| **sqrt** | 13.3% | 179.0 | 12.826s | 23.4x |
| **minimal** | 3.3% | 96.0 | 2.436s | 123.3x |

### InternationalDeclarations.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 302.3 | 81.034s | 1.0x |
| **optimized** | 69.3% | 229.7 | 47.552s | 1.7x |
| **sqrt** | 14.5% | 102.3 | 5.997s | 13.5x |
| **minimal** | 6.9% | 67.3 | 2.400s | 33.8x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=170, max_eps=1.66`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 750 | 100% | 17m 7.5s | 1.0x |
| minimal | 95 | 12.7% | 2.57s | 399.9x |
| sqrt | 175 | 23.3% | 12.04s | 85.4x |
| optimized | 286 | 38.1% | 43.68s | 23.5x |

#### DomesticDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=1.66`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 341 | 100% | 58.27s | 1.0x |
| minimal | 44 | 12.9% | 0.36s | 160.3x |
| sqrt | 71 | 20.8% | 1.31s | 44.5x |
| optimized | 297 | 87.1% | 37.25s | 1.6x |

#### Hospital_log.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=48.52`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 399 | 100% | 4m 20.8s | 1.0x |
| minimal | 80 | 20.1% | 2.40s | 108.5x |
| sqrt | 165 | 41.4% | 12.16s | 21.4x |
| optimized | 269 | 67.4% | 43.37s | 6.0x |

#### InternationalDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=2.59`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 375 | 100% | 1m 17.2s | 1.0x |
| minimal | 82 | 21.9% | 2.43s | 31.7x |
| sqrt | 117 | 31.2% | 5.78s | 13.3x |
| optimized | 290 | 77.3% | 43.00s | 1.8x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=170, max_eps=938.87`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 522 | 100% | 11m 29.0s | 1.0x |
| minimal | 2 | 0.4% | 2.58s | 267.3x |
| sqrt | 21 | 4.0% | 13.67s | 50.4x |
| optimized | 220 | 42.1% | 40.56s | 17.0x |

#### DomesticDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=749.87`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 152 | 100% | 41.62s | 1.0x |
| minimal | 51 | 33.6% | 0.38s | 109.2x |
| sqrt | 38 | 25.0% | 1.29s | 32.1x |
| optimized | 125 | 82.2% | 30.48s | 1.4x |

#### Hospital_log.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=81.70`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 465 | 100% | 4m 33.1s | 1.0x |
| minimal | 99 | 21.3% | 2.39s | 114.4x |
| sqrt | 183 | 39.4% | 12.38s | 22.1x |
| optimized | 298 | 64.1% | 46.22s | 5.9x |

#### InternationalDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=444.95`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 161 | 100% | 1m 1.6s | 1.0x |
| minimal | 23 | 14.3% | 2.30s | 26.7x |
| sqrt | 56 | 34.8% | 5.82s | 10.6x |
| optimized | 73 | 45.3% | 45.99s | 1.3x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=164, max_eps=4.46`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 758 | 100% | 13m 34.0s | 1.0x |
| minimal | 97 | 12.8% | 2.36s | 345.0x |
| sqrt | 178 | 23.5% | 11.54s | 70.6x |
| optimized | 304 | 40.1% | 47.13s | 17.3x |

#### DomesticDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=79, max_eps=0.17`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 362 | 100% | 59.96s | 1.0x |
| minimal | 48 | 13.3% | 0.38s | 158.4x |
| sqrt | 84 | 23.2% | 1.35s | 44.6x |
| optimized | 313 | 86.5% | 35.22s | 1.7x |

#### Hospital_log.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=129, max_eps=1.70`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 454 | 100% | 6m 7.2s | 1.0x |
| minimal | 109 | 24.0% | 2.52s | 146.0x |
| sqrt | 189 | 41.6% | 13.93s | 26.4x |
| optimized | 262 | 57.7% | 52.00s | 7.1x |

#### InternationalDeclarations.xes

**Cluster (OPTICS)**
*Parameters: `algorithm=optics, min_samples=89, max_eps=0.13`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 371 | 100% | 1m 44.3s | 1.0x |
| minimal | 97 | 26.1% | 2.46s | 42.3x |
| sqrt | 134 | 36.1% | 6.39s | 16.3x |
| optimized | 326 | 87.9% | 53.66s | 1.9x |
