# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 08:07:51

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| BPI_Challenge_2019.xes | 694.8 | 1,595,923 | 2m 15.1s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2019.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 1,595,923 / 1,595,923 | 100.0% | 5355.0 | 49.318s | 1.0x |
| **optimized** | 49,998 / 1,595,923 | 3.1% | 4767.7 | 2.608s | 18.9x |
| **sqrt** | 19,998 / 1,595,923 | 1.3% | 2236.0 | 0.862s | 57.2x |
| **minimal** | 4,996 / 1,595,923 | 0.3% | 964.7 | 0.230s | 214.2x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2019.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 4700 | 100% | 20.66s | 1.0x |
| minimal | 0 | 0.0% | 0.06s | 318.4x |
| sqrt | 0 | 0.0% | 0.27s | 76.4x |
| optimized | 0 | 0.0% | 0.53s | 39.1x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9966 | 100% | 1m 43.6s | 1.0x |
| minimal | 2894 | 29.0% | 0.55s | 187.0x |
| sqrt | 6708 | 67.3% | 2.02s | 51.2x |
| optimized | 10000 | 100.3% | 5.37s | 19.3x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1399 | 100% | 23.73s | 1.0x |
| minimal | 0 | 0.0% | 0.07s | 328.6x |
| sqrt | 0 | 0.0% | 0.29s | 80.6x |
| optimized | 4303 | 307.6% | 1.93s | 12.3x |
