# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 07:54:44

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| BPI_Challenge_2019.xes | 694.8 | 1,595,923 | 2m 8.5s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2019.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 1,595,923 / 1,595,923 | 100.0% | 79797.0 | 18.996s | 1.0x |
| **optimized** | 49,998 / 1,595,923 | 3.1% | 2500.0 | 0.832s | 22.8x |
| **sqrt** | 19,998 / 1,595,923 | 1.3% | 1000.0 | 0.417s | 45.6x |
| **minimal** | 4,996 / 1,595,923 | 0.3% | 250.0 | 0.232s | 81.8x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2019.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 19.18s | 1.0x |
| minimal | 250 | 0.3% | 0.24s | 79.8x |
| sqrt | 1000 | 1.3% | 0.37s | 52.0x |
| optimized | 2500 | 3.1% | 0.83s | 23.1x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 19.17s | 1.0x |
| minimal | 250 | 0.3% | 0.25s | 77.7x |
| sqrt | 1000 | 1.3% | 0.48s | 39.7x |
| optimized | 2500 | 3.1% | 0.85s | 22.6x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Outlier Detection**
*Parameters: `isolation_forest + statistical`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 79797 | 100% | 18.64s | 1.0x |
| minimal | 250 | 0.3% | 0.21s | 89.0x |
| sqrt | 1000 | 1.3% | 0.40s | 46.9x |
| optimized | 2500 | 3.1% | 0.82s | 22.8x |
