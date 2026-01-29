# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 07:48:58

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| BPI_Challenge_2019.xes | 694.8 | 1,595,923 | 1m 41.8s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2019.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 1,595,923 / 1,595,923 | 100.0% | 1.0 | 8.536s | 1.0x |
| **optimized** | 49,998 / 1,595,923 | 3.1% | 1.0 | 0.025s | 336.3x |
| **sqrt** | 19,998 / 1,595,923 | 1.3% | 1.0 | 0.516s | 16.5x |
| **minimal** | 4,996 / 1,595,923 | 0.3% | 1.0 | 0.188s | 45.5x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2019.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 8.47s | 1.0x |
| minimal | 1 | 100.0% | 0.19s | 45.4x |
| sqrt | 1 | 100.0% | 0.51s | 16.6x |
| optimized | 1 | 100.0% | 0.03s | 333.9x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 8.62s | 1.0x |
| minimal | 1 | 100.0% | 0.19s | 45.8x |
| sqrt | 1 | 100.0% | 0.53s | 16.3x |
| optimized | 1 | 100.0% | 0.03s | 339.0x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Case Arrival Trend**
*Parameters: `aggregation=W, mann_kendall`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 1 | 100% | 8.52s | 1.0x |
| minimal | 1 | 100.0% | 0.19s | 45.2x |
| sqrt | 1 | 100.0% | 0.51s | 16.6x |
| optimized | 1 | 100.0% | 0.03s | 336.0x |
