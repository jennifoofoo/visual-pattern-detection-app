# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 08:00:20

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| BPI_Challenge_2019.xes | 694.8 | 1,595,923 | 2m 16.4s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2019.xes

| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|---:|
| **full** | 1,595,923 / 1,595,923 | 100.0% | 293.0 | 14.244s | 1.0x |
| **optimized** | 49,998 / 1,595,923 | 3.1% | 15.7 | 0.221s | 64.5x |
| **sqrt** | 19,998 / 1,595,923 | 1.3% | 2.0 | 0.063s | 224.9x |
| **minimal** | 4,996 / 1,595,923 | 0.3% | 0.3 | 0.029s | 486.8x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2019.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=421, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 443 | 100% | 28.19s | 1.0x |
| minimal | 1 | 0.2% | 0.06s | 448.2x |
| sqrt | 4 | 0.9% | 0.13s | 215.5x |
| optimized | 27 | 6.1% | 0.47s | 59.7x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=421, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.00s | 1.0x |
| minimal | 0 | 100% | 0.00s | 0.9x |
| sqrt | 0 | 100% | 0.00s | 0.8x |
| optimized | 0 | 100% | 0.00s | 0.9x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2019.xes

**Temporal Cluster**
*Parameters: `min_cluster_size=421, temporal_eps=1.00`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 436 | 100% | 14.55s | 1.0x |
| minimal | 0 | 0.0% | 0.02s | 585.9x |
| sqrt | 2 | 0.5% | 0.06s | 246.0x |
| optimized | 20 | 4.6% | 0.19s | 76.5x |
