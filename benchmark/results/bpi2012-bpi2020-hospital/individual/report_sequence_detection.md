# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 22:50:37

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| Hospital_log.xes | 83.0 | 150,291 | 11.24s |

## 📈 Sampling Performance by Dataset

### Hospital_log.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 257.7 | 0.665s | 1.0x |
| **optimized** | 33.2% | 148.3 | 0.317s | 2.1x |
| **sqrt** | 13.3% | 90.7 | 0.171s | 3.9x |
| **minimal** | 3.3% | 135.7 | 0.077s | 8.6x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.68s | 1.0x |
| minimal | 0 | 100% | 0.06s | 11.2x |
| sqrt | 0 | 100% | 0.19s | 3.5x |
| optimized | 0 | 100% | 0.37s | 1.9x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 773 | 100% | 1.14s | 1.0x |
| minimal | 407 | 52.7% | 0.15s | 7.8x |
| sqrt | 272 | 35.2% | 0.26s | 4.4x |
| optimized | 445 | 57.6% | 0.48s | 2.3x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### Hospital_log.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 0 | 100% | 0.17s | 1.0x |
| minimal | 0 | 100% | 0.03s | 6.8x |
| sqrt | 0 | 100% | 0.06s | 2.8x |
| optimized | 0 | 100% | 0.10s | 1.7x |
