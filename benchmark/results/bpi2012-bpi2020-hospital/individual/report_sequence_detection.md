# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-28 01:39:53

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| BPI_Challenge_2012.xes | 70.7 | 262,200 | 9.06s |

## 📈 Sampling Performance by Dataset

### BPI_Challenge_2012.xes

| Sampling | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |
|:---|---:|---:|---:|---:|
| **full** | 100.0% | 6567.7 | 11.073s | 1.0x |
| **optimized** | 19.1% | 4952.3 | 2.510s | 4.4x |
| **sqrt** | 7.6% | 4893.7 | 6.856s | 1.6x |
| **minimal** | 1.9% | 3266.7 | 0.560s | 19.8x |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`, Color = `Resource`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 2930 | 100% | 10.59s | 1.0x |
| minimal | 2429 | 82.9% | 0.34s | 31.2x |
| sqrt | 4181 | 142.7% | 1.01s | 10.5x |
| optimized | 5780 | 197.3% | 2.40s | 4.4x |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 7557 | 100% | 12.58s | 1.0x |
| minimal | 952 | 12.6% | 0.28s | 45.5x |
| sqrt | 4326 | 57.2% | 1.19s | 10.6x |
| optimized | 2038 | 27.0% | 2.59s | 4.8x |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`, Color = `Activity`

#### BPI_Challenge_2012.xes

**Sequence Detection**
*Parameters: `min_support=30, prefixspan`*

| Sampling | Patterns | Retention | Time | Speedup |
|:---|---:|---:|---:|---:|
| full | 9216 | 100% | 10.05s | 1.0x |
| minimal | 6419 | 69.7% | 1.06s | 9.4x |
| sqrt | 6174 | 67.0% | 18.37s | 0.5x |
| optimized | 7039 | 76.4% | 2.54s | 4.0x |
