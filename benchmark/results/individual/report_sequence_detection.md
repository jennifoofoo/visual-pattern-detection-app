# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 19:38:43

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| Hospital_log.xes | 83.0 | 150,291 | 12.29s |
| BPI_Challenge_2019.xes | 694.8 | 1,595,923 | 1m 42.0s |

## 📊 Sampling Performance Summary

| Sampling | Events Retained | Patterns Found (Avg) | Sampling Time |
|:---|---:|---:|---:|
| **full** | 0.0% lost | 809534.0 | 0.100s |
| **minimal** | 96.7% lost | 559.3 | 3.200s |
| **optimized** | 66.8% lost | 3583.7 | 5.500s |
| **sqrt** | 86.7% lost | 6467.7 | 3.200s |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`

#### Sequence Detection
| Sampling | Patterns | Detection Time |
|:---|---:|---:|
| full | 0 | 0.70s |
| minimal | 0 | 0.06s |
| sqrt | 0 | 0.20s |
| optimized | 0 | 0.38s |
| full | 17360 | 16.77s |
| minimal | 55 | 0.19s |
| sqrt | 411 | 0.61s |
| optimized | 279 | 1.08s |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`

#### Sequence Detection
| Sampling | Patterns | Detection Time |
|:---|---:|---:|
| full | 773 | 1.23s |
| minimal | 407 | 0.15s |
| sqrt | 272 | 0.26s |
| optimized | 445 | 0.51s |
| full | 4545857 | 6m 41.9s |
| minimal | 2894 | 0.43s |
| sqrt | 38123 | 3.28s |
| optimized | 12783 | 3.96s |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`

#### Sequence Detection
| Sampling | Patterns | Detection Time |
|:---|---:|---:|
| full | 0 | 0.18s |
| minimal | 0 | 0.03s |
| sqrt | 0 | 0.06s |
| optimized | 0 | 0.10s |
| full | 293214 | 41.05s |
| minimal | 0 | 0.07s |
| sqrt | 0 | 0.20s |
| optimized | 7995 | 1.45s |
