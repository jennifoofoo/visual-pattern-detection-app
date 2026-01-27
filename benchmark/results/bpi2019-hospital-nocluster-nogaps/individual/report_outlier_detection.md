# Pattern Detection Benchmark Report (v3)

**Generated:** 2026-01-27 18:59:52

## 📁 Data Management

| File | Size (MB) | Events | Load Time |
|:---|---:|---:|---:|
| Hospital_log.xes | 83.0 | 150,291 | 13.26s |
| BPI_Challenge_2019.xes | 694.8 | 1,595,923 | 2m 4.1s |
| BPI Challenge 2018.xes | 1856.7 | 2,514,266 | 5m 0.9s |

## 📊 Sampling Performance Summary

| Sampling | Events Retained | Patterns Found (Avg) | Sampling Time |
|:---|---:|---:|---:|
| **full** | 0.0% lost | 29104.0 | 0.100s |
| **minimal** | 96.7% lost | 250.0 | 4.000s |
| **optimized** | 66.8% lost | 2498.0 | 6.600s |
| **sqrt** | 86.7% lost | 999.5 | 3.900s |

## 🔍 Detection Statistics by Configuration

### Activity Overview
**Axes:** X = `Actual time`, Y = `Activity`

#### Outlier Detection
| Sampling | Patterns | Detection Time |
|:---|---:|---:|
| full | 7515 | 1.83s |
| minimal | 250 | 0.18s |
| sqrt | 999 | 0.34s |
| optimized | 2496 | 0.60s |
| full | 79797 | 17.07s |
| minimal | 250 | 0.20s |
| sqrt | 1000 | 0.42s |
| optimized | 2500 | 0.66s |
| full | 0 | 1.50s |

### Case Progression
**Axes:** X = `Actual time`, Y = `Case ID`

#### Outlier Detection
| Sampling | Patterns | Detection Time |
|:---|---:|---:|
| full | 7515 | 1.82s |
| minimal | 250 | 0.17s |
| sqrt | 999 | 0.33s |
| optimized | 2496 | 0.61s |
| full | 79797 | 17.58s |
| minimal | 250 | 0.18s |
| sqrt | 1000 | 0.38s |
| optimized | 2500 | 0.63s |
| full | 0 | 1.56s |

### Resource Timeline
**Axes:** X = `Actual time`, Y = `Resource`

#### Outlier Detection
| Sampling | Patterns | Detection Time |
|:---|---:|---:|
| full | 7515 | 1.82s |
| minimal | 250 | 0.17s |
| sqrt | 999 | 0.40s |
| optimized | 2496 | 0.64s |
| full | 79797 | 17.59s |
| minimal | 250 | 0.18s |
| sqrt | 1000 | 0.38s |
| optimized | 2500 | 0.67s |
| full | 0 | 1.54s |
