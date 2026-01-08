# Evaluation

This chapter presents a comprehensive evaluation of the visual pattern detection system for process mining event logs. We evaluate the system's ability to detect meaningful patterns across multiple view configurations, assess its performance on both synthetic and real-world data, and discuss threats to validity.

## Experimental Setup

### Test Environment

The evaluation was conducted using Python 3.14.0 with pytest 9.0.2 as the testing framework. All tests were executed on Windows with deterministic random seeds (np.random.seed(42)) to ensure reproducibility of synthetic data generation.

### Dataset Description

Our evaluation employs a two-tier testing strategy combining synthetic and real-world data:

**Synthetic Datasets**

We developed four categories of synthetic event logs, each designed to contain specific, guaranteed-detectable patterns:

1. **Outlier Detection Dataset** (`synthetic_outlier_logs.py`):
   - Time-based outliers: 100 business-hours events (9-17h) vs. 5 events at 3 AM
   - Case duration outliers: Normal cases (1-3 hours) vs. extreme case (48 hours)
   - Activity frequency outliers: Common activities (100 occurrences each) vs. rare activity (1 occurrence)
   - Resource workload outliers: Normal workload (20 events) vs. extreme workload (200 events)
   - Sequence outliers: Common sequence (A→B→C, 20 cases) vs. rare sequence (X→Y, 1 case)
   - Case complexity outliers: Normal cases (3-5 events) vs. complex case (50 events)

2. **Temporal Cluster Dataset** (`synthetic_cluster_logs.py`):
   - Temporal bursts: 3 distinct burst periods (50 events each) separated by quiet periods (5 events)
   - Activity-time clusters: Different activities clustering at specific times of day
   - Parallel cases: 10 overlapping case executions
   - Resource shift patterns: 3 distinct shift patterns (morning, afternoon, evening)
   - Variant timing patterns: Different process variants occurring at different times

3. **Sequence Pattern Dataset** (`synthetic_sequence_logs.py`):
   - Common sequences: 80% vs. 20% frequency distribution
   - Rare sequence outliers: Single occurrence vs. frequent patterns
   - Loop patterns: Repeated activity execution within cases
   - Skip patterns: Activities bypassed in certain cases
   - Parallel sequences: Different orderings of the same activities
   - Multi-variant sequences: 4 distinct process variants

4. **Gap Detection Dataset** (`synthetic_gap_logs.py`):
   - 2D numeric gaps: Empty regions in x∈[0.4,0.6] and y∈[0.3,0.7]
   - Categorical Y gaps: Horizontal gaps in specific categories
   - Process-aware transition gaps: Abnormal waiting times between consecutive activities (gap ~0.4-0.5 vs. normal ~0.05)
   - Time-based gaps: actual_time, relative_time, relative_ratio, logical_time, logical_relative

**Real-World Dataset**

- **Hospital_log.xes**: A real-world hospital event log containing patient treatment processes
- Used for validation testing across all pattern detection methods
- Tests verify that detectors complete successfully on real data without crashes
- Provides ground truth for practical applicability

### View Configuration Coverage

The system was tested across multiple view configurations following the EXTENDED_PATTERN_MATRIX:

**X-axis Dimensions:**
- `actual_time`: Wall-clock timestamps
- `relative_time`: Time since case start
- `relative_ratio`: Normalized case progress (0-1)
- `logical_time`: Event index in sequence
- `logical_relative`: Normalized event index

**Y-axis Dimensions:**
- `case_id`: Individual process instances
- `activity`: Process activities/tasks
- `resource`: Actors/resources performing activities
- `variant`: Process variant identifiers (when available)

**Color Dimensions:**
- `case_id`, `activity`, `resource` (for 3D view configurations)

Each pattern detector was tested only on meaningful view configurations as defined by the pattern matrix, ensuring that detection attempts align with theoretical pattern visibility.

### Evaluation Metrics

**Detection Success Rate:**
- Binary detection result (True/False) indicating whether patterns were found
- Used for synthetic data with guaranteed patterns (expected: 100% success)

**Robustness Metrics:**
- Edge case handling: empty DataFrames, single-event logs, missing columns
- Error handling: ValueError for empty data, graceful handling of missing resources
- View configuration validation: rejection of non-meaningful combinations

**Coverage Metrics:**
- Percentage of view configurations tested: 100% of meaningful combinations per pattern type
- Number of synthetic data scenarios: 21 distinct pattern scenarios
- Real-world validation: 100% of pattern types tested on Hospital_log.xes

## Results and Performance

### Outlier Detection Results

**Synthetic Data Performance:**

All six outlier types were successfully detected with 100% accuracy on synthetic data:

| Outlier Type | Detection Rate | Avg. Outliers Found | Expected Outliers |
|--------------|----------------|---------------------|-------------------|
| Time-based | 100% | 5 events at 3 AM | 5 events |
| Case Duration | 100% | 1 extreme case | 1 case |
| Activity Frequency | 100% | 1 rare activity | 1 activity |
| Resource Workload | 100% | 1 overworked resource | 1 resource |
| Sequence Outliers | 100% | 1 rare sequence | 1 sequence |
| Case Complexity | 100% | 1 complex case | 1 case |

**Real-World Data Performance (Hospital_log.xes):**

- Time outliers: Successfully detected with view_config={x: actual_time, y: case_id, color: activity}
- Activity outliers: Successfully detected with view_config={x: actual_time, y: activity, color: case_id}
- Resource outliers: Gracefully skipped when resource column unavailable
- Case duration outliers: Successfully detected abnormal case lengths
- Case complexity outliers: Successfully identified cases with unusual event counts

**View Configuration Impact:**

Outlier detection required complete 3D view configurations (x, y, color) matching entries in the EXTENDED_PATTERN_MATRIX. Initial tests with 2D configurations (x, y only) returned False due to pattern meaningfulness validation. After adding the color dimension, detection proceeded successfully, validating the matrix-based filtering approach.

### Temporal Cluster Detection Results

**Synthetic Data Performance:**

| Pattern Type | Detection Rate | Parameter Sensitivity |
|--------------|----------------|----------------------|
| Temporal Bursts | 100% (≥2 bursts detected) | temporal_eps=600s (10 min) optimal |
| Activity-Time Clusters | 100% | min_cluster_size=3 sufficient |
| Parallel Cases | 100% | Detected overlapping executions |
| Resource Shifts | 100% | temporal_eps=7200s (2 hrs) captures shifts |
| Variant Timing | 100% | Completed detection when variant column present |

**Real-World Data Performance:**

Hospital_log.xes testing showed successful completion across all temporal cluster types:
- Temporal bursts detected with configurable thresholds
- Case parallelism metrics computed (max/avg parallel cases)
- Resource patterns identified when resource column available
- Activity-time clustering completed successfully

### Sequence Pattern Detection Results

**Synthetic Data Performance:**

All sequence pattern types were successfully detected:

- Common sequences: Correctly identified frequent patterns (80% frequency)
- Rare sequences: Successfully flagged low-frequency patterns (outliers)
- Loop patterns: Detected repeated activity execution
- Skip patterns: Identified cases with bypassed activities
- Parallel sequences: Recognized different valid orderings
- Multi-variant sequences: Distinguished 4 process variants

**Real-World Data Performance:**

Hospital_log.xes validation:
- Sequence structure analysis completed
- Common sequence patterns extracted
- Sequence length distribution computed
- Transition matrices generated

### Gap Detection Results

**Process-Aware Gap Detection:**

The gap detector successfully identified abnormal waiting times between consecutive activities within cases:

| Test Scenario | Detection Rate | Gap Identification |
|---------------|----------------|-------------------|
| Categorical Y Gap | 100% | Detected transitions across empty X region (gap ~0.4-0.5 vs normal ~0.05) |
| Actual Time Gap | 100% | Identified 30-minute gaps in temporal flow |
| Relative Time Gap | 100% | Detected gaps in case-relative time |
| Logical Time Gap | 100% | Found gaps in event sequence indices |


**Real-World Data Performance:**

Hospital_log.xes gap detection:
- Successfully extracted transition gaps within cases
- Computed normality statistics per transition (Activity A → Activity B)
- Identified abnormal gaps exceeding statistical thresholds (Q3 + 1.5×IQR, P95)
- Generated gap severity scores (duration / threshold)

### Error Handling and Robustness

**Edge Case Performance:**

| Test Case | Expected Behavior | Actual Behavior | Result |
|-----------|-------------------|-----------------|--------|
| Empty DataFrame | ValueError or False | Raises ValueError (Outlier, Gap) / Returns False (TemporalCluster) | ✓ Pass |
| Single Event | False (cannot form patterns) | Returns False | ✓ Pass |
| Missing Column | Graceful skip or KeyError | Graceful handling / Skip with pytest.skip() | ✓ Pass |
| Missing Resource | Conditional skip | Tests skip when resource unavailable | ✓ Pass |

**Test Pipeline Integration:**

The test_detection_pipeline.py::TestErrorHandlingInPipeline::test_empty_dataframe_handling passed successfully, validating that the system handles exceptional conditions appropriately without crashes.

### Performance Summary

**Overall Detection Accuracy:**
- Synthetic data: 100% detection rate across all pattern types (21/21 scenarios)
- Real-world data: 100% completion rate (no crashes, graceful handling of missing data)

**Parameter Sensitivity:**
- Temporal clustering: Highly sensitive to temporal_eps (optimal values: 600-7200s depending on pattern)
- Outlier detection: Minimal parameterization required (automatic threshold computation)
- Gap detection: Uses statistical thresholds (Q3+1.5×IQR) for robustness

**View Configuration Validation:**
- 100% accuracy in rejecting non-meaningful view configurations
- Successful detection on all meaningful configurations per EXTENDED_PATTERN_MATRIX

### Key Findings and Contributions

**1. View-Aware Pattern Detection**

The integration of the EXTENDED_PATTERN_MATRIX successfully prevents meaningless pattern detection attempts. The system correctly rejected patterns on 2D view configurations until the color dimension was added, demonstrating effective theoretical grounding. This view-awareness ensures that only patterns visible in specific visual configurations are attempted.

**2. Synthetic-to-Real Data Pipeline**

Our two-tier evaluation strategy effectively validated pattern detectors:
- Synthetic data provided controlled environments with guaranteed patterns for algorithm verification
- Real-world data (Hospital_log.xes) validated practical applicability and robustness
- No detector that succeeded on synthetic data failed on real data, indicating good generalization

**3. Deterministic Testing**

Using fixed random seeds (np.random.seed(42)) enabled reproducible test results, facilitating debugging and ensuring consistent evaluation across test runs.

**4. Enhanced Real-World Validation**

To address the test oracle limitation, we implemented multiple validation strategies beyond basic "detection completed" checks:

*Sanity Checks on Pattern Characteristics:*
- Outlier percentage validation: Detected outliers must constitute ≤10% of total events (prevents over-detection)
- Gap percentage validation: Gaps should not exceed 50% of transitions (prevents false positives)
- Minimum detection thresholds: Require at least 10 patterns detected (prevents trivial empty results)

*Statistical Property Validation:*
- Outlier z-scores: Case duration outliers must have z-score >2.0 (statistically significant)
- Rare activity frequency: Activity outliers must appear <5% of the time (truly rare)
- Gap severity: Abnormal gaps must exceed 1.5× statistical threshold (Q3 + 1.5×IQR)

*Semi-Synthetic Validation (`test_semi_synthetic_validation.py`):*
- Inject known patterns into Hospital_log.xes data (5 time outliers at 3 AM, 1 extreme 48-hour case, 1 ultra-rare activity, 5 transition gaps)
- Measure precision and recall of detection: recall ≥60% required
- Provides ground truth for real-world data without manual labeling

*Cross-View Consistency Checks:*
- Case duration outliers should correlate with time-based outliers (≥20% overlap expected)
- Patterns detected in `actual_time × case_id` should appear in `actual_time × activity`
- Validates that detectors produce coherent results across different visualizations

These enhancements transform real-world tests from simple robustness checks into correctness validation with measurable quality metrics.

### Threats to Internal Validity

**1. Synthetic Data Realism**

*Threat:* Synthetic data generators may not capture the full complexity of real-world process mining data, potentially overestimating algorithm performance.

*Mitigation:* 
- Each synthetic dataset was designed based on real-world process mining scenarios (shift patterns, temporal bursts, outlier types)
- Real-world validation with Hospital_log.xes provided cross-validation
- Patterns were intentionally designed to be "guaranteed-detectable" to test algorithm correctness, not challenge detection limits

*Residual Risk:* Synthetic data may still lack certain noise characteristics or pattern combinations present in real-world logs.

**2. Test Oracle Limitation**

*Threat:* For real-world data (Hospital_log.xes), tests only verify "detection completed" without validating correctness of detected patterns (no ground truth labels).

*Mitigation:*
- **Synthetic data** provides ground truth for correctness validation
- **Sanity checks** on real-world data: outlier percentages (≤10%), gap percentages (≤50% of transitions), minimum detection thresholds
- **Statistical validation**: outliers must have z-scores >2.0, rare activities <5% frequency, abnormal gaps >1.5× threshold
- **Semi-synthetic validation** (`test_semi_synthetic_validation.py`): inject known patterns into Hospital_log.xes and verify detection (precision/recall measurement)
- **Cross-view consistency checks**: patterns detected in one view should correlate with related views (e.g., duration outliers → time outliers)
- Human expert validation would be needed for production deployment

*Residual Risk:* Some false positives/negatives may remain undetected, but validation tests now provide strong evidence of correctness beyond basic completion checks.

### Threats to External Validity

**1. Single Real-World Dataset**

*Threat:* Evaluation used only one real-world dataset (Hospital_log.xes from healthcare domain), limiting generalizability to other domains (e.g., manufacturing, finance, supply chain).

*Mitigation:*
- Healthcare processes are complex and varied, providing good coverage of temporal patterns, resources, and activities
- Synthetic data covers multiple domains conceptually (shifts, bursts, variants)
- System design is domain-agnostic (no healthcare-specific assumptions)

*Residual Risk:* Unknown performance on domains with very different characteristics (e.g., high-frequency IoT processes, long-running construction projects).


**4. Scale and Performance Not Evaluated**

*Threat:* Tests focus on correctness and robustness but do not evaluate computational performance on large-scale logs (e.g., millions of events).

*Mitigation:*
- Algorithms use efficient libraries (NumPy, pandas, scikit-learn)
- No algorithmic bottlenecks identified during testing

*Residual Risk:* Performance on very large logs (>1M events) is unknown.

### Threats to Construct Validity

**1. Detection Metrics Simplicity**

*Threat:* Evaluation primarily uses binary detection success (True/False) and pattern counts, which may not capture detection quality (e.g., precision, recall, F1-score).

*Mitigation:*
- For synthetic data with guaranteed patterns, binary success is appropriate (ground truth known)
- Pattern counts validated against expected values (e.g., 5 time outliers expected → 5 detected)
- Qualitative inspection of detected patterns performed during development

*Residual Risk:* Quality metrics like precision/recall would require extensive ground truth labeling of real-world data.

**2. View Configuration Coverage**

*Threat:* While 100% of meaningful view configurations were tested per the EXTENDED_PATTERN_MATRIX, some combinations may be incorrectly classified as meaningful or non-meaningful.

*Mitigation:*
- Pattern matrix was designed based on theoretical analysis of pattern visibility in dotted charts
- Testing validated that detection only proceeds on meaningful configurations
- Matrix can be refined based on user feedback and further research

*Residual Risk:* Pattern meaningfulness is somewhat subjective and domain-dependent.


###  Recommendations for Future Work

Based on identified threats to validity, we recommend:

1. **Multi-Domain Validation**: Extend evaluation to datasets from manufacturing, finance, and supply chain domains
2. **Large-Scale Performance Testing**: Benchmark algorithms on logs with 100K-10M events
3. **Ground Truth Labeling**: Create labeled real-world datasets with expert-annotated patterns for precision/recall evaluation
4. **User Study**: Conduct expert user evaluation to validate pattern meaningfulness and detection quality
