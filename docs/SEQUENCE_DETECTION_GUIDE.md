# Sequence Detection Guide

## 📋 Overview

Sequence Detection identifies **frequent sequential patterns** in event logs using the **PrefixSpan algorithm**. It discovers common activity sequences that occur repeatedly across cases, helping analysts understand typical process flows, behavioral patterns, and recurring execution paths.

**Key Features:**
- Discovers frequent subsequences automatically
- Support-based filtering (minimum occurrence threshold)
- Strict vs. non-strict matching modes
- Top-k pattern ranking
- Visual highlighting of pattern occurrences

---

## 🎯 What It Detects

**Frequent Activity Sequences:**
- Recurring patterns of activities across multiple cases
- Both complete paths and partial subsequences
- Ordered sequences (A → B → C, not arbitrary combinations)
- Patterns that meet minimum support threshold

**Example:**
```
Common sequence: "Order → Approve → Ship → Deliver"
Found in: 127 cases (support count)
Min support threshold: 30 cases

→ Pattern detected as significant recurring flow
```

---

## 🔧 Configuration & Parameters

### 1. **Min Support (%)**
**What it is:** Minimum percentage of cases/groups that must contain the pattern

**Default:** 30% (adaptive based on data)

**How it works:**
- Calculated as percentage of unique Y-axis groups (e.g., cases, resources)
- Converted to absolute count: `min_support = ceil(unique_groups × 0.30)`
- Guards against tiny datasets: minimum 2 occurrences

**Examples:**
```
100 cases × 30% = 30 cases minimum
50 cases × 30% = 15 cases minimum
5 cases × 30% = 2 cases minimum (guarded)
```

**When to adjust:**
- **Increase (40-50%)** for large datasets to focus on highly common patterns
- **Decrease (10-20%)** for small datasets to discover less frequent patterns
- **Keep default (30%)** for balanced exploration

### 2. **Strict Mode**
**What it is:** Toggle between strict and non-strict pattern matching

**Default:** Non-strict (allows gaps between events)

**Difference:**

| Mode | Description | Pattern Match |
|------|-------------|---------------|
| **Non-Strict** (default) | Events with same timestamp form itemsets<br/>Patterns can skip intermediate events | A → B → C matches:<br/>• A, B, C<br/>• A, X, B, Y, C<br/>• A, B, B, C |
| **Strict** | Consecutive events only<br/>No gaps allowed | A → B → C matches:<br/>• A, B, C only<br/>Not: A, X, B, C |

**When to use Strict:**
- When you need exact consecutive sequences
- For compliance checking (must follow exact path)
- When intermediate activities are significant

**When to use Non-Strict:**
- When you want flexible pattern matching (default)
- To find patterns despite process variations
- For exploratory analysis

### 3. **Top-K Filtering**
**What it is:** Display only the top K most frequent patterns

**Default:** Top 5 patterns

**Purpose:** Prevent information overload from hundreds of patterns

**Ranking:** By support count (number of occurrences)

---

## 📊 Algorithm: PrefixSpan

### How It Works

**PrefixSpan** (Prefix-Projected Sequential Pattern Mining) is an efficient algorithm for discovering frequent subsequences.

**Key Concepts:**

1. **Sequence:** Ordered list of events within a group (case/resource/activity)
   ```
   Case 1: [A, B, C, D]
   Case 2: [A, B, E, D]
   Case 3: [A, B, C, D]
   ```

2. **Subsequence:** Ordered subset (gaps allowed in non-strict mode)
   ```
   Pattern [A, B, D] is a subsequence of:
   - [A, B, C, D] ✓
   - [A, B, E, D] ✓
   - [A, X, Y, B, Z, D] ✓ (non-strict)
   ```

3. **Support Count:** Number of sequences containing the pattern
   ```
   Pattern [A, B, D]: Found in 78 cases → support = 78
   ```

4. **Frequent Pattern:** Pattern meeting min_support threshold
   ```
   If min_support = 30 and support = 78 → Frequent ✓
   If min_support = 30 and support = 12 → Not frequent ✗
   ```

### Algorithm Steps

```
1. Data Preparation:
   - Sort events by (grouping_key, timestamp)
   - Group events with same timestamp into itemsets
   - Build sequence database per group

2. Pattern Mining (PrefixSpan):
   - Start with length-1 patterns (single events)
   - Recursively extend patterns by appending next events
   - Prune patterns below min_support
   - Continue until no more frequent extensions

3. Post-Processing:
   - Map patterns back to original event indices
   - Calculate support counts
   - Rank by frequency
   - Filter to top-k

4. Visualization:
   - Highlight pattern occurrences on dotted chart
   - Color-code by pattern
   - Show pattern metadata
```

### Performance Guards

**To handle large datasets, the detector includes:**

1. **Sequence Truncation:** Max 50 events per sequence
2. **Recursion Limit:** Increased to 5000 for deep pattern trees
3. **Pattern Cap:** Max 500 patterns before top-k filtering
4. **Data Validation:** Remove NaN values, ensure string types

---

## 🎨 Visual Representation

**On Dotted Chart:**
- **Markers:** Colored dots highlighting events in detected patterns
- **Colors:** Each pattern gets a distinct color
- **Hover Info:** Pattern details and event information
- **Legend:** Pattern labels with support counts

**Pattern Tab UI:**
- **Metrics:** Total patterns, total occurrences, coverage
- **Pattern Table:** Ranked list with support counts
- **Filtering:** Select specific patterns to visualize
- **Strict Mode Toggle:** Switch between matching modes

---

## ✅ Meaningful Configurations

### Best Configurations (High Signal)

| X-Axis | Y-Axis | Color | Interpretation |
|--------|--------|-------|----------------|
| **Actual time** | **Case ID** | **Activity** | Case progression sequences<br/>Discover typical process flows |
| **Actual time** | **Resource** | **Activity** | Resource work sequences<br/>Understand resource behaviors |
| **Actual time** | **Activity** | **Case ID** | Cases executing similar activities<br/>Find process variants |
| **Logical time** | **Case ID** | **Activity** | Event-order sequences<br/>Independent of timestamps |

### ❌ Invalid/Meaningless Configurations

| X-Axis | Y-Axis | Color | Why Invalid? |
|--------|--------|-------|--------------|
| Any | Any | **Same as Y** | Cannot group by same dimension |
| Any | **< 3 unique** | Any | Insufficient groups for patterns |
| **Non-temporal** | Any | Any | Sequences need ordering |

**Grouping Key Requirements:**
- Must have ≥3 unique values
- Cannot be same as event key (color)
- Should represent logical groupings (cases, resources, activities)

---

## 📈 Interpretation Guidelines

### Understanding Support Counts

**High Support (>50% of groups):**
- Common process flows
- Standard operating procedures
- Core process variants
- **Action:** Document as typical behavior

**Medium Support (20-50% of groups):**
- Frequent but not dominant patterns
- Significant process variants
- Department-specific flows
- **Action:** Investigate conditional triggers

**Low Support (5-20% of groups):**
- Exception handling paths
- Rare but recurring patterns
- Edge cases worth documenting
- **Action:** Analyze when/why they occur

**Very Low Support (<5%):**
- May be noise or data quality issues
- Extremely rare exceptions
- **Action:** Verify data quality first

### Pattern Length Insights

**Short Patterns (2-3 events):**
- Basic transitions
- Core building blocks
- Often have high support

**Medium Patterns (4-7 events):**
- Process segments
- Functional workflows
- Moderate support

**Long Patterns (8+ events):**
- Complete process flows
- End-to-end paths
- Typically lower support
- More specific scenarios

### Coverage Analysis

**Coverage = (Events in patterns) / (Total events)**

- **High Coverage (>80%):** Well-structured, predictable process
- **Medium Coverage (50-80%):** Mix of structured and ad-hoc work
- **Low Coverage (<50%):** Highly variable, flexible process

---

## 🔍 Use Cases

### 1. Process Discovery
**Goal:** Understand typical process flows

**Approach:**
- Use default settings (30% support)
- View: Actual time × Case ID × Activity
- Focus on high-support patterns

**Questions Answered:**
- What are the most common execution paths?
- How do cases typically progress?
- What are the main process variants?

### 2. Compliance Checking
**Goal:** Verify required sequences are followed

**Approach:**
- Enable **Strict Mode**
- Increase min_support (40-50%)
- View: Actual time × Case ID × Activity

**Questions Answered:**
- Do cases follow mandated approval sequences?
- Are safety checks performed in order?
- Where do compliance violations occur?

### 3. Resource Behavior Analysis
**Goal:** Understand how resources work

**Approach:**
- View: Actual time × Resource × Activity
- Default settings
- Focus on medium-support patterns

**Questions Answered:**
- What activities do resources typically perform together?
- Are there work specialization patterns?
- Do resources follow consistent routines?

### 4. Bottleneck Investigation
**Goal:** Find problematic activity sequences

**Approach:**
- Combine with Gap Detection
- Look for patterns preceding long gaps
- View: Actual time × Case ID × Activity

**Questions Answered:**
- What sequences lead to delays?
- Where in the process do bottlenecks form?
- Are delays associated with specific patterns?

### 5. Process Variant Analysis
**Goal:** Identify and categorize process variants

**Approach:**
- Decrease min_support (10-20%)
- View: Actual time × Case ID × Activity
- Compare pattern frequencies

**Questions Answered:**
- How many distinct variants exist?
- Which variants are most common?
- What triggers different execution paths?

---

## ⚙️ Configuration Examples

### Example 1: Standard Process Discovery
```
View Config:
- X: Actual time
- Y: Case ID  
- Color: Activity

Settings:
- Min Support: 30% (default)
- Strict Mode: OFF (default)
- Top-K: 5

Expected Output:
- 3-5 common process flows
- Support: 50-200 cases
- Length: 4-8 activities
```

### Example 2: Rare Exception Paths
```
View Config:
- X: Actual time
- Y: Case ID
- Color: Activity

Settings:
- Min Support: 5% (low)
- Strict Mode: OFF
- Top-K: 10

Expected Output:
- 5-10 less common variants
- Support: 5-20 cases
- Length: varies
```

### Example 3: Exact Compliance Sequences
```
View Config:
- X: Actual time
- Y: Case ID
- Color: Activity

Settings:
- Min Support: 40% (high)
- Strict Mode: ON (consecutive only)
- Top-K: 3

Expected Output:
- 1-3 mandatory sequences
- Support: >100 cases
- Length: 3-5 activities (exact match)
```

### Example 4: Resource Work Patterns
```
View Config:
- X: Actual time
- Y: Resource
- Color: Activity

Settings:
- Min Support: 30%
- Strict Mode: OFF
- Top-K: 5

Expected Output:
- 3-5 typical work sequences per resource
- Support: 10-50 occurrences
- Length: 2-5 activities
```

---

## 🐛 Troubleshooting

### "No patterns detected"

**Possible Causes:**
1. **Min support too high**
   - Solution: Lower to 10-20%
   
2. **Insufficient data**
   - Check: Y-axis has ≥3 unique values?
   - Solution: Use larger dataset or change grouping

3. **Highly variable process**
   - Every case is unique
   - Solution: Expected behavior for ad-hoc processes

4. **Grouping key = Event key**
   - Error: Same dimension used twice
   - Solution: Choose different Y-axis or Color

### "Pattern detected = False but some patterns shown"

**Cause:** Top-k filtering removed all patterns

**Solution:**
- Increase top-k to 10 or 20
- Lower min_support threshold
- Check pattern support counts in details

### "Too many patterns (>100)"

**Cause:** Min support too low or very structured process

**Solution:**
- Increase min_support to 40-50%
- Focus on top-5 or top-10 patterns
- Use strict mode to reduce pattern variations

### "Patterns seem random/meaningless"

**Possible Causes:**
1. **Wrong view configuration**
   - Check: Is Y-axis a meaningful grouping?
   - Solution: Use Case ID or Resource

2. **Data quality issues**
   - Events have incorrect timestamps
   - Solution: Verify data preprocessing

3. **Too low support threshold**
   - Finding noise patterns
   - Solution: Increase to 30-50%

### "Performance is slow"

**Causes and Solutions:**

1. **Large dataset (>50K events)**
   - Expected: First run may take 10-30 seconds
   - Optimization: Use sampling mode

2. **Many unique groups (>1000)**
   - PrefixSpan is compute-intensive
   - Solution: Increase min_support to 40%+

3. **Very long sequences (>100 events per group)**
   - Auto-truncated to 50 events
   - Already optimized

---

## 📚 Advanced Topics

### Itemsets vs. Individual Events

**Non-Strict Mode** groups events with the same timestamp into **itemsets**:

```
Sequence:
  t1: [A]
  t2: [B, C]  ← Itemset (concurrent)
  t3: [D]

Represented as: [(A), (B,C), (D)]

Patterns found:
- A → (B,C) → D  ✓
- A → B → D      ✓ (subsequence)
- A → C → D      ✓ (subsequence)
```

**Use Case:** Capturing concurrent activities in workflows

### Pattern Projection

PrefixSpan uses **projection** for efficiency:

```
1. Find patterns starting with 'A'
2. Project database to sequences after 'A'
3. Recursively find patterns in projected DB
4. Combine: A + found_patterns
```

**Benefit:** Avoids re-scanning entire database

### Support vs. Confidence

**Support:** How often pattern occurs (absolute)
```
Support([A,B,C]) = 50 cases
```

**Confidence:** How often B follows A (conditional)
```
Confidence(A → B) = P(B|A) = Support(A,B) / Support(A)
```

**Note:** Sequence Detection focuses on support only. Confidence would require additional analysis.

---

## 🎓 Best Practices

### ✅ Do's

1. **Start with default settings** (30% support, non-strict)
2. **Use Case ID grouping** for process flow analysis
3. **Check pattern support counts** before interpretation
4. **Combine with other patterns** (gaps, outliers) for insights
5. **Validate patterns** against domain knowledge
6. **Document significant patterns** for process documentation

### ❌ Don'ts

1. **Don't set min_support too low** (<5%) - finds noise
2. **Don't ignore data quality** - garbage in, garbage out
3. **Don't over-interpret rare patterns** - verify significance
4. **Don't use same dimension for Y and Color** - invalid config
5. **Don't expect patterns in truly ad-hoc processes** - may not exist
6. **Don't rely solely on automated detection** - validate findings

---

## 📖 Related Patterns

**Complement Sequence Detection with:**

- **Gap Detection:** Find delays within discovered patterns
- **Outlier Detection:** Identify cases deviating from common sequences
- **Temporal Clusters:** Understand when patterns occur
- **Case Arrival Trend:** See if pattern frequency changes over time

**Combined Analysis Example:**
```
1. Discover common sequences (Sequence Detection)
2. Find gaps within those sequences (Gap Detection)  
3. Identify cases not following patterns (Outlier Detection)
4. Analyze if patterns cluster in time (Temporal Clusters)
```

---

## 🔗 References

**PrefixSpan Algorithm:**
- Pei, J., et al. (2004). "Mining Sequential Patterns by Pattern-Growth: The PrefixSpan Approach"
- IEEE Transactions on Knowledge and Data Engineering

**Implementation:**
- Python `prefixspan` library
- Core detector: `core/detection/sequence_detector.py`
- UI handler: `core/app_utils/pattern_ui.py`

**Related Documentation:**
- [Pattern Matrix Structure](./PATTERN_MATRIX_STRUCTURE.md) - View configurations
- [Gap Detection Guide](./GAP_DETECTION_GUIDE.md) - Delay analysis
- [System Architecture](./SYSTEM_ARCHITECTURE.md) - Technical details

---

**Last updated:** January 2026
