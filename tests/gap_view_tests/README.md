# Gap Detection Tests

Synthetic test logs and tests for validating GapPattern detection across different X/Y view combinations.

## Synthetic Logs (`synthetic_gap_logs.py`)

Minimal, deterministic test data generators:

- **`make_numeric_y_gap()`** - 2D gap (x: [0.4, 0.6], y: [0.3, 0.7])
- **`make_categorical_y_gap()`** - Horizontal gap in category A (x: [0.4, 0.6])
- **`make_actual_time_gap()`** - 30-minute gap in actual_time
- **`make_relative_time_gap()`** - Gap in relative_time (600-900s)
- **`make_relative_ratio_gap()`** - Gap in relative_ratio (0.35-0.50)
- **`make_logical_time_gap()`** - Gap in logical_time (100-150)
- **`make_logical_relative_gap()`** - Gap in logical_relative (0.40-0.55)

## Gap Detection Semantics

- **Categorical Y**: Horizontal gaps within category bands (uses `min_gap_x_width`)
- **Numeric Y**: True 2D area gaps (uses `min_gap_area`)

## Running Tests

```bash
pytest tests/gap_view_tests/test_gap_pattern_views.py -v
```

Each test verifies that at least 1 gap is detected in the expected region.
