"""
Pattern support matrix defining which detection methods are available per view.
"""

from __future__ import annotations

PATTERN_MATRIX = {
    "time_activity_resource": {
        "cluster": 1,
        "gap": 1,
        "trend": 1,
        "burst": 1,
        "outlier": 1,
        "seasonality": 1,
        "sequential": 0,
        "overlap": 0,
    },
    "time_resource_activity": {
        "cluster": 1,
        "gap": 1,
        "trend": 1,
        "burst": 1,
        "outlier": 1,
        "seasonality": 0,
        "sequential": 0,
        "overlap": 0,
    },
    "case_time_activity": {
        "cluster": 1,
        "gap": 1,
        "trend": 0,
        "burst": 0,
        "outlier": 1,
        "seasonality": 0,
        "sequential": 0,
        "overlap": 1,
    },
    "time_duration_resource": {
        "cluster": 1,
        "gap": 0,
        "trend": 1,
        "burst": 0,
        "outlier": 1,
        "seasonality": 0,
        "sequential": 0,
        "overlap": 0,
    },
    "activity_time_resource": {
        "cluster": 1,
        "gap": 0,
        "trend": 1,
        "burst": 1,
        "outlier": 1,
        "seasonality": 1,
        "sequential": 0,
        "overlap": 0,
    },
    "variant_duration_resource": {
        "cluster": 0,
        "gap": 0,
        "trend": 1,
        "burst": 0,
        "outlier": 1,
        "seasonality": 0,
        "sequential": 0,
        "overlap": 0,
    },
    "case_eventindex_activity": {
        "cluster": 0,
        "gap": 0,
        "trend": 0,
        "burst": 0,
        "outlier": 0,
        "seasonality": 0,
        "sequential": 1,
        "overlap": 1,
    },
    "time_case_resource": {
        "cluster": 1,
        "gap": 1,
        "trend": 1,
        "burst": 1,
        "outlier": 1,
        "seasonality": 0,
        "sequential": 0,
        "overlap": 1,
    },
}


def is_pattern_supported(config_key: str, pattern_name: str) -> bool:
    """Return whether a pattern is supported for a given view configuration."""
    return bool(
        PATTERN_MATRIX.get(config_key, {}).get(pattern_name.lower(), 0)
    )


__all__ = ["PATTERN_MATRIX", "is_pattern_supported"]

