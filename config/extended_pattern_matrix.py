"""
Extended Pattern Matrix for Paper Documentation and Frontend Filtering.

This matrix defines which patterns are detectable and meaningful for each view configuration.
Each pattern entry contains comprehensive metadata for documentation and UI purposes.

Structure: Tuple-based keys (x_axis, y_axis, color) for explicit 3D view configuration.
"""

from typing import Dict, Any, Optional, List, Tuple

# ============================================================================
# PATTERN BASE DEFINITIONS (Reusable metadata for each pattern type)
# ============================================================================

PATTERN_BASE_DEFINITIONS = {
    "gap": {
        "algorithm": "Transition-specific normality learning using IQR and P95 thresholds",
        "visual_base": "Red rectangles showing time spans of abnormal gaps",
        "requirements_base": ["case_id", "activity"],
        "output_base": "List of abnormal gaps with severity scores (duration/threshold)"
    },
    "temporal_cluster_x": {
        "algorithm": "DBSCAN clustering on temporal event distribution",
        "visual_base": "Colored circles around event dots highlighting temporal bursts",
        "requirements_base": [],
        "output_base": "Temporal clusters with event density and time ranges"
    },
    "outlier": {
        "algorithm": "IQR-based statistical analysis for multiple anomaly types",
        "visual_base": "Red highlighted circles over outlier events with anomaly scores",
        "requirements_base": [],
        "output_base": "List of outlier events with detection reasons and confidence scores"
    }
}

# ============================================================================
# EXTENDED PATTERN MATRIX
# Key: (x_axis, y_axis, color)
# ============================================================================

EXTENDED_PATTERN_MATRIX: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {
    
    # ========================================================================
    # ACTUAL_TIME × RESOURCE × CASE_ID
    # ========================================================================
    ("actual_time", "resource", "case_id"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps between activities",
            "interpretation": "Detects abnormal waiting times between consecutive activities within cases. Shows which resources have process delays. Color-coded by case for case-specific analysis.",
            "use_case": "Finding process bottlenecks, resource unavailability, weekend delays per case",
            "requirements": ["case_id", "activity", "actual_time", "resource"],
            "output": "List of abnormal gaps with severity scores, grouped by resource and colored by case",
            "color_impact": "Case coloring helps identify which specific cases experience delays at each resource"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots showing temporal bursts of resource activity",
            "interpretation": "Detects time periods with high event concentration for resources. Color-coded by case to see case distribution in busy periods.",
            "use_case": "Finding peak workload periods per resource, identifying batch processing times",
            "requirements": ["actual_time", "resource", "case_id"],
            "output": "Temporal clusters with event density, time ranges, and case distribution",
            "color_impact": "Case coloring reveals whether busy periods are caused by many cases or few intensive cases"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted circles over outlier events with anomaly reasons",
            "interpretation": "Detects unusual events based on timing, resource behavior, and frequency patterns. Color-coded by case for case-level anomaly analysis.",
            "use_case": "Finding exceptional cases, data quality issues, resource violations",
            "requirements": ["actual_time", "resource", "case_id"],
            "output": "Outlier events with resource-specific and case-specific anomaly reasons",
            "color_impact": "Case coloring helps identify whether outliers are case-specific or resource-specific"
        }
    },

    # ========================================================================
    # ACTUAL_TIME × RESOURCE × ACTIVITY
    # ========================================================================
    ("actual_time", "resource", "activity"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps between activities",
            "interpretation": "Detects abnormal waiting times between consecutive activities within cases. Shows which resources have process delays. Color-coded by activity to see which activities cause delays.",
            "use_case": "Finding process bottlenecks per resource, identifying which activities cause delays",
            "requirements": ["case_id", "activity", "actual_time", "resource"],
            "output": "List of abnormal gaps with severity scores, grouped by resource and colored by activity",
            "color_impact": "Activity coloring reveals which specific activities are delayed at each resource"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots showing temporal bursts, colored by activity",
            "interpretation": "Detects time periods with high event concentration for resources. Color-coded by activity to see activity distribution in busy periods.",
            "use_case": "Finding peak workload periods per resource, identifying which activities cluster",
            "requirements": ["actual_time", "resource", "activity"],
            "output": "Temporal clusters with event density, time ranges, and activity distribution",
            "color_impact": "Activity coloring shows which activities dominate busy periods"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted circles over outlier events, colored by activity",
            "interpretation": "Detects unusual events based on timing, resource behavior, and frequency patterns. Color-coded by activity for activity-level anomaly analysis.",
            "use_case": "Finding exceptional resource-activity combinations, rare activities",
            "requirements": ["actual_time", "resource", "activity"],
            "output": "Outlier events with resource-activity-specific anomaly reasons",
            "color_impact": "Activity coloring helps identify which activities are outliers at each resource"
        }
    },

    # ========================================================================
    # ACTUAL_TIME × RESOURCE × RESOURCE (Same dimension on Y and Color)
    # ========================================================================
    ("actual_time", "resource", "resource"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps, colored by resource",
            "interpretation": "Detects abnormal waiting times between consecutive activities within cases. Shows which resources have process delays. Color matches Y-axis for clear resource identification.",
            "use_case": "Finding process bottlenecks per resource with clear visual resource separation",
            "requirements": ["case_id", "activity", "actual_time", "resource"],
            "output": "List of abnormal gaps with severity scores, grouped and colored by resource",
            "color_impact": "Resource coloring provides redundant but clear visual separation of resources"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots, colored by resource",
            "interpretation": "Detects time periods with high event concentration for resources. Color matches Y-axis for clear resource identification.",
            "use_case": "Finding peak workload periods per resource with clear visual separation",
            "requirements": ["actual_time", "resource"],
            "output": "Temporal clusters with event density and time ranges per resource",
            "color_impact": "Resource coloring provides redundant but clear visual separation"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted circles over outlier events, colored by resource",
            "interpretation": "Detects unusual events based on timing, resource behavior, and frequency patterns. Color matches Y-axis for clear resource identification.",
            "use_case": "Finding exceptional resource behavior with clear visual separation",
            "requirements": ["actual_time", "resource"],
            "output": "Outlier events with resource-specific anomaly reasons",
            "color_impact": "Resource coloring provides redundant but clear visual separation"
        }
    },

    # ========================================================================
    # ACTUAL_TIME × ACTIVITY × CASE_ID
    # ========================================================================
    ("actual_time", "activity", "case_id"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps between activities",
            "interpretation": "Detects abnormal waiting times between specific activity transitions. Shows which activity sequences have delays. Color-coded by case for case-specific analysis.",
            "use_case": "Identifying bottlenecks in specific process steps, analyzing handover times per case",
            "requirements": ["case_id", "activity", "actual_time"],
            "output": "List of abnormal gaps per transition (Activity A → Activity B), colored by case",
            "color_impact": "Case coloring reveals which cases experience delays in specific activity transitions"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots showing temporal bursts of activities",
            "interpretation": "Detects time periods with high event concentration for activities. Color-coded by case to see case distribution.",
            "use_case": "Finding peak workload periods per activity, batch processing detection",
            "requirements": ["actual_time", "activity", "case_id"],
            "output": "Temporal clusters with event density, time ranges, and case distribution",
            "color_impact": "Case coloring shows whether activity bursts involve many cases or few intensive cases"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by case",
            "interpretation": "Detects activities that happen at unusual times or with unusual frequency. Color-coded by case for case-level analysis.",
            "use_case": "Finding rare activities, off-hours events, frequency anomalies per case",
            "requirements": ["actual_time", "activity", "case_id"],
            "output": "Outlier events with activity-specific and case-specific anomaly reasons",
            "color_impact": "Case coloring helps identify whether activity outliers are case-specific"
        }
    },

    # ========================================================================
    # ACTUAL_TIME × ACTIVITY × ACTIVITY (Same dimension on Y and Color)
    # ========================================================================
    ("actual_time", "activity", "activity"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps, colored by activity",
            "interpretation": "Detects abnormal waiting times between specific activity transitions. Color matches Y-axis for clear activity identification.",
            "use_case": "Identifying bottlenecks in specific process steps with clear visual separation",
            "requirements": ["case_id", "activity", "actual_time"],
            "output": "List of abnormal gaps per transition (Activity A → Activity B), colored by activity",
            "color_impact": "Activity coloring provides redundant but clear visual separation of activities"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots, colored by activity",
            "interpretation": "Detects time periods with high event concentration for activities. Color matches Y-axis for clear activity identification.",
            "use_case": "Finding peak workload periods per activity with clear visual separation",
            "requirements": ["actual_time", "activity"],
            "output": "Temporal clusters with event density and time ranges per activity",
            "color_impact": "Activity coloring provides redundant but clear visual separation"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by activity",
            "interpretation": "Detects activities that happen at unusual times or with unusual frequency. Color matches Y-axis for clear activity identification.",
            "use_case": "Finding rare activities, off-hours events with clear visual separation",
            "requirements": ["actual_time", "activity"],
            "output": "Outlier events with activity-specific anomaly reasons",
            "color_impact": "Activity coloring provides redundant but clear visual separation"
        }
    },

    # ========================================================================
    # ACTUAL_TIME × ACTIVITY × RESOURCE
    # ========================================================================
    ("actual_time", "activity", "resource"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps, colored by resource",
            "interpretation": "Detects abnormal waiting times between specific activity transitions. Shows which activity sequences have delays. Color-coded by resource to see resource involvement.",
            "use_case": "Identifying bottlenecks in specific process steps, analyzing which resources cause delays",
            "requirements": ["case_id", "activity", "actual_time", "resource"],
            "output": "List of abnormal gaps per transition (Activity A → Activity B), colored by resource",
            "color_impact": "Resource coloring reveals which resources are involved in activity transition delays"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots, colored by resource",
            "interpretation": "Detects time periods with high event concentration for activities. Color-coded by resource to see resource distribution.",
            "use_case": "Finding peak workload periods per activity, identifying resource involvement",
            "requirements": ["actual_time", "activity", "resource"],
            "output": "Temporal clusters with event density, time ranges, and resource distribution",
            "color_impact": "Resource coloring shows which resources are active during activity bursts"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by resource",
            "interpretation": "Detects activities that happen at unusual times or with unusual frequency. Color-coded by resource for resource-level analysis.",
            "use_case": "Finding rare activities, identifying which resources perform unusual activities",
            "requirements": ["actual_time", "activity", "resource"],
            "output": "Outlier events with activity-resource-specific anomaly reasons",
            "color_impact": "Resource coloring helps identify which resources are involved in activity outliers"
        }
    },

    # ========================================================================
    # ACTUAL_TIME × CASE_ID × CASE_ID (Same dimension on Y and Color)
    # ========================================================================
    ("actual_time", "case_id", "case_id"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps within individual case timelines",
            "interpretation": "Detects abnormal waiting times within specific cases. Each case is analyzed individually. Color matches Y-axis for clear case identification.",
            "use_case": "Finding case-specific delays, comparing case execution times with clear visual separation",
            "requirements": ["case_id", "activity", "actual_time"],
            "output": "Abnormal gaps with case identification, colored by case",
            "color_impact": "Case coloring provides redundant but clear visual separation of cases"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots, colored by case",
            "interpretation": "Detects time periods with high event concentration across cases. Color matches Y-axis for clear case identification.",
            "use_case": "Finding peak workload periods, batch processing times with clear case separation",
            "requirements": ["actual_time", "case_id"],
            "output": "Temporal clusters with event density and time ranges per case",
            "color_impact": "Case coloring provides redundant but clear visual separation"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events within cases",
            "interpretation": "Detects cases with unusual execution patterns or extreme durations. Color matches Y-axis for clear case identification.",
            "use_case": "Finding exceptional cases, compliance violations with clear visual separation",
            "requirements": ["actual_time", "case_id"],
            "output": "Outlier cases with anomaly reasons, colored by case",
            "color_impact": "Case coloring provides redundant but clear visual separation"
        }
    },

    # ========================================================================
    # ACTUAL_TIME × CASE_ID × ACTIVITY
    # ========================================================================
    ("actual_time", "case_id", "activity"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps within individual case timelines, colored by activity",
            "interpretation": "Detects abnormal waiting times within specific cases. Each case is analyzed individually. Color-coded by activity to see which activities cause delays.",
            "use_case": "Finding case-specific delays, identifying which activities cause delays in each case",
            "requirements": ["case_id", "activity", "actual_time"],
            "output": "Abnormal gaps with case identification, colored by activity",
            "color_impact": "Activity coloring reveals which activities are delayed within each case"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots, colored by activity",
            "interpretation": "Detects time periods with high event concentration across cases. Color-coded by activity to see activity distribution.",
            "use_case": "Finding peak workload periods, identifying which activities cluster across cases",
            "requirements": ["actual_time", "case_id", "activity"],
            "output": "Temporal clusters with event density, time ranges, and activity distribution",
            "color_impact": "Activity coloring shows which activities dominate busy periods across cases"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by activity",
            "interpretation": "Detects cases with unusual execution patterns or extreme durations. Color-coded by activity for activity-level analysis.",
            "use_case": "Finding exceptional cases, identifying which activities are outliers in each case",
            "requirements": ["actual_time", "case_id", "activity"],
            "output": "Outlier cases with activity-specific anomaly reasons",
            "color_impact": "Activity coloring helps identify which activities are outliers within cases"
        }
    },

    # ========================================================================
    # ACTUAL_TIME × CASE_ID × RESOURCE
    # ========================================================================
    ("actual_time", "case_id", "resource"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps within individual case timelines, colored by resource",
            "interpretation": "Detects abnormal waiting times within specific cases. Each case is analyzed individually. Color-coded by resource to see resource involvement in delays.",
            "use_case": "Finding case-specific delays, identifying which resources cause delays in each case",
            "requirements": ["case_id", "activity", "actual_time", "resource"],
            "output": "Abnormal gaps with case identification, colored by resource",
            "color_impact": "Resource coloring reveals which resources are involved in case-specific delays"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots, colored by resource",
            "interpretation": "Detects time periods with high event concentration across cases. Color-coded by resource to see resource distribution.",
            "use_case": "Finding peak workload periods, identifying which resources are active across cases",
            "requirements": ["actual_time", "case_id", "resource"],
            "output": "Temporal clusters with event density, time ranges, and resource distribution",
            "color_impact": "Resource coloring shows which resources are active during busy periods"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by resource",
            "interpretation": "Detects cases with unusual execution patterns or extreme durations. Color-coded by resource for resource-level analysis.",
            "use_case": "Finding exceptional cases, identifying which resources are involved in outlier cases",
            "requirements": ["actual_time", "case_id", "resource"],
            "output": "Outlier cases with resource-specific anomaly reasons",
            "color_impact": "Resource coloring helps identify which resources are involved in case outliers"
        }
    },

    # ========================================================================
    # NON-MEANINGFUL COMBINATIONS (Examples)
    # ========================================================================
    
    ("logical_time", "resource", "case_id"): {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_time is a sequential counter, not actual time. Gap detection requires temporal data.",
            "use_case": "Use actual_time or relative_time instead",
            "requirements": [],
            "output": "N/A",
            "color_impact": "N/A"
        },
        "temporal_cluster_x": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_time has no temporal meaning, just sequential order.",
            "use_case": "Use actual_time for temporal analysis",
            "requirements": [],
            "output": "N/A",
            "color_impact": "N/A"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Red dots on sequential outliers",
            "interpretation": "Limited meaning: detects events out of typical sequence order, but not time-based.",
            "use_case": "Better to use actual_time for meaningful outlier detection",
            "requirements": ["logical_time"],
            "output": "Sequential position outliers",
            "color_impact": "Case coloring shows which cases have sequential outliers, but limited value"
        }
    },

    ("relative_ratio", "resource", "case_id"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing normalized gaps in resource activity",
            "interpretation": "Detects abnormal waiting times between activities, normalized by case duration. Shows which resources have delays relative to total case time. Color-coded by case.",
            "use_case": "Comparing delays across cases of different lengths, identifying resource bottlenecks independent of case duration",
            "requirements": ["case_id", "activity", "relative_ratio"],
            "output": "Abnormal gaps with normalized time ratios, colored by case",
            "color_impact": "Case coloring helps compare normalized delays across different cases"
        },
        "temporal_cluster_x": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: relative_ratio is not a time measurement.",
            "use_case": "Use temporal axis for temporal analysis",
            "requirements": [],
            "output": "N/A",
            "color_impact": "N/A"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Current implementation detects outliers based on actual_time, not relative_ratio. Results don't match visualization.",
            "use_case": "Use actual_time or relative_time views for meaningful outlier detection",
            "requirements": [],
            "output": "N/A",
            "color_impact": "N/A"
        }
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pattern_info(
    x_axis: str, 
    y_axis: str, 
    color: str, 
    pattern_name: str
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a pattern for a specific view configuration.

    Parameters
    ----------
    x_axis : str
        X-axis column name (e.g., 'actual_time', 'relative_time')
    y_axis : str
        Y-axis column name (e.g., 'resource', 'activity', 'case_id')
    color : str
        Color/dot color column name (e.g., 'case_id', 'activity', 'resource')
    pattern_name : str
        Pattern name (e.g., 'gap', 'temporal_cluster_x', 'outlier')

    Returns
    -------
    Optional[Dict[str, Any]]
        Pattern information dictionary or None if not found
    """
    view_key = (x_axis, y_axis, color)
    return EXTENDED_PATTERN_MATRIX.get(view_key, {}).get(pattern_name.lower())


def is_pattern_meaningful(
    x_axis: str, 
    y_axis: str, 
    color: str, 
    pattern_name: str
) -> bool:
    """
    Check if a pattern is both technically possible AND semantically meaningful.

    Parameters
    ----------
    x_axis : str
        X-axis column name
    y_axis : str
        Y-axis column name
    color : str
        Color/dot color column name
    pattern_name : str
        Pattern name

    Returns
    -------
    bool
        True if pattern is meaningful for this view configuration
    """
    info = get_pattern_info(x_axis, y_axis, color, pattern_name)
    if not info:
        return False
    return info.get("can_be_found", False) and info.get("makes_sense", False)


def get_meaningful_patterns(
    x_axis: str, 
    y_axis: str, 
    color: str
) -> List[str]:
    """
    Get list of all meaningful patterns for a view configuration.

    Parameters
    ----------
    x_axis : str
        X-axis column name
    y_axis : str
        Y-axis column name
    color : str
        Color/dot color column name

    Returns
    -------
    list[str]
        List of meaningful pattern names
    """
    view_key = (x_axis, y_axis, color)
    view_patterns = EXTENDED_PATTERN_MATRIX.get(view_key, {})

    return [
        pattern_name
        for pattern_name, info in view_patterns.items()
        if info.get("can_be_found") and info.get("makes_sense")
    ]


def get_all_view_combinations() -> List[Tuple[str, str, str]]:
    """
    Get all defined view combinations (x, y, color) in the matrix.

    Returns
    -------
    list[tuple]
        List of (x_axis, y_axis, color) tuples
    """
    return list(EXTENDED_PATTERN_MATRIX.keys())


def get_color_impact(
    x_axis: str, 
    y_axis: str, 
    color: str, 
    pattern_name: str
) -> Optional[str]:
    """
    Get the impact description of the color dimension for a specific pattern.

    Parameters
    ----------
    x_axis : str
        X-axis column name
    y_axis : str
        Y-axis column name
    color : str
        Color/dot color column name
    pattern_name : str
        Pattern name

    Returns
    -------
    Optional[str]
        Description of how color affects pattern interpretation, or None if not found
    """
    info = get_pattern_info(x_axis, y_axis, color, pattern_name)
    if info:
        return info.get("color_impact")
    return None


__all__ = [
    "EXTENDED_PATTERN_MATRIX",
    "PATTERN_BASE_DEFINITIONS",
    "get_pattern_info",
    "is_pattern_meaningful",
    "get_meaningful_patterns",
    "get_all_view_combinations",
    "get_color_impact"
]
