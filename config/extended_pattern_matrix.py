"""
Extended Pattern Matrix for Paper Documentation and Frontend Filtering.

This matrix defines which patterns are detectable and meaningful for each view configuration.
Each pattern entry contains comprehensive metadata for documentation and UI purposes.

Structure: Tuple-based keys (x_axis, y_axis, color) for explicit 3D view configuration.
"""

from typing import Dict, Any, Optional, List, Tuple

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
            "visual": "Red rectangles showing time spans of abnormal gaps between activities. Case coloring helps identify which specific cases experience delays at each resource.",
            "interpretation": "Detects abnormal waiting times between consecutive activities within cases. Shows which resources have process delays. Color-coded by case for case-specific analysis.",
            "use_case": "Finding process bottlenecks, resource unavailability, weekend delays per case",
            "output": "List of abnormal gaps with severity scores, grouped by resource and colored by case"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots showing temporal bursts of resource activity. Case coloring reveals whether busy periods are caused by many cases or few intensive cases.",
            "interpretation": "Detects time periods with high event concentration for resources. Color-coded by case to see case distribution in busy periods.",
            "use_case": "Finding peak workload periods per resource, identifying batch processing times",
            "output": "Temporal clusters with event density, time ranges, and case distribution"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted circles over outlier events with anomaly reasons. Case coloring helps identify whether outliers are case-specific or resource-specific.",
            "interpretation": "Detects unusual events based on timing, resource behavior, and frequency patterns. Color-coded by case for case-level anomaly analysis.",
            "use_case": "Finding exceptional cases, data quality issues, resource violations",
            "output": "Outlier events with resource-specific and case-specific anomaly reasons"
        }, 
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored rectangles or highlights showing detected clusters of events based on resource and time. Case coloring shows which cases are in each cluster.",
            "interpretation": "f you use case ID as color, you’ll get a unique color per case, which is visually overwhelming and not useful for pattern discovery.",
            "use_case": "Not recommended.",
            "output": "N/A"

        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored horizontal lines showing sequences of cases processed by each resource over time. Colors show different sequences.",
            "interpretation": "Detects sequences of cases processed by resources over time. A resource often works in a specific order on cases.",
            "use_case": "Finding cases that are frequently processed after each other by the same resource.",
            "output": "List of detected sequences with frequency counts."
        }
    },

    # ========================================================================
    # ACTUAL_TIME × RESOURCE × ACTIVITY
    # ========================================================================
    ("actual_time", "resource", "activity"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps between activities. Activity coloring reveals which specific activities are delayed at each resource.",
            "interpretation": "Detects abnormal waiting times between consecutive activities within cases. Shows which resources have process delays. Color-coded by activity to see which activities cause delays.",
            "use_case": "Finding process bottlenecks per resource, identifying which activities cause delays",
            "output": "List of abnormal gaps with severity scores, grouped by resource and colored by activity"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots showing temporal bursts, colored by activity. Activity coloring shows which activities dominate busy periods.",
            "interpretation": "Detects time periods with high event concentration for resources. Color-coded by activity to see activity distribution in busy periods.",
            "use_case": "Finding peak workload periods per resource, identifying which activities cluster",
            "output": "Temporal clusters with event density, time ranges, and activity distribution"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted circles over outlier events, colored by activity. Activity coloring helps identify which activities are outliers at each resource.",
            "interpretation": "Detects unusual events based on timing, resource behavior, and frequency patterns. Color-coded by activity for activity-level anomaly analysis.",
            "use_case": "Finding exceptional resource-activity combinations, rare activities",
            "output": "Outlier events with resource-activity-specific anomaly reasons"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored rectangles or highlights showing detected clusters of events based on resource and time. Activity coloring shows which activities are in each cluster.",
            "interpretation": "Groups events that are similar in time and resource. Color-coded by activity to show activity distribution in clusters.",
            "use_case": "Finding groups of activities that are processed similarly by resources over time.",
            "output": "Cluster assignments for each event, colored by activity."
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored horizontal lines showing sequences of activities executed by each resource over time. Colors show different sequences.",
            "interpretation": "Detects sequences of activities executed by resources over time. A resource often performs activities in a specific order.",
            "use_case": "Finding activities that are frequently executed after each other by the same resource.",
            "output": "List of detected sequences with frequency counts."
        }
    },

    # ========================================================================
    # ACTUAL_TIME × RESOURCE × RESOURCE (Same dimension on Y and Color)
    # ========================================================================
    ("actual_time", "resource", "resource"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps, colored by resource. Resource coloring provides redundant but clear visual separation of resources.",
            "interpretation": "Detects abnormal waiting times between consecutive activities within cases. Shows which resources have process delays. Color matches Y-axis for clear resource identification.",
            "use_case": "Finding process bottlenecks per resource with clear visual resource separation",
            "output": "List of abnormal gaps with severity scores, grouped and colored by resource"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored circles around event dots, colored by resource. Resource coloring provides redundant but clear visual separation.",
            "interpretation": "Does not make sense to cluster by resource when resource is both Y-axis and color. Color matches Y-axis for clear resource identification.",
            "use_case": "Finding peak workload periods per resource with clear visual separation",
            "output": "Temporal clusters with event density and time ranges per resource"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted circles over outlier events, colored by resource. Resource coloring provides redundant but clear visual separation.",
            "interpretation": "Detects unusual events based on timing, resource behavior, and frequency patterns. Color matches Y-axis for clear resource identification.",
            "use_case": "Finding exceptional resource behavior with clear visual separation",
            "output": "Outlier events with resource-specific anomaly reasons"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored rectangles or highlights showing detected clusters of events based on resource and time. Resource coloring provides redundant but clear visual separation of resources.",
            "interpretation": "Grouping by resource does not reveal new patterns, as each resource is its own group.",
            "use_case": "Finding groups of resources that are processed similarly over time.",
            "output": "Cluster assignments for each event, colored by resource."
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored horizontal lines showing sequences of resource used by each resource over time. Colors show different sequences.",
            "interpretation": "Detects each resource as one sequence, which is redundant.",
            "use_case": "N/A",
            "output": "List of all resources."
        }
    },

    # ========================================================================
    # ACTUAL_TIME × ACTIVITY × CASE_ID
    # ========================================================================
    ("actual_time", "activity", "case_id"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps between activities. Case coloring reveals which cases experience delays in specific activity transitions.",
            "interpretation": "Detects abnormal waiting times between specific activity transitions. Shows which activity sequences have delays. Color-coded by case for case-specific analysis.",
            "use_case": "Identifying bottlenecks in specific process steps, analyzing handover times per case",
            "output": "List of abnormal gaps per transition (Activity A → Activity B), colored by case"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots showing temporal bursts of activities. Case coloring shows whether activity bursts involve many cases or few intensive cases.",
            "interpretation": "Detects time periods with high event concentration for activities. Color-coded by case to see case distribution.",
            "use_case": "Finding peak workload periods per activity, batch processing detection",
            "output": "Temporal clusters with event density, time ranges, and case distribution"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by case. Case coloring helps identify whether activity outliers are case-specific.",
            "interpretation": "Detects activities that happen at unusual times or with unusual frequency. Color-coded by case for case-level analysis.",
            "use_case": "Finding rare activities, off-hours events, frequency anomalies per case",
            "output": "Outlier events with activity-specific and case-specific anomaly reasons"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored rectangles or highlights showing detected clusters of events based on activity and time. Case coloring shows which cases are in each cluster.",
            "interpretation": "If you use case ID as color, you’ll get a unique color per case, which is visually overwhelming and not useful for pattern discovery.",
            "use_case": "Not recommended.",
            "output": "N/A"
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored horizontal lines showing sequences of case ids worked on by each activity over time. Colors show different sequences.",
            "interpretation": "Detects sequences of case ids worked on by activity over time. An activity often works in a specific order on cases.",
            "use_case": "Finding case ids that are frequently worked on after each other by the same activity.",
            "output": "List of detected sequences with frequency counts."
        }
    },

    # ========================================================================
    # ACTUAL_TIME × ACTIVITY × ACTIVITY (Same dimension on Y and Color)
    # ========================================================================
    ("actual_time", "activity", "activity"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps, colored by activity. Activity coloring provides redundant but clear visual separation of activities.",
            "interpretation": "Detects abnormal waiting times between specific activity transitions. Color matches Y-axis for clear activity identification.",
            "use_case": "Identifying bottlenecks in specific process steps with clear visual separation",
            "output": "List of abnormal gaps per transition (Activity A → Activity B), colored by activity"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored circles around event dots, colored by activity. Activity coloring provides redundant but clear visual separation.",
            "interpretation": "Does not make sense to cluster by activity when activity is both Y-axis and color. Color matches Y-axis for clear activity identification.",
            "use_case": "Finding peak workload periods per activity with clear visual separation",
            "output": "Temporal clusters with event density and time ranges per activity"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by activity. Activity coloring provides redundant but clear visual separation.",
            "interpretation": "Detects activities that happen at unusual times or with unusual frequency. Color matches Y-axis for clear activity identification.",
            "use_case": "Finding rare activities, off-hours events with clear visual separation",
            "output": "Outlier events with activity-specific anomaly reasons"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored rectangles or highlights showing detected clusters of events based on activity and time. Activity coloring provides redundant but clear visual separation of activities.",
            "interpretation": "Grouping by activity does not reveal new patterns, as each activity is its own group.",
            "use_case": "Finding groups of activities that are processed similarly over time.",
            "output": "Cluster assignments for each event, colored by activity."
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored horizontal lines showing sequences of activities used by each activity over time. Colors show different sequences.",
            "interpretation": "Detects each activity as one sequence, which is redundant.",
            "use_case": "N/A",
            "output": "List of all activities."
        }
    },

    # ========================================================================
    # ACTUAL_TIME × ACTIVITY × RESOURCE
    # ========================================================================
    ("actual_time", "activity", "resource"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps, colored by resource. Resource coloring reveals which resources are involved in activity transition delays.",
            "interpretation": "Detects abnormal waiting times between specific activity transitions. Shows which activity sequences have delays. Color-coded by resource to see resource involvement.",
            "use_case": "Identifying bottlenecks in specific process steps, analyzing which resources cause delays",
            "output": "List of abnormal gaps per transition (Activity A → Activity B), colored by resource"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored circles around event dots, colored by resource. Resource coloring shows which resources are active during activity bursts.",
            "interpretation": "Detects time periods with high event concentration for activities. Color-coded by resource to see resource distribution.",
            "use_case": "Finding peak workload periods per activity, identifying resource involvement",
            "output": "Temporal clusters with event density, time ranges, and resource distribution"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by resource. Resource coloring helps identify which resources are involved in activity outliers.",
            "interpretation": "Detects activities that happen at unusual times or with unusual frequency. Color-coded by resource for resource-level analysis.",
            "use_case": "Finding rare activities, identifying which resources perform unusual activities",
            "output": "Outlier events with activity-resource-specific anomaly reasons"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored rectangles or highlights showing detected clusters of events based on activity and time. Resource coloring shows which resources are in each cluster.",
            "interpretation": "Groups events that are similar in time and activity. Color-coded by resource to show resource distribution in clusters.",
            "use_case": "Finding groups of resources that are processed similarly by activities over time.",
            "output": "Cluster assignments for each event, colored by resource."
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored horizontal lines showing sequences of resources that are used by an activity over time. Colors show different sequences.",
            "interpretation": "Detects sequences of resources that are used by an activity over time.",
            "use_case": "Finding resources that are frequently used after each other by the same activity.",
            "output": "List of detected sequences with frequency counts."
        }
    },

    # ========================================================================
    # ACTUAL_TIME × CASE_ID × CASE_ID (Same dimension on Y and Color)
    # ========================================================================
    ("actual_time", "case_id", "case_id"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps within individual case timelines. Case coloring provides redundant but clear visual separation of cases.",
            "interpretation": "Detects abnormal waiting times within specific cases. Each case is analyzed individually. Color matches Y-axis for clear case identification.",
            "use_case": "Finding case-specific delays, comparing case execution times with clear visual separation",
            "output": "Abnormal gaps with case identification, colored by case"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored circles around event dots, colored by case. Case coloring provides redundant but clear visual separation.",
            "interpretation": "Detects time periods with high event concentration across cases. Color matches Y-axis for clear case identification.",
            "use_case": "Not recommended.",
            "output": "Temporal clusters with event density and time ranges per case"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Clusters are not meaningful when using case_id as axis or color, since each case is unique.",
            "interpretation": "Clustering by case_id does not reveal process patterns, as each case is its own group.",
            "use_case": "Not recommended.",
            "output": "N/A"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events within cases. Case coloring provides redundant but clear visual separation.",
            "interpretation": "Detects cases with unusual execution patterns or extreme durations. Color matches Y-axis for clear case identification.",
            "use_case": "Finding exceptional cases, compliance violations with clear visual separation",
            "output": "Outlier cases with anomaly reasons, colored by case"
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored horizontal lines showing sequences of case ids processed by each case id over time. Colors show different sequences.",
            "interpretation": "Detects each case id as one sequence, which is redundant.",
            "use_case": "N/A",
            "output": "List of all case ids."
        }
    },

    # ========================================================================
    # ACTUAL_TIME × CASE_ID × ACTIVITY
    # ========================================================================
    ("actual_time", "case_id", "activity"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps within individual case timelines, colored by activity. Activity coloring reveals which activities are delayed within each case.",
            "interpretation": "Detects abnormal waiting times within specific cases. Each case is analyzed individually. Color-coded by activity to see which activities cause delays.",
            "use_case": "Finding case-specific delays, identifying which activities cause delays in each case",
            "output": "Abnormal gaps with case identification, colored by activity"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored circles around event dots, colored by activity. Activity coloring shows which activities dominate busy periods across cases.",
            "interpretation": "Clustering by case_id does not reveal process patterns, as each case is its own group.",
            "use_case": "Not recommended.",
            "output": "N/A"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Clusters are not meaningful when using case_id as axis or color, since each case is unique. It separates data by instance, not by any meaningful pattern.",
            "interpretation": "Clustering by case_id does not reveal process patterns, as each case is its own group.",
            "use_case": "Not recommended.The core issue: When case_id is on an axis, you're viewing individual case trajectories. Clustering is designed to find groups of similar behavior, which conflicts with the case-by-case perspective.",
            "output": "N/A"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by activity. Activity coloring helps identify which activities are outliers within cases.",
            "interpretation": "Detects cases with unusual execution patterns or extreme durations. Color-coded by activity for activity-level analysis.",
            "use_case": "Finding exceptional cases, identifying which activities are outliers in each case",
            "output": "Outlier cases with activity-specific anomaly reasons"
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored horizontal lines showing sequences of activities executed for each case id over time. Colors show different sequences.",
            "interpretation": "Detects sequences of activities executed for each case id over time.",
            "use_case": "Finding activities that are frequently executed after each other within cases.",
            "output": "List of detected sequences with frequency counts."
        }

    },

    # ========================================================================
    # ACTUAL_TIME × CASE_ID × RESOURCE
    # ========================================================================
    ("actual_time", "case_id", "resource"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps within individual case timelines, colored by resource. Resource coloring reveals which resources are involved in case-specific delays.",
            "interpretation": "Detects abnormal waiting times within specific cases. Each case is analyzed individually. Color-coded by resource to see resource involvement in delays.",
            "use_case": "Finding case-specific delays, identifying which resources cause delays in each case",
            "output": "Abnormal gaps with case identification, colored by resource"
        },
        "temporal_cluster_x": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Colored circles around event dots, colored by resource. Resource coloring shows which resources are active during busy periods.",
            "interpretation": "Using case ID as color will only separate data by instance, not by any meaningful temporal pattern.",
            "use_case": "Not recommended.",
            "output": "N/A"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Clusters are not meaningful when using case_id as axis or color, since each case is unique.",
            "interpretation": "Clustering by case_id does not reveal process patterns, as each case is its own group.",
            "use_case": "Not recommended.The core issue: When case_id is on an axis, you're viewing individual case trajectories. Clustering is designed to find groups of similar behavior, which conflicts with the case-by-case perspective.",
            "output": "N/A"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events, colored by resource. Resource coloring helps identify which resources are involved in case outliers.",
            "interpretation": "Detects cases with unusual execution patterns or extreme durations. Color-coded by resource for resource-level analysis.",
            "use_case": "Finding exceptional cases, identifying which resources are involved in outlier cases",
            "output": "Outlier cases with resource-specific anomaly reasons"
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored horizontal lines showing sequences of resources working on each case id over time. Colors show different sequences.",
            "interpretation": "Detects sequences of resources working on each case id over time.",
            "use_case": "Finding resources that frequently work after each other within cases.",
            "output": "List of detected sequences with frequency counts."
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
            "output": "N/A"
        },
        "temporal_cluster_x": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_time has no temporal meaning, just sequential order.",
            "use_case": "Use actual_time for temporal analysis",
            "output": "N/A"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Red dots on sequential outliers",
            "interpretation": "Limited meaning: detects events out of typical sequence order, but not time-based.",
            "use_case": "Better to use actual_time for meaningful outlier detection",
            "output": "Sequential position outliers"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "CaseID as color does not provide useful clustering information.",
            "use_case": "Not recommended.",
            "output": "N/A"
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored horizontal lines showing sequences of cases processed by each resource over logical time. Colors show different sequences. Sequences elements can be spaced further apart due to logical time.",
            "interpretation": "Detects sequences of cases processed by resources over logical time. A resource often works in a specific order on cases.",
            "use_case": "Finding cases that are frequently processed after each other by the same resource.",
            "output": "List of detected sequences with frequency counts."
        }
    },

    ("relative_ratio", "resource", "case_id"): {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing normalized gaps in resource activity. Case coloring helps compare normalized delays across different cases.",
            "interpretation": "Detects abnormal waiting times between activities, normalized by case duration. Shows which resources have delays relative to total case time. Color-coded by case.",
            "use_case": "Comparing delays across cases of different lengths, identifying resource bottlenecks independent of case duration",
            "output": "Abnormal gaps with normalized time ratios, colored by case"
        },
        "temporal_cluster_x": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: relative_ratio is not a time measurement.",
            "use_case": "Use temporal axis for temporal analysis",
            "output": "N/A"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Current implementation detects outliers based on actual_time, not relative_ratio. Results don't match visualization.",
            "use_case": "Use actual_time or relative_time views for meaningful outlier detection",
            "output": "N/A"
        },
        "cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: case_id as color does not provide useful clustering information.",
            "use_case": "Not recommended.",
            "output": "N/A"
        },
        "sequence": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored horizontal lines showing sequences of cases processed by each resource over relative time. Colors show different sequences. Sequences elements can be spaced closer due to relative time.",
            "interpretation": "Detects sequences of cases processed by resources over relative time. A resource often works in a specific order on cases.",
            "use_case": "Finding cases that are frequently processed after each other by the same resource.",
            "output": "List of detected sequences with frequency counts."
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


__all__ = [
    "EXTENDED_PATTERN_MATRIX",
    "get_pattern_info",
    "is_pattern_meaningful",
    "get_meaningful_patterns",
    "get_all_view_combinations"
]
