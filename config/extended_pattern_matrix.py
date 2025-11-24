"""
Extended Pattern Matrix for Paper Documentation and Frontend Filtering.

This matrix defines which patterns are detectable and meaningful for each view configuration.
Each pattern entry contains comprehensive metadata for documentation and UI purposes.
"""

from typing import Dict, Any, Optional, List

EXTENDED_PATTERN_MATRIX: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ========== TIME-BASED VIEWS ==========
    
    "actual_time_resource": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps",
            "interpretation": "Detects abnormal waiting times between consecutive activities within cases. Shows which resources have process delays.",
            "algorithm": "Transition-specific normality learning using IQR and P95 thresholds",
            "use_case": "Finding process bottlenecks, resource unavailability, weekend delays",
            "requirements": ["case_id", "activity", "actual_time"],
            "output": "List of abnormal gaps with severity scores (duration/threshold)",
            "x_axis": "actual_time",
            "y_axis": "resource"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored regions highlighting dense temporal periods",
            "interpretation": "Identifies periods when resources are particularly busy or idle. Shows resource work patterns and shift structures.",
            "algorithm": "DBSCAN/OPTICS clustering on temporal event distribution",
            "use_case": "Understanding resource utilization, identifying shift patterns, workload analysis",
            "requirements": ["actual_time", "resource"],
            "output": "Temporal clusters with event density and time ranges",
            "x_axis": "actual_time",
            "y_axis": "resource"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events",
            "interpretation": "Detects unusual events based on timing, resource behavior, and frequency patterns.",
            "algorithm": "IQR-based statistical analysis for time, duration, frequency, and resource anomalies",
            "use_case": "Finding exceptional cases, data quality issues, process violations",
            "requirements": ["actual_time", "resource"],
            "output": "List of outlier events with detection reasons and confidence scores",
            "x_axis": "actual_time",
            "y_axis": "resource"
        }
    },
    
    "actual_time_activity": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing time spans of abnormal gaps",
            "interpretation": "Detects abnormal waiting times between specific activity transitions. Shows which activity sequences have delays.",
            "algorithm": "Transition-specific normality learning using IQR and P95 thresholds",
            "use_case": "Identifying bottlenecks in specific process steps, analyzing handover times",
            "requirements": ["case_id", "activity", "actual_time"],
            "output": "List of abnormal gaps per transition (Activity A → Activity B)",
            "x_axis": "actual_time",
            "y_axis": "activity"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored regions highlighting when specific activities occur",
            "interpretation": "Shows when specific activities typically happen (e.g., all payments happen in morning).",
            "algorithm": "DBSCAN clustering per activity type",
            "use_case": "Understanding activity timing patterns, seasonal effects on activities",
            "requirements": ["actual_time", "activity"],
            "output": "Temporal clusters grouped by activity",
            "x_axis": "actual_time",
            "y_axis": "activity"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events",
            "interpretation": "Detects activities that happen at unusual times or with unusual frequency.",
            "algorithm": "IQR-based statistical analysis for time and frequency anomalies",
            "use_case": "Finding rare activities, off-hours events, frequency anomalies",
            "requirements": ["actual_time", "activity"],
            "output": "Outlier events with activity-specific anomaly reasons",
            "x_axis": "actual_time",
            "y_axis": "activity"
        }
    },
    
    "actual_time_case_id": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps within individual case timelines",
            "interpretation": "Detects abnormal waiting times within specific cases. Each case is analyzed individually.",
            "algorithm": "Transition-specific normality learning per case",
            "use_case": "Finding case-specific delays, comparing case execution times",
            "requirements": ["case_id", "activity", "actual_time"],
            "output": "Abnormal gaps with case identification",
            "x_axis": "actual_time",
            "y_axis": "case_id"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: Each case is independent, temporal clustering across cases doesn't provide useful insights.",
            "algorithm": "N/A",
            "use_case": "Use actual_time × resource or actual_time × activity instead",
            "requirements": [],
            "output": "N/A",
            "x_axis": "actual_time",
            "y_axis": "case_id"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots on outlier events within cases",
            "interpretation": "Detects cases with unusual execution patterns or extreme durations.",
            "algorithm": "IQR-based case duration and complexity analysis",
            "use_case": "Finding exceptional cases, compliance violations",
            "requirements": ["actual_time", "case_id"],
            "output": "Outlier cases with anomaly reasons",
            "x_axis": "actual_time",
            "y_axis": "case_id"
        }
    },
    
    "relative_time_resource": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps in process execution time",
            "interpretation": "Detects delays at specific stages of the process, regardless of calendar time.",
            "algorithm": "Transition-specific normality learning",
            "use_case": "Finding process bottlenecks that occur at specific process stages",
            "requirements": ["case_id", "activity", "relative_time"],
            "output": "Abnormal gaps with process-relative timing",
            "x_axis": "relative_time",
            "y_axis": "resource"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored regions showing when resources typically get involved",
            "interpretation": "Shows at which process stage resources typically become active.",
            "algorithm": "DBSCAN clustering on relative time per resource",
            "use_case": "Understanding resource involvement patterns in process flow",
            "requirements": ["relative_time", "resource"],
            "output": "Clusters showing resource involvement timing",
            "x_axis": "relative_time",
            "y_axis": "resource"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots",
            "interpretation": "Detects unusual resource behavior at specific process stages.",
            "algorithm": "IQR-based analysis",
            "use_case": "Finding resource anomalies in process execution",
            "requirements": ["relative_time", "resource"],
            "output": "Outlier events with resource behavior anomalies",
            "x_axis": "relative_time",
            "y_axis": "resource"
        }
    },
    
    "relative_time_activity": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps between activities",
            "interpretation": "Detects abnormal delays between activities in process flow, independent of calendar time.",
            "algorithm": "Transition-specific normality learning",
            "use_case": "Analyzing process flow efficiency, finding bottlenecks in activity sequences",
            "requirements": ["case_id", "activity", "relative_time"],
            "output": "Abnormal transition gaps",
            "x_axis": "relative_time",
            "y_axis": "activity"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Colored regions showing typical activity timing in process",
            "interpretation": "Shows when in the process flow specific activities typically occur.",
            "algorithm": "DBSCAN clustering per activity",
            "use_case": "Understanding activity sequencing patterns",
            "requirements": ["relative_time", "activity"],
            "output": "Activity timing clusters in process flow",
            "x_axis": "relative_time",
            "y_axis": "activity"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red highlighted dots",
            "interpretation": "Detects activities happening at unusual process stages.",
            "algorithm": "IQR-based analysis",
            "use_case": "Finding process violations, unusual activity order",
            "requirements": ["relative_time", "activity"],
            "output": "Outlier activities in process flow",
            "x_axis": "relative_time",
            "y_axis": "activity"
        }
    },
    
    "relative_time_case_id": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing gaps within individual case timelines",
            "interpretation": "Detects abnormal delays at specific stages of process execution within each case.",
            "algorithm": "Transition-specific normality learning per case",
            "use_case": "Finding process bottlenecks at specific execution stages, comparing case flow efficiency",
            "requirements": ["case_id", "activity", "relative_time"],
            "output": "Abnormal gaps with process-relative timing per case",
            "x_axis": "relative_time",
            "y_axis": "case_id"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_time",
            "y_axis": "case_id"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_time",
            "y_axis": "case_id"
        }
    },
    
    "relative_ratio_resource": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing normalized gaps in resource activity",
            "interpretation": "Detects abnormal waiting times between activities, normalized by case duration. Shows which resources have delays relative to total case time.",
            "algorithm": "Transition-specific normality learning using normalized time ratios",
            "use_case": "Comparing delays across cases of different lengths, identifying resource bottlenecks independent of case duration",
            "requirements": ["case_id", "activity", "relative_ratio"],
            "output": "Abnormal gaps with normalized time ratios",
            "x_axis": "relative_ratio",
            "y_axis": "resource"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "resource"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "resource"
        }
    },
    
    "relative_ratio_activity": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing normalized gaps between activities",
            "interpretation": "Detects abnormal delays between activity transitions, normalized by case duration. Enables comparison across cases of different lengths.",
            "algorithm": "Transition-specific normality learning using normalized time ratios",
            "use_case": "Finding relative bottlenecks in activity sequences, comparing process efficiency across fast and slow cases",
            "requirements": ["case_id", "activity", "relative_ratio"],
            "output": "Abnormal gaps with normalized time ratios per transition",
            "x_axis": "relative_ratio",
            "y_axis": "activity"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "activity"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "activity"
        }
    },
    
    "relative_ratio_case_id": {
        "gap": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "Red rectangles showing normalized gaps within individual cases",
            "interpretation": "Detects abnormal delays within cases on a normalized scale [0,1]. Useful for comparing delay patterns across cases of vastly different durations.",
            "algorithm": "Transition-specific normality learning using normalized time ratios",
            "use_case": "Identifying relative bottlenecks within cases, detecting structural delays independent of absolute time",
            "requirements": ["case_id", "activity", "relative_ratio"],
            "output": "Abnormal gaps with normalized positions within case timeline",
            "x_axis": "relative_ratio",
            "y_axis": "case_id"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "case_id"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "case_id"
        }
    },
    
    "logical_relative_resource": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_relative is a sequential index, not temporal data. Gap detection requires actual time measurements.",
            "algorithm": "N/A",
            "use_case": "Use relative_time or relative_ratio for meaningful time-based gap detection",
            "requirements": [],
            "output": "N/A",
            "x_axis": "logical_relative",
            "y_axis": "resource"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_relative",
            "y_axis": "resource"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_relative",
            "y_axis": "resource"
        }
    },
    
    "logical_relative_activity": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_relative is a sequential index without temporal meaning.",
            "algorithm": "N/A",
            "use_case": "Use relative_time or relative_ratio instead",
            "requirements": [],
            "output": "N/A",
            "x_axis": "logical_relative",
            "y_axis": "activity"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_relative",
            "y_axis": "activity"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_relative",
            "y_axis": "activity"
        }
    },
    
    "logical_relative_case_id": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_relative is just event numbering, not time-based.",
            "algorithm": "N/A",
            "use_case": "Use relative_time or relative_ratio for temporal analysis",
            "requirements": [],
            "output": "N/A",
            "x_axis": "logical_relative",
            "y_axis": "case_id"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_relative",
            "y_axis": "case_id"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_relative",
            "y_axis": "case_id"
        }
    },
    
    # ========== NON-TEMPORAL VIEWS ==========
    
    "logical_time_resource": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_time is a sequential counter, not actual time. Gap detection requires temporal data.",
            "algorithm": "N/A",
            "use_case": "Use actual_time or relative_time instead",
            "requirements": [],
            "output": "N/A",
            "x_axis": "logical_time",
            "y_axis": "resource"
        },
        "temporal_cluster": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_time has no temporal meaning, just sequential order.",
            "algorithm": "N/A",
            "use_case": "Use actual_time for temporal analysis",
            "requirements": [],
            "output": "N/A",
            "x_axis": "logical_time",
            "y_axis": "resource"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": False,
            "visual": "Red dots on sequential outliers",
            "interpretation": "Limited meaning: detects events out of typical sequence order, but not time-based.",
            "algorithm": "Sequential position analysis",
            "use_case": "Better to use actual_time for meaningful outlier detection",
            "requirements": ["logical_time"],
            "output": "Sequential position outliers",
            "x_axis": "logical_time",
            "y_axis": "resource"
        }
    },
    
    "logical_time_activity": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_time lacks temporal information needed for gap detection.",
            "algorithm": "N/A",
            "use_case": "Use actual_time or relative_time instead",
            "requirements": [],
            "output": "N/A",
            "x_axis": "logical_time",
            "y_axis": "activity"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_time",
            "y_axis": "activity"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_time",
            "y_axis": "activity"
        }
    },
    
    "logical_time_case_id": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: logical_time is just sequential ordering, no temporal data for gap analysis.",
            "algorithm": "N/A",
            "use_case": "Use actual_time or relative_time for meaningful gap detection",
            "requirements": [],
            "output": "N/A",
            "x_axis": "logical_time",
            "y_axis": "case_id"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_time",
            "y_axis": "case_id"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "logical_time",
            "y_axis": "case_id"
        }
    },
    
    # ========== EVENT INDEX VIEWS ==========
    
    "actual_time_event_index": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: event_index is a counter within each case. Gap detection on Y-axis event numbers doesn't provide useful process insights.",
            "algorithm": "N/A",
            "use_case": "Use case_id or activity on Y-axis instead",
            "requirements": [],
            "output": "N/A",
            "x_axis": "actual_time",
            "y_axis": "event_index"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "actual_time",
            "y_axis": "event_index"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "actual_time",
            "y_axis": "event_index"
        }
    },
    
    "relative_time_event_index": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: event_index doesn't provide semantic grouping for gap analysis.",
            "algorithm": "N/A",
            "use_case": "Use activity or resource on Y-axis for meaningful patterns",
            "requirements": [],
            "output": "N/A",
            "x_axis": "relative_time",
            "y_axis": "event_index"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_time",
            "y_axis": "event_index"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_time",
            "y_axis": "event_index"
        }
    },
    
    "relative_ratio_event_index": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: event_index lacks semantic meaning for gap patterns.",
            "algorithm": "N/A",
            "use_case": "Choose activity or resource on Y-axis",
            "requirements": [],
            "output": "N/A",
            "x_axis": "relative_ratio",
            "y_axis": "event_index"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "event_index"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "event_index"
        }
    },
    
    # ========== VARIANT VIEWS ==========
    
    "actual_time_variant": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: Gap detection analyzes transitions WITHIN cases. Variants are case-level groupings and don't contain transition information.",
            "algorithm": "N/A",
            "use_case": "Use activity or case_id on Y-axis for transition-based gap detection",
            "requirements": [],
            "output": "N/A",
            "x_axis": "actual_time",
            "y_axis": "variant"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "actual_time",
            "y_axis": "variant"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "actual_time",
            "y_axis": "variant"
        }
    },
    
    "relative_time_variant": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: Variants don't provide event-level transitions needed for gap detection.",
            "algorithm": "N/A",
            "use_case": "Use activity or case_id on Y-axis",
            "requirements": [],
            "output": "N/A",
            "x_axis": "relative_time",
            "y_axis": "variant"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_time",
            "y_axis": "variant"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_time",
            "y_axis": "variant"
        }
    },
    
    "relative_ratio_variant": {
        "gap": {
            "can_be_found": False,
            "makes_sense": False,
            "visual": "N/A",
            "interpretation": "Not meaningful: Variant is too high-level for gap detection.",
            "algorithm": "N/A",
            "use_case": "Use activity or case_id on Y-axis for detailed gap analysis",
            "requirements": [],
            "output": "N/A",
            "x_axis": "relative_ratio",
            "y_axis": "variant"
        },
        "temporal_cluster": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "variant"
        },
        "outlier": {
            "can_be_found": True,
            "makes_sense": True,
            "visual": "TODO: To be filled by Tai/Anna",
            "interpretation": "TODO: To be filled by Tai/Anna",
            "algorithm": "TODO: To be filled by Tai/Anna",
            "use_case": "TODO: To be filled by Tai/Anna",
            "requirements": [],
            "output": "TODO: To be filled by Tai/Anna",
            "x_axis": "relative_ratio",
            "y_axis": "variant"
        }
    }
}


def get_pattern_info(x_axis: str, y_axis: str, pattern_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a pattern for a specific view configuration.
    
    Parameters
    ----------
    x_axis : str
        X-axis column name (e.g., 'actual_time', 'relative_time')
    y_axis : str
        Y-axis column name (e.g., 'resource', 'activity', 'case_id')
    pattern_name : str
        Pattern name (e.g., 'gap', 'temporal_cluster', 'outlier')
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Pattern information dictionary or None if not found
    """
    view_key = f"{x_axis}_{y_axis}"
    return EXTENDED_PATTERN_MATRIX.get(view_key, {}).get(pattern_name.lower())


def is_pattern_meaningful(x_axis: str, y_axis: str, pattern_name: str) -> bool:
    """
    Check if a pattern is both technically possible AND semantically meaningful.
    
    Parameters
    ----------
    x_axis : str
        X-axis column name
    y_axis : str
        Y-axis column name
    pattern_name : str
        Pattern name
        
    Returns
    -------
    bool
        True if pattern is meaningful for this view configuration
    """
    info = get_pattern_info(x_axis, y_axis, pattern_name)
    if not info:
        return False
    return info.get("can_be_found", False) and info.get("makes_sense", False)


def get_meaningful_patterns(x_axis: str, y_axis: str) -> List[str]:
    """
    Get list of all meaningful patterns for a view configuration.
    
    Parameters
    ----------
    x_axis : str
        X-axis column name
    y_axis : str
        Y-axis column name
        
    Returns
    -------
    list[str]
        List of meaningful pattern names
    """
    view_key = f"{x_axis}_{y_axis}"
    view_patterns = EXTENDED_PATTERN_MATRIX.get(view_key, {})
    
    return [
        pattern_name
        for pattern_name, info in view_patterns.items()
        if info.get("can_be_found") and info.get("makes_sense")
    ]


__all__ = [
    "EXTENDED_PATTERN_MATRIX",
    "get_pattern_info",
    "is_pattern_meaningful",
    "get_meaningful_patterns"
]

