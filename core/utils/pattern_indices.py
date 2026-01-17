"""Collects point indices from visible patterns for focus mode."""


def get_all_pattern_indices(ss) -> set:
    """Collect indices of all points belonging to visible patterns."""
    idx = set()

    # Outlier
    if ss.get('visible_outlier') and ss.get('outlier_detected') and 'outlier_pattern' in ss:
        sel = ss.get('selected_outlier_indices')
        idx.update(sel if sel else ss.outlier_pattern.outliers.get('combined', []))

    # Cluster (OPTICS)
    if ss.get('visible_cluster') and ss.get('cluster_detected') and 'cluster_detector' in ss:
        d = ss.cluster_detector.detected
        if d and d.get('labels') is not None:
            idx.update(d['original_indices'][d['labels'] >= 0])

    # Sequence
    if ss.get('visible_sequence') and ss.get('sequence_detected') and 'sequence_detector' in ss:
        if ss.sequence_detector.df_found_patterns is not None:
            idx.update(ss.sequence_detector.df_found_patterns.index)

    # Temporal Cluster
    if ss.get('visible_temporal_cluster') and ss.get('temporal_detected') and 'temporal_clusters' in ss:
        for lst in getattr(ss.temporal_clusters, 'cluster_point_indices', {}).values():
            idx.update(lst)

    # Gap
    if ss.get('visible_gap') and 'gap_detector' in ss:
        for s in getattr(ss.gap_detector, 'gap_point_indices', {}).values():
            idx.update(s)

    return idx
