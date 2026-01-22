"""
Interactive Extended Pattern Matrix Viewer
Displays all view configurations with pattern support status and hover tooltips.
"""

import streamlit as st
from config.extended_pattern_matrix import (
    get_all_view_combinations,
    get_pattern_info
)
from typing import List, Dict, Tuple


# Define the pattern order and display names
PATTERNS = ["gap", "burst", "outlier", "cluster", "sequence"]
PATTERN_DISPLAY_NAMES = {
    "gap": "Gap",
    "burst": "Burst",
    "outlier": "Outlier",
    "cluster": "Cluster",
    "sequence": "Sequence"
}

# Map pattern names in matrix to display pattern names
PATTERN_NAME_MAPPING = {
    "temporal_cluster_x": "burst",
    "gap": "gap",
    "outlier": "outlier",
    "cluster": "cluster",
    "sequence": "sequence"
}


def get_all_view_configs() -> List[Tuple[str, str, str]]:
    """Get all unique view configurations from the matrix."""
    return get_all_view_combinations()


def format_dimension_name(dim: str) -> str:
    """Format dimension name for display."""
    if dim == "actual_time":
        return "actual_time"
    elif dim == "relative_ratio":
        return "relative_ratio"
    elif dim == "logical_time":
        return "logical_time"
    return dim.replace("_", " ").title()


def get_view_label(x: str, y: str, color: str) -> str:
    """Create a readable label for a view configuration."""
    x_fmt = format_dimension_name(x)
    y_fmt = format_dimension_name(y)
    color_fmt = format_dimension_name(color)
    return f"{x_fmt} × {y_fmt} × {color_fmt}"


def get_pattern_status(view_config: Tuple[str, str, str], pattern: str) -> Dict:
    """
    Get the status of a pattern for a given view configuration.

    Returns:
        Dict with keys: status ('found', 'not_meaningful', 'not_found'), tooltip
    """
    x, y, color = view_config

    # Check all possible pattern names in the matrix
    pattern_names_to_check = [pattern]
    if pattern == "burst":
        pattern_names_to_check = ["temporal_cluster_x", "burst"]

    info = None
    for pattern_name in pattern_names_to_check:
        info = get_pattern_info(x, y, color, pattern_name)
        if info:
            break

    if not info:
        return {
            "status": "not_found",
            "tooltip": "This pattern is not defined for this view configuration."
        }

    can_be_found = info.get("can_be_found", False)
    makes_sense = info.get("makes_sense", False)

    if not can_be_found:
        return {
            "status": "not_found",
            "tooltip": "Cannot be detected in this view configuration."
        }

    if not makes_sense:
        tooltip = f"**Why not meaningful:**\n\n{info.get('interpretation', 'Not semantically useful for analysis.')}"
        return {
            "status": "not_meaningful",
            "tooltip": tooltip
        }

    # Build detailed tooltip for meaningful patterns
    tooltip_parts = []

    if "use_case" in info:
        tooltip_parts.append(f"**Use Case:**\n{info['use_case']}")

    if "visual" in info:
        tooltip_parts.append(f"\n**Visual:**\n{info['visual']}")

    tooltip = "\n".join(
        tooltip_parts) if tooltip_parts else "Pattern is meaningful for this view."

    return {
        "status": "found",
        "tooltip": tooltip
    }


def render_matrix_cell(status: str, tooltip: str, cell_id: str):
    """Render a single matrix cell with appropriate styling and tooltip."""

    # Define symbols and colors based on status
    if status == "found":
        symbol = "✓"
        bg_color = "#22c55e"  # green
        text_color = "white"
    elif status == "not_meaningful":
        symbol = "△"
        bg_color = "#fbbf24"  # yellow/amber
        text_color = "#1f2937"  # dark gray
    else:  # not_found
        symbol = "✗"
        bg_color = "#ef4444"  # red
        text_color = "white"

    return f"""
    <div class="matrix-cell" style="background-color: {bg_color}; color: {text_color};" 
         data-tooltip="{cell_id}" title="">
        <span class="matrix-symbol">{symbol}</span>
        <div class="matrix-tooltip" id="tooltip-{cell_id}">
            {tooltip.replace('"', '&quot;')}
        </div>
    </div>
    """


def render_pattern_matrix():
    """Render the complete interactive pattern matrix."""

    st.markdown("""
    <div class="matrix-intro">
        <h2>Extended Pattern Matrix</h2>
        <p>This matrix shows which pattern detection methods are applicable and meaningful for each view configuration.
        Hover over cells to see detailed explanations.</p>
        <div class="matrix-legend">
            <div class="legend-item">
                <span class="legend-symbol" style="background: #22c55e; color: white;">✓</span>
                <span>Can be found and makes sense</span>
            </div>
            <div class="legend-item">
                <span class="legend-symbol" style="background: #fbbf24; color: #1f2937;">△</span>
                <span>Can be found but doesn't make sense</span>
            </div>
            <div class="legend-item">
                <span class="legend-symbol" style="background: #ef4444; color: white;">✗</span>
                <span>Cannot be found</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get all view configurations
    view_configs = get_all_view_configs()

    # Build the matrix HTML
    matrix_html = ['<div class="pattern-matrix-container">']
    matrix_html.append('<table class="pattern-matrix">')

    # Header row
    matrix_html.append('<thead><tr>')
    matrix_html.append(
        '<th class="matrix-header-col">View Configuration (X-axis × Y-axis × Color)</th>')
    for pattern in PATTERNS:
        matrix_html.append(
            f'<th class="matrix-header-pattern">{PATTERN_DISPLAY_NAMES[pattern]}</th>')
    matrix_html.append('</tr></thead>')

    # Body rows
    matrix_html.append('<tbody>')

    for idx, view_config in enumerate(view_configs):
        x, y, color = view_config
        view_label = get_view_label(x, y, color)

        # Check if this is a "meaningful" or "non-meaningful" row
        # We'll visually separate logical_time and relative_ratio views
        row_class = ""
        if x == "logical_time" or x == "relative_ratio":
            row_class = "matrix-row-non-meaningful"

        matrix_html.append(f'<tr class="{row_class}">')
        matrix_html.append(f'<td class="matrix-view-label">{view_label}</td>')

        # Pattern cells
        for pattern in PATTERNS:
            status_info = get_pattern_status(view_config, pattern)
            cell_id = f"{idx}-{pattern}"

            # Escape tooltip for HTML attribute
            tooltip_html = status_info['tooltip'].replace(
                '\n', '<br>').replace('"', '&quot;')

            matrix_html.append(f'<td class="matrix-cell-container">')
            matrix_html.append(render_matrix_cell(
                status_info['status'], tooltip_html, cell_id))
            matrix_html.append('</td>')

        matrix_html.append('</tr>')

    matrix_html.append('</tbody>')
    matrix_html.append('</table>')
    matrix_html.append('</div>')

    st.markdown(''.join(matrix_html), unsafe_allow_html=True)

    # Add JavaScript for enhanced tooltip behavior
    st.markdown("""
    <script>
    // Enhanced tooltip functionality
    document.addEventListener('DOMContentLoaded', function() {
        const cells = document.querySelectorAll('.matrix-cell');
        
        cells.forEach(cell => {
            const tooltip = cell.querySelector('.matrix-tooltip');
            if (tooltip) {
                cell.addEventListener('mouseenter', function(e) {
                    tooltip.style.display = 'block';
                    
                    // Position tooltip using fixed positioning
                    const rect = cell.getBoundingClientRect();
                    const tooltipRect = tooltip.getBoundingClientRect();
                    
                    // Position below the cell by default
                    let topPos = rect.bottom + 8;
                    
                    // If tooltip would go below viewport, show above instead
                    if (topPos + tooltipRect.height > window.innerHeight - 10) {
                        topPos = rect.top - tooltipRect.height - 8;
                    }
                    
                    tooltip.style.top = topPos + 'px';
                    
                    // Center horizontally, adjust if goes off-screen
                    let leftPos = rect.left + (rect.width / 2);
                    
                    if (leftPos + (tooltipRect.width / 2) > window.innerWidth - 10) {
                        tooltip.style.left = 'auto';
                        tooltip.style.right = '10px';
                        tooltip.style.transform = 'translateY(0)';
                    } else if (leftPos - (tooltipRect.width / 2) < 10) {
                        tooltip.style.left = '10px';
                        tooltip.style.transform = 'translateY(0)';
                    } else {
                        tooltip.style.left = leftPos + 'px';
                        tooltip.style.transform = 'translateX(-50%) translateY(0)';
                    }
                });
                
                cell.addEventListener('mouseleave', function() {
                    tooltip.style.display = 'none';
                });
            }
        });
    });
    </script>
    """, unsafe_allow_html=True)


def display_matrix_viewer():
    """Main entry point for displaying the pattern matrix viewer."""
    render_pattern_matrix()


if __name__ == "__main__":
    # For testing
    display_matrix_viewer()
