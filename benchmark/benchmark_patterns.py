"""
Benchmark Script for Pattern Detection Algorithms.

Runs all detection algorithms across XES files in data/ directory,
using all sampling methods. Generates CSV results and a markdown report.

Usage:
    python benchmark_patterns.py              # Full benchmark
    python benchmark_patterns.py --test-run   # Quick test (1 file, MINIMAL only)
    python benchmark_patterns.py --verify     # Verify all modes on smallest file
"""

import time
import argparse
import os
import sys
import traceback
from datetime import datetime
from glob import glob
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from core.detection.temporal_cluster import TemporalClusterPattern
from core.detection.cluster_pattern import ClusterPattern
from core.detection.outlier_detection import OutlierDetectionPattern
from core.detection.gap_pattern import GapPattern
from core.detection.case_arrival_trend_pattern import CaseArrivalTrendPattern
from core.detection.sequence_detector import HorizontalSequencePatternDetector
from core.data_processing.loader import load_xes_log
from core.utils.demo_sampling import sample_eventlog_variant_aware, SamplingMode
from core.app_utils.mappings import VIEW_PRESETS, X_AXIS_COLUMN_MAP, Y_AXIS_COLUMN_MAP, DOTS_COLOR_MAP


def log(msg: str) -> None:
    """Log a message with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def get_xes_files(data_dir: str) -> list[str]:
    """Get all XES files sorted by size (smallest first)."""
    files = [f for f in glob(os.path.join(data_dir, "*.xes")) if os.path.isfile(f)]
    files.sort(key=os.path.getsize)
    return files


def get_resource_column(df: pd.DataFrame) -> str | None:
    """Find the resource column in the dataframe."""
    for col in ["org:resource", "resource", "user", "performer"]:
        if col in df.columns:
            return col
    return None


def run_single_detector(
    name: str,
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    resource_col: str | None,
) -> dict[str, Any]:
    """
    Run a single detector and return timing and pattern count.

    Args:
        name: Detector name.
        df: DataFrame to run detection on.
        x: X-axis column.
        y: Y-axis column.
        color: Color column.
        resource_col: Resource column name if available.

    Returns:
        Dictionary with detection results.
    """
    result = {
        "Algorithm": name,
        "Parameters": "",
        "Detection Time (s)": np.nan,
        "Detected": "Error",
        "Patterns Found": np.nan,
    }

    try:
        t0 = time.time()

        if name == "Temporal Cluster":
            detector = TemporalClusterPattern(df=df, x_axis=x, y_axis=y)
            detector.detect()
            count = detector.get_summary().get("count", 0)
            params = f"min_cluster_size={detector.min_cluster_size}, temporal_eps={detector.temporal_eps:.2f}"
            detected = detector.detected

        elif name == "Cluster (OPTICS)":
            detector = ClusterPattern(view_config={"x": x, "y": y}, algorithm="optics")
            detector.detect(df)
            count = detector.get_summary().get("count", 0)
            used = detector.detected["params"] if detector.detected else detector.algorithm_params
            params = f"algorithm=optics, min_samples={used.get('min_samples')}, max_eps={used.get('max_eps'):.2f}"
            detected = detector.detected is not None

        elif name == "Outlier Detection":
            detector = OutlierDetectionPattern(df=df, view_config={"x": x, "y": y, "color": color})
            detector.detect()
            count = len(detector.outliers.get("combined", []))
            params = "isolation_forest + statistical"
            detected = detector.detected

        elif name == "Gap (Transition)":
            is_cat = y == resource_col and df[y].nunique() <= 1000 if resource_col else False
            detector = GapPattern(view_config={"x": x, "y": y}, gap_mode="transition", y_is_categorical=is_cat)
            detector.detect(df)
            count = detector.get_summary().get("count", 0)
            params = "mode=transition, min_samples=5"
            detected = detector.detected is not None

        elif name == "Sequence Detection":
            detector = HorizontalSequencePatternDetector(x_axis=x, y_axis=y, dot_color=color, df=df, min_support=30)
            detector.detect()
            count = detector.get_summary().get("count", 0)
            params = "min_support=30, prefixspan"
            detected = detector.detected

        elif name == "Case Arrival Trend":
            detector = CaseArrivalTrendPattern(view_config={"x": x})
            detector.detect(df)
            detected = detector.detected is not None and detector.detected.get("direction") != "no_trend"
            count = detector.get_summary().get("count", 0)
            params = "aggregation=W, mann_kendall"

        else:
            return result

        result["Parameters"] = params
        result["Detection Time (s)"] = time.time() - t0
        result["Detected"] = detected
        result["Patterns Found"] = count
        log(f"    {name}: {format_time(result['Detection Time (s)'])}, {count} patterns")

    except Exception as e:
        log(f"    {name}: ERROR - {str(e)[:50]}")
        result["Parameters"] = f"error: {str(e)[:30]}"

    return result


def run_all_detectors(
    df: pd.DataFrame,
    config: dict[str, Any],
    resource_col: str | None,
) -> list[dict[str, Any]]:
    """
    Run all detectors on a dataframe with given configuration.

    Args:
        df: DataFrame to analyze.
        config: View configuration with x_axis, y_axis, color.
        resource_col: Resource column name if available.

    Returns:
        List of result dictionaries.
    """
    x = X_AXIS_COLUMN_MAP.get(config["x_axis"], config["x_axis"])
    y = Y_AXIS_COLUMN_MAP.get(config["y_axis"], config["y_axis"])
    color = DOTS_COLOR_MAP.get(config["color"], config["color"])

    detectors = [
        "Temporal Cluster",
        "Cluster (OPTICS)",
        "Outlier Detection",
        "Gap (Transition)",
        "Sequence Detection",
    ]
    if x == "actual_time":
        detectors.append("Case Arrival Trend")

    results = []
    for name in detectors:
        result = run_single_detector(name, df, x, y, color, resource_col)
        result["Config"] = config.get("name", "Custom")
        result["X Axis"] = config["x_axis"]
        result["Y Axis"] = config["y_axis"]
        result["Color"] = config["color"]
        results.append(result)

    return results


def compute_baseline_metrics(
    result: dict[str, Any],
    mode: SamplingMode,
    sampling_time: float,
    baseline_stats: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """
    Compute and add baseline comparison metrics to result.

    Args:
        result: Result dictionary to update.
        mode: Current sampling mode.
        sampling_time: Time spent on sampling.
        baseline_stats: Dictionary of baseline statistics keyed by (Config, Algorithm).
    """
    key = (result["Config"], result["Algorithm"])

    if mode == SamplingMode.FULL:
        baseline_stats[key] = {
            "patterns": result["Patterns Found"],
            "total_time": (result["Detection Time (s)"] or 0) + sampling_time,
        }
        result["Patterns Lost"] = 0
        result["Patterns Lost %"] = "0.0%"
        result["Time Saved"] = 0
        result["Time Saved %"] = "0.0%"
        result["Retention Rate %"] = "100.0%"
    else:
        baseline = baseline_stats.get(key)
        if baseline and pd.notnull(result["Patterns Found"]) and pd.notnull(baseline["patterns"]):
            p_lost = baseline["patterns"] - result["Patterns Found"]
            p_lost_pct = (p_lost / baseline["patterns"] * 100) if baseline["patterns"] > 0 else 0
            result["Patterns Lost"] = p_lost
            result["Patterns Lost %"] = f"{p_lost_pct:.1f}%"
            result["Retention Rate %"] = f"{100 - p_lost_pct:.1f}%"

            total_time = (result["Detection Time (s)"] or 0) + sampling_time
            time_saved = baseline["total_time"] - total_time
            time_saved_pct = (time_saved / baseline["total_time"] * 100) if baseline["total_time"] > 0 else 0
            result["Time Saved"] = time_saved
            result["Time Saved %"] = f"{time_saved_pct:.1f}%"
        else:
            result["Patterns Lost"] = np.nan
            result["Patterns Lost %"] = "N/A"
            result["Time Saved"] = np.nan
            result["Time Saved %"] = "N/A"
            result["Retention Rate %"] = "N/A"


def save_results(results: list[dict[str, Any]], csv_path: str) -> None:
    """Save results to CSV."""
    if not results:
        return

    df = pd.DataFrame(results)
    col_order = [
        "File", "Sampling", "Config", "X Axis", "Y Axis", "Color",
        "Algorithm", "Parameters",
        "Detection Time (s)", "Sampling Time (s)", "Patterns Found", "Detected",
        "Events Lost", "Events Lost %", "Traces Lost", "Traces Lost %",
        "Patterns Lost", "Patterns Lost %", "Time Saved", "Time Saved %", "Retention Rate %",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(csv_path, index=False)


def generate_report(
    results_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate a markdown benchmark report."""
    lines = [
        "# Pattern Detection Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    valid_df = results_df[results_df["Detected"] != "Error"].copy()
    if valid_df.empty:
        lines.extend(["> [!WARNING]", "> No valid results were generated."])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # File Loading Times
    if not timing_df.empty:
        lines.extend(["## 📁 Data Management", "", "| File | Size (MB) | Events | Load Time |", "|:---|---:|---:|---:|"])
        for _, row in timing_df.iterrows():
            lines.append(f"| {row['File']} | {row['Size (MB)']:.1f} | {row['Events']:,} | {format_time(row['Load Time (s)'])} |")
        lines.append("")

    # Sampling Summary
    def pct_to_float(s: Any) -> float:
        if pd.isna(s) or s == "N/A" or not isinstance(s, str):
            return 0.0
        return float(s.strip("%"))

    summary_df = valid_df.copy()
    summary_df["Events Retained %"] = 100 - summary_df["Events Lost %"].apply(pct_to_float)
    summary_df["Pattern Retention %"] = summary_df["Retention Rate %"].apply(pct_to_float)
    summary_df["Time Saved Num"] = summary_df["Time Saved %"].apply(pct_to_float)

    samp_summary = summary_df.groupby("Sampling").agg({
        "Events Retained %": "mean",
        "Pattern Retention %": "mean",
        "Time Saved Num": "mean",
        "Sampling Time (s)": "mean",
    }).round(1)

    lines.extend([
        "## 📊 Sampling Performance Summary",
        "",
        "| Sampling | Events Retained | Pattern Retention | Time Saved | Sampling Time |",
        "|:---|---:|---:|---:|---:|",
    ])
    order = {"full": 0, "optimized": 1, "sqrt": 2, "minimal": 3}
    for mode in sorted(samp_summary.index, key=lambda x: order.get(x, 99)):
        row = samp_summary.loc[mode]
        lines.append(
            f"| **{mode}** | {row['Events Retained %']}% | {row['Pattern Retention %']}% | "
            f"{row['Time Saved Num']}% | {row['Sampling Time (s)']:.3f}s |"
        )
    lines.append("")

    # Detailed Breakdown
    lines.extend(["## 🔍 Detection Statistics by Configuration", ""])
    for config_name, config_group in valid_df.groupby("Config"):
        # Get axis info from first row of config group
        first_row = config_group.iloc[0]
        x_axis = first_row.get("X Axis", "N/A")
        y_axis = first_row.get("Y Axis", "N/A")
        color_axis = first_row.get("Color", "N/A")
        lines.extend([
            f"### {config_name}",
            f"**Axes:** X = `{x_axis}`, Y = `{y_axis}`, Color = `{color_axis}`",
            "",
        ])
        for algo_name, algo_group in config_group.groupby("Algorithm"):
            # Get parameters from first row (they're the same across sampling modes)
            params = algo_group.iloc[0].get("Parameters", "")
            lines.extend([
                f"#### {algo_name}",
                f"**Parameters:** `{params}`",
                "",
                "| Sampling | Patterns Found | Retention Rate | Detection Time | Time Saved % |",
                "|:---|---:|---:|---:|---:|",
            ])
            algo_group = algo_group.copy()
            algo_group["sort_order"] = algo_group["Sampling"].apply(lambda x: order.get(x, 99))
            for _, row in algo_group.sort_values("sort_order").iterrows():
                d_time = format_time(row["Detection Time (s)"]) if pd.notnull(row["Detection Time (s)"]) else "N/A"
                lines.append(
                    f"| {row['Sampling']} | {row['Patterns Found']:.0f} | {row['Retention Rate %']} | {d_time} | {row['Time Saved %']} |"
                )
            lines.append("")

    # Errors
    error_df = results_df[results_df["Detected"] == "Error"]
    if not error_df.empty:
        lines.extend(["## ❌ Errors", "", f"**{len(error_df)} runs failed.**", ""])
        for _, row in error_df.iterrows():
            lines.append(f"- `{row['File']}` / `{row['Sampling']}` / `{row['Config']}` / `{row['Algorithm']}`: {row['Parameters']}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Report saved to: {output_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark."""
    data_dir = os.path.join(project_root, "data")
    xes_files = get_xes_files(data_dir)

    if not xes_files:
        log(f"No XES files found in {data_dir}")
        return

    # Handle file selection
    if args.file:
        # Use specific file
        file_path = args.file if os.path.isabs(args.file) else os.path.join(data_dir, args.file)
        if not os.path.exists(file_path):
            log(f"File not found: {file_path}")
            return
        xes_files = [file_path]
        sampling_modes = [SamplingMode.MINIMAL, SamplingMode.SQRT, SamplingMode.OPTIMIZED, SamplingMode.FULL]
        log(f"*** SINGLE FILE: {os.path.basename(file_path)} ***")
    elif args.test_run:
        xes_files = xes_files[:1]
        sampling_modes = [SamplingMode.MINIMAL]
        log("*** TEST RUN: 1 file, MINIMAL only ***")
    elif args.verify:
        xes_files = xes_files[:1]
        sampling_modes = [SamplingMode.MINIMAL, SamplingMode.SQRT, SamplingMode.OPTIMIZED, SamplingMode.FULL]
        log("*** VERIFY: All modes on smallest file ***")
    else:
        sampling_modes = [SamplingMode.MINIMAL, SamplingMode.SQRT, SamplingMode.OPTIMIZED, SamplingMode.FULL]

    log(f"Files: {len(xes_files)}, Modes: {[m.value for m in sampling_modes]}")

    csv_path = os.path.join(project_root, "benchmark_results.csv")
    report_path = os.path.join(project_root, "docs", "benchmark_report.md")

    all_results: list[dict[str, Any]] = []
    file_timing: list[dict[str, Any]] = []
    total_start = time.time()

    for file_idx, xes_path in enumerate(xes_files, 1):
        file_name = os.path.basename(xes_path)
        file_size_mb = os.path.getsize(xes_path) / (1024 * 1024)

        log("=" * 60)
        log(f"FILE {file_idx}/{len(xes_files)}: {file_name} ({file_size_mb:.1f} MB)")

        # Load file once
        load_start = time.time()
        try:
            df_full = load_xes_log(xes_path)
            load_time = time.time() - load_start
            log(f"Loaded in {format_time(load_time)} ({len(df_full):,} events)")
            file_timing.append({"File": file_name, "Size (MB)": file_size_mb, "Load Time (s)": load_time, "Events": len(df_full)})
        except Exception as e:
            log(f"ERROR loading: {e}")
            traceback.print_exc()
            continue

        resource_col = get_resource_column(df_full)
        baseline_stats: dict[tuple[str, str], dict[str, Any]] = {}

        # Run FULL first as baseline (unless test-run)
        modes_to_run = sampling_modes if args.test_run else [SamplingMode.FULL] + [m for m in sampling_modes if m != SamplingMode.FULL]

        for mode in modes_to_run:
            log(f"\n--- Sampling: {mode.value} ---")

            sampling_start = time.time()
            try:
                if mode == SamplingMode.FULL:
                    df = df_full.copy()
                    stats = {
                        "original_events": len(df_full),
                        "sampled_events": len(df_full),
                        "original_traces": df_full["case_id"].nunique(),
                        "sampled_traces": df_full["case_id"].nunique(),
                    }
                    log(f"Using full dataset: {len(df):,} events")
                else:
                    df, stats = sample_eventlog_variant_aware(
                        df_full, mode=mode, case_col="case_id", activity_col="activity", time_col="actual_time"
                    )
                    log(f"Sampled: {stats['sampled_events']:,} events ({stats['reduction_ratio']*100:.1f}%)")
                sampling_time = time.time() - sampling_start
            except Exception as e:
                log(f"ERROR sampling: {e}")
                traceback.print_exc()
                continue

            events_lost = stats["original_events"] - stats["sampled_events"]
            traces_lost = stats["original_traces"] - stats["sampled_traces"]
            events_lost_pct = (events_lost / stats["original_events"] * 100) if stats["original_events"] > 0 else 0
            traces_lost_pct = (traces_lost / stats["original_traces"] * 100) if stats["original_traces"] > 0 else 0

            # Run detectors for each view preset
            for name, preset in VIEW_PRESETS.items():
                config = {**preset, "name": name}
                log(f"  Config: {name}")

                for result in run_all_detectors(df, config, resource_col):
                    result["File"] = file_name
                    result["Sampling"] = mode.value
                    result["Sampling Time (s)"] = sampling_time
                    result["Events Lost"] = events_lost
                    result["Events Lost %"] = f"{events_lost_pct:.1f}%"
                    result["Traces Lost"] = traces_lost
                    result["Traces Lost %"] = f"{traces_lost_pct:.1f}%"
                    compute_baseline_metrics(result, mode, sampling_time, baseline_stats)
                    all_results.append(result)

            save_results(all_results, csv_path)

    # Final output
    total_time = time.time() - total_start
    if all_results:
        results_df = pd.DataFrame(all_results)
        timing_df = pd.DataFrame(file_timing)
        generate_report(results_df, timing_df, report_path)

        log("")
        log("=" * 60)
        log("BENCHMARK COMPLETE")
        log(f"Total runs: {len(results_df)}, Errors: {len(results_df[results_df['Detected'] == 'Error'])}")
        log(f"Total time: {format_time(total_time)}")
        log(f"Results: {csv_path}")
        log(f"Report: {report_path}")
    else:
        log("No results generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Pattern Detection Algorithms")
    parser.add_argument("--file", "-f", type=str, help="Specific XES file to benchmark (filename or path)")
    parser.add_argument("--test-run", action="store_true", help="Quick test (1 file, MINIMAL only)")
    parser.add_argument("--verify", action="store_true", help="Verify all modes on smallest file")
    args = parser.parse_args()

    try:
        run_benchmark(args)
    except KeyboardInterrupt:
        log("\nBenchmark interrupted.")
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
