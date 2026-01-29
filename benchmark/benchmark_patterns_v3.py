"""
Benchmark Script for Pattern Detection Algorithms (v3).

Modifications from v2:
- Added support for custom output paths (--output-csv, --output-report).
- Designed to be invoked by run_benchmark_suite.py.
- Added 10-minute timeout per worker to prevent hangs.

Usage:
    python benchmark_patterns_v3.py --output-csv "path/to/res.csv" --output-report "path/to/rep.md"

Timeout:
    Each worker process has a 10-minute timeout (configurable via WORKER_TIMEOUT constant).
    Timeouts are logged and recorded in the CSV with Detected='Timeout'.
"""

import time
import argparse
import os
import sys
import traceback
import subprocess
import pickle
import uuid
import gc
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


# Timeout for each worker process (in seconds)
WORKER_TIMEOUT = 600  # 10 minutes

ALL_ALGORITHMS = [
    "Temporal Cluster",
    "Cluster (OPTICS)",
    "Outlier Detection",
    "Gap (Transition)",
    "Sequence Detection",
    "Case Arrival Trend",
]


def log(msg: str) -> None:
    """Log a message with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    try:
        # Check for NaN or invalid values
        if pd.isna(seconds) or seconds is None:
            return "N/A"
        
        # Convert to float to handle any edge cases
        seconds = float(seconds)
        
        if seconds < 60:
            return f"{seconds:.2f}s"
        if seconds < 3600:
            return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    except (ValueError, TypeError):
        return "N/A"


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


# =============================================================================
# DETECTOR LOGIC (Shared)
# =============================================================================

def run_single_detector(
    name: str,
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    resource_col: str | None,
) -> dict[str, Any]:
    """Run a single detector and return timing and pattern count."""
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
    detectors_to_run: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all detectors on a dataframe with given configuration."""
    x = X_AXIS_COLUMN_MAP.get(config["x_axis"], config["x_axis"])
    y = Y_AXIS_COLUMN_MAP.get(config["y_axis"], config["y_axis"])
    color = DOTS_COLOR_MAP.get(config["color"], config["color"])

    potential_detectors = [
        "Temporal Cluster",
        "Cluster (OPTICS)",
        "Outlier Detection",
        "Gap (Transition)",
        "Sequence Detection",
    ]
    if x == "actual_time":
        potential_detectors.append("Case Arrival Trend")

    detectors = []
    for d in potential_detectors:
        if detectors_to_run is None or d in detectors_to_run:
            detectors.append(d)

    results = []
    for name in detectors:
        result = run_single_detector(name, df, x, y, color, resource_col)
        result["Config"] = config.get("name", "Custom")
        result["X Axis"] = config["x_axis"]
        result["Y Axis"] = config["y_axis"]
        result["Color"] = config["color"]
        results.append(result)

    return results

# =============================================================================
# WORKER PROCESS
# =============================================================================

def run_worker(args: argparse.Namespace) -> None:
    """
    Worker process: Loads pickle, samples, runs ONE config, appends to CSV.
    """
    try:
        # 1. Load Data
        with open(args.pickle_path, "rb") as f:
            df_full = pickle.load(f)
        
        file_name = os.path.basename(args.file) # Args file passed from orchestrator
        resource_col = get_resource_column(df_full)
        mode = SamplingMode(args.sampling_mode)
        
        # 2. Sample
        sampling_start = time.time()
        if mode == SamplingMode.FULL:
            df = df_full  # No copy needed since process will die
            stats = {
                "original_events": len(df_full),
                "sampled_events": len(df_full),
                "original_traces": df_full["case_id"].nunique(),
                "sampled_traces": df_full["case_id"].nunique(),
                "reduction_ratio": 1.0
            }
        else:
            df, stats = sample_eventlog_variant_aware(
                df_full, mode=mode, case_col="case_id", activity_col="activity", time_col="actual_time"
            )
        sampling_time = time.time() - sampling_start
        
        # 3. Filter Detectors
        current_detectors = args.algorithms if args.algorithms else ALL_ALGORITHMS.copy()
        if mode == SamplingMode.FULL and args.exclude_on_full:
            current_detectors = [d for d in current_detectors if d not in args.exclude_on_full]

        # 4. Get Config
        preset = VIEW_PRESETS[args.config_name]
        config = {**preset, "name": args.config_name}

        # 5. Run Detectors
        results = run_all_detectors(df, config, resource_col, detectors_to_run=current_detectors)

        # 6. Post-process results
        events_lost = stats["original_events"] - stats["sampled_events"]
        traces_lost = stats["original_traces"] - stats["sampled_traces"]
        events_lost_pct = (events_lost / stats["original_events"] * 100) if stats["original_events"] > 0 else 0
        traces_lost_pct = (traces_lost / stats["original_traces"] * 100) if stats["original_traces"] > 0 else 0
        
        final_rows = []
        for r in results:
            r["File"] = file_name
            r["Sampling"] = mode.value
            r["Sampling Time (s)"] = sampling_time
            r["Events Lost"] = events_lost
            r["Events Lost %"] = f"{events_lost_pct:.1f}%"
            r["Traces Lost"] = traces_lost
            r["Traces Lost %"] = f"{traces_lost_pct:.1f}%"
            # Add absolute counts
            r["Original Events"] = stats["original_events"]
            r["Sampled Events"] = stats["sampled_events"]
            # Placeholders for baseline metrics
            r["Patterns Lost"] = np.nan
            r["Patterns Lost %"] = "N/A" 
            r["Time Saved"] = np.nan
            r["Time Saved %"] = "N/A"
            r["Retention Rate %"] = "N/A"
            final_rows.append(r)

        # 7. Append to CSV (Atomic-ish append)
        df_res = pd.DataFrame(final_rows)
        if not df_res.empty:
            # Use arg-provided path or default to standard v3 path
            csv_path = args.output_csv if args.output_csv else os.path.join(project_root, "benchmark_results_v3.csv")
            
            # Ensure dir exists if a custom path is provided
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

            header = not os.path.exists(csv_path)
            # Use specific columns to ensure order
            col_order = [
                "File", "Sampling", "Config", "X Axis", "Y Axis", "Color",
                "Algorithm", "Parameters",
                "Detection Time (s)", "Sampling Time (s)", "Patterns Found", "Detected",
                "Original Events", "Sampled Events", 
                "Events Lost", "Events Lost %", "Traces Lost", "Traces Lost %",
                "Patterns Lost", "Patterns Lost %", "Time Saved", "Time Saved %", "Retention Rate %",
            ]
            df_res = df_res.reindex(columns=col_order)
            df_res.to_csv(csv_path, mode='a', header=header, index=False)
            
            # Print brief summary for orchestrator to show
            for r in final_rows:
                 log(f"    Worker Finished: {r['Algorithm']} ({format_time(r['Detection Time (s)'])}s, {r['Patterns Found']} patterns)")

    except Exception as e:
        log(f"WORKER ERROR ({args.config_name}): {e}")
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# ORCHESTRATOR PROCESS
# =============================================================================

def run_orchestrator(args: argparse.Namespace) -> None:
    """
    Orchestrator: Manages XES-to-Pickle and spawns workers.
    """
    data_dir = os.path.join(current_dir, "eventdata")
    xes_files = get_xes_files(data_dir)

    if not xes_files:
        log(f"No XES files found in {data_dir}")
        return

    # Filter files
    if args.file:
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
    elif args.verify:
        xes_files = xes_files[:1]
        sampling_modes = [SamplingMode.MINIMAL, SamplingMode.SQRT, SamplingMode.OPTIMIZED, SamplingMode.FULL]
    else:
        sampling_modes = [SamplingMode.MINIMAL, SamplingMode.SQRT, SamplingMode.OPTIMIZED, SamplingMode.FULL]

    log(f"Files: {len(xes_files)}, Modes: {[m.value for m in sampling_modes]}")
    if args.algorithms:
        log(f"Algorithms allowed globally: {args.algorithms}")
    if args.exclude_on_full:
        log(f"Algorithms excluded on FULL: {args.exclude_on_full}")
    
    # Use specified CSV path or default
    csv_path = args.output_csv if args.output_csv else os.path.join(project_root, "benchmark_results_v3.csv")
    
    # If not in suite mode (i.e. if running standalone), we might want to clear previous results.
    # However, since this script is now designed to be called multiple times by the suite runner
    # directly or via subprocess, we should be careful about deleting.
    # FOR STANDALONE RUN: We might want to clear.
    # FOR SUITE RUN: The suite runner handles file management.
    # We will ONLY clear if explicitly asked or if it's the default file and we are not appending.
    # For safety in v3, we will NOT auto-delete unless it's a fresh orchestrated run started by user manually without specific flags.
    # Currently, we'll assume append mode is safer for the suite runner.
    # But for a clean single run, let's just log where it goes.
    log(f"Results will be appended to: {csv_path}")

    total_start = time.time()
    file_timing = []

    for file_idx, xes_path in enumerate(xes_files, 1):
        file_name = os.path.basename(xes_path)
        file_size_mb = os.path.getsize(xes_path) / (1024 * 1024)

        log("=" * 60)
        log(f"FILE {file_idx}/{len(xes_files)}: {file_name} ({file_size_mb:.1f} MB)")

        # 1. Load XES and Convert to Pickle (Cached for workers)
        pkl_name = f"temp_log_{uuid.uuid4().hex[:8]}.pkl" # Unique pickle to avoid collisions if multiple run
        pkl_path = os.path.join(project_root, pkl_name)
        
        load_start = time.time()
        try:
            log("Loading XES and creating binary cache...")
            df_full = load_xes_log(xes_path)
            with open(pkl_path, "wb") as f:
                pickle.dump(df_full, f)
            
            load_time = time.time() - load_start
            log(f"Cached in {format_time(load_time)} ({len(df_full):,} events)")
            file_timing.append({"File": file_name, "Size (MB)": file_size_mb, "Load Time (s)": load_time, "Events": len(df_full)})
        except Exception as e:
            log(f"ERROR loading: {e}")
            traceback.print_exc()
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
            continue

        modes_to_run = sampling_modes if args.test_run else [SamplingMode.FULL] + [m for m in sampling_modes if m != SamplingMode.FULL]

        for mode in modes_to_run:
            log(f"\n--- Sampling: {mode.value} ---")
            
            # Subprocess Args Base
            cmd_base = [
                sys.executable,
                __file__,
                "--worker",
                "--file", xes_path,
                "--pickle_path", pkl_path,
                "--sampling_mode", mode.value
            ]
            
            # Pass filters
            if args.algorithms:
                cmd_base.append("--algorithms")
                cmd_base.extend(args.algorithms)
            if args.exclude_on_full:
                cmd_base.append("--exclude-on-full")
                cmd_base.extend(args.exclude_on_full)
            if args.output_csv:
                cmd_base.extend(["--output-csv", args.output_csv])

            # Iterate Configurations
            for config_name in VIEW_PRESETS.keys():
                log(f"  Running Config: {config_name} (Fresh Process)...")
                
                cmd = cmd_base + ["--config_name", config_name]
                
                try:
                    # Run Worker process with timeout (Direct output streaming)
                    proc = subprocess.run(cmd, check=True, timeout=WORKER_TIMEOUT)
                    
                except subprocess.TimeoutExpired:
                    log(f"    ⚠️ TIMEOUT: Worker exceeded {WORKER_TIMEOUT/60:.0f} minutes for {config_name}")
                    
                    # Record timeout in CSV
                    timeout_rows = []
                    preset = VIEW_PRESETS[config_name]
                    config = {**preset, "name": config_name}
                    
                    # Get algorithms that would have run
                    current_detectors = args.algorithms if args.algorithms else ALL_ALGORITHMS.copy()
                    if mode == SamplingMode.FULL and args.exclude_on_full:
                        current_detectors = [d for d in current_detectors if d not in args.exclude_on_full]
                    
                    # Create timeout entry for each algorithm
                    for algo in current_detectors:
                        timeout_row = {
                            "File": file_name,
                            "Sampling": mode.value,
                            "Config": config_name,
                            "X Axis": config["x_axis"],
                            "Y Axis": config["y_axis"],
                            "Color": config["color"],
                            "Algorithm": algo,
                            "Parameters": f"TIMEOUT after {WORKER_TIMEOUT}s",
                            "Detection Time (s)": np.nan,
                            "Sampling Time (s)": np.nan,
                            "Patterns Found": np.nan,
                            "Detected": "Timeout",
                            "Original Events": np.nan,
                            "Sampled Events": np.nan,
                            "Events Lost": np.nan,
                            "Events Lost %": "N/A",
                            "Traces Lost": np.nan,
                            "Traces Lost %": "N/A",
                            "Patterns Lost": np.nan,
                            "Patterns Lost %": "N/A",
                            "Time Saved": np.nan,
                            "Time Saved %": "N/A",
                            "Retention Rate %": "N/A",
                        }
                        timeout_rows.append(timeout_row)
                    
                    # Write timeout entries to CSV
                    if timeout_rows:
                        df_timeout = pd.DataFrame(timeout_rows)
                        header = not os.path.exists(csv_path)
                        col_order = [
                            "File", "Sampling", "Config", "X Axis", "Y Axis", "Color",
                            "Algorithm", "Parameters",
                            "Detection Time (s)", "Sampling Time (s)", "Patterns Found", "Detected",
                            "Original Events", "Sampled Events", 
                            "Events Lost", "Events Lost %", "Traces Lost", "Traces Lost %",
                            "Patterns Lost", "Patterns Lost %", "Time Saved", "Time Saved %", "Retention Rate %",
                        ]
                        df_timeout = df_timeout.reindex(columns=col_order)
                        df_timeout.to_csv(csv_path, mode='a', header=header, index=False)
                    
                except subprocess.CalledProcessError as e:
                     log(f"    ERROR in worker (Exit {e.returncode})")
                except Exception as e:
                    log(f"    Subprocess failure: {e}")

        # Cleanup Pickle
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            log("Cache cleared.")

    # Generate Report
    total_time = time.time() - total_start
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        
        report_path = args.output_report if args.output_report else os.path.join(project_root, "docs", "benchmark_report_v3.md")
        # Ensure report dir exists
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        
        generate_report(results_df, pd.DataFrame(file_timing), report_path)

        log("")
        log("=" * 60)
        log("BENCHMARK COMPLETE")
        log(f"Total time: {format_time(total_time)}")
        log(f"Results: {csv_path}")
        log(f"Report: {report_path}")
    else:
        log("No results generated.")


# =============================================================================
# SHARED REPORTING
# =============================================================================

def generate_report(results_df: pd.DataFrame, timing_df: pd.DataFrame, output_path: str) -> None:
    """Generate a markdown benchmark report."""
    lines = [
        "# Pattern Detection Benchmark Report (v3)",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # Include all non-error results (including timeouts)
    valid_df = results_df[results_df["Detected"] != "Error"].copy()
    timeout_df = results_df[results_df["Detected"] == "Timeout"].copy()
    
    if valid_df.empty:
        lines.extend(["> [!WARNING]", "> No valid results were generated."])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    
    # Add timeout summary if any
    if not timeout_df.empty:
        lines.extend([
            "> [!CAUTION]",
            f"> **{len(timeout_df)} timeout(s) detected** - Some workers exceeded the {WORKER_TIMEOUT/60:.0f}-minute limit.",
            ""
        ])

    if not timing_df.empty:
        lines.extend(["## 📁 Data Management", "", "| File | Size (MB) | Events | Load Time |", "|:---|---:|---:|---:|"])
        for _, row in timing_df.iterrows():
            lines.append(f"| {row['File']} | {row['Size (MB)']:.1f} | {row['Events']:,} | {format_time(row['Load Time (s)'])} |")
        lines.append("")

    # Identify Full baselines for calculation per (File, Config, Algorithm)
    # This helps in calculating Retention and Speedup for each sampling mode.
    baselines = valid_df[valid_df["Sampling"] == "full"].set_index(["File", "Config", "Algorithm"])

    # 1. 📈 Sampling Performance by Dataset
    # This section provides a high-level view of how sampling methods performed per dataset.
    lines.extend(["## 📈 Sampling Performance by Dataset", ""])
    for file_name, file_group in valid_df.groupby("File"):
        lines.extend([f"### {file_name}", 
                      "",
                      "| Sampling | Events (Abs) | Events Retained | Patterns (Avg) | Time (Avg) | Speedup |",
                      "|:---|---:|---:|---:|---:|---:|"])
        
        # Aggregate by sampling for this dataset
        file_samp = file_group.groupby("Sampling").agg({
            "Sampled Events": "max",
            "Original Events": "max",
            "Events Lost %": lambda x: x.iloc[0],
            "Patterns Found": "mean",
            "Detection Time (s)": "mean"
        })
        
        full_time = file_samp.loc["full", "Detection Time (s)"] if "full" in file_samp.index else None
        
        # Sort by sampling order preferred
        order = ["full", "optimized", "sqrt", "minimal"]
        existing_order = [m for m in order if m in file_samp.index] + [m for m in file_samp.index if m not in order]
        
        for mode in existing_order:
            row = file_samp.loc[mode]
            
            # Format Absolute Events
            orig = int(row.get("Original Events", 0))
            samp = int(row.get("Sampled Events", 0))
            if orig > 0:
                events_abs = f"{samp:,} / {orig:,}"
            else:
                events_abs = "N/A"

            # Convert "XX.X%" string to retained percentage
            try:
                lost_pct = float(str(row["Events Lost %"]).strip('%'))
                retained_pct = f"{100 - lost_pct:.1f}%"
            except (ValueError, TypeError):
                retained_pct = "N/A"
                
            speedup = f"{full_time / row['Detection Time (s)']:.1f}x" if full_time and row['Detection Time (s)'] > 0 else "1.0x"
            lines.append(f"| **{mode}** | {events_abs} | {retained_pct} | {row['Patterns Found']:.1f} | {row['Detection Time (s)']:.3f}s | {speedup} |")
        lines.append("")

    # 2. 🔍 Detection Statistics by Configuration
    # Detailed breakdown grouped by Configuration -> File -> Algorithm
    lines.extend(["## 🔍 Detection Statistics by Configuration", ""])
    for config_name, config_group in valid_df.groupby("Config"):
        first_row = config_group.iloc[0]
        lines.extend([f"### {config_name}", 
                      f"**Axes:** X = `{first_row.get('X Axis')}`, Y = `{first_row.get('Y Axis')}`, Color = `{first_row.get('Color')}`", 
                      ""])
        
        for file_name, file_group in config_group.groupby("File"):
            lines.extend([f"#### {file_name}", ""])
            
            for algo_name, algo_group in file_group.groupby("Algorithm"):
                params = algo_group.iloc[0].get("Parameters", "N/A")
                lines.extend([f"**{algo_name}**", 
                              f"*Parameters: `{params}`*",
                              "",
                              "| Sampling | Patterns | Retention | Time | Speedup |", 
                              "|:---|---:|---:|---:|---:|"])
                
                baseline_row = None
                try:
                    baseline_row = baselines.loc[(file_name, config_name, algo_name)]
                except KeyError:
                    pass

                for _, row in algo_group.iterrows():
                    d_time = format_time(row["Detection Time (s)"])
                    
                    retention = "100%"
                    speedup = "1.0x"
                    
                    if baseline_row is not None and row["Sampling"] != "full":
                        # Retention calculation
                        if baseline_row["Patterns Found"] > 0:
                            retention = f"{(row['Patterns Found'] / baseline_row['Patterns Found'] * 100):.1f}%"
                        elif row["Patterns Found"] == 0:
                            retention = "100%" 
                        else:
                            retention = "N/A"
                            
                        # Speedup calculation
                        if row["Detection Time (s)"] > 0:
                            speedup = f"{(baseline_row['Detection Time (s)'] / row['Detection Time (s)']):.1f}x"
                        else:
                            speedup = "N/A"
                            
                    lines.append(f"| {row['Sampling']} | {row['Patterns Found']:.0f} | {retention} | {d_time} | {speedup} |")
                lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Report saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Pattern Detection Algorithms (v3)")
    parser.add_argument("--file", "-f", type=str, help="Specific XES file to benchmark")
    parser.add_argument("--test-run", action="store_true", help="Quick test")
    parser.add_argument("--verify", action="store_true", help="Verify mode")
    parser.add_argument("--algorithms", nargs="+", choices=ALL_ALGORITHMS, help="Global algorithm filter")
    parser.add_argument("--exclude-on-full", nargs="+", choices=ALL_ALGORITHMS, help="Exclude on FULL mode")
    
    # New Arguments for V3
    parser.add_argument("--output-csv", type=str, help="Path for output CSV")
    parser.add_argument("--output-report", type=str, help="Path for output Report (MD)")

    # Internal Worker Args
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pickle_path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--sampling_mode", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--config_name", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    try:
        if args.worker:
            run_worker(args)
        else:
            run_orchestrator(args)
    except KeyboardInterrupt:
        pass # Clean exit
    except Exception as e:
        if not args.worker:
            log(f"\nFATAL ERROR: {e}")
            traceback.print_exc()
