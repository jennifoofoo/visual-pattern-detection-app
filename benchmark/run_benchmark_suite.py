"""
Benchmark Suite Orchestrator.

This script runs the pattern detection benchmark suite. 
It allows running specific algorithms in isolation and consolidating their results.

Usage:
    python run_benchmark_suite.py 
    python run_benchmark_suite.py --algorithms "Sequence Detection" "Gap (Transition)"
    python run_benchmark_suite.py --test-run
"""

import sys
import os
import argparse
import subprocess
import glob
import pandas as pd
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

# Import report generation from v3
from benchmark.benchmark_patterns_v3 import generate_report, ALL_ALGORITHMS


# Define the desired execution order
ORDERED_ALGORITHMS = [
    "Case Arrival Trend",
    "Outlier Detection",
    "Temporal Cluster",
    "Sequence Detection",
    "Gap (Transition)",
    "Cluster (OPTICS)",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def slugify(text: str) -> str:
    """Make a string safe for filenames."""
    return text.lower().replace(" ", "_").replace("(", "").replace(")", "")


def run_suite(args: argparse.Namespace) -> None:
    # 1. Setup Directories
    results_dir_ind = os.path.join(project_root, "benchmark", "results", "individual")
    results_dir_cons = os.path.join(project_root, "benchmark", "results", "consolidated")
    
    os.makedirs(results_dir_ind, exist_ok=True)
    os.makedirs(results_dir_cons, exist_ok=True)
    
    # 2. Determine Algorithms to Run
    if args.algorithms:
        # Filter ordered list by user selection to maintain order
        # Normalize for case-insensitive matching slightly if needed, but strict is safer
        to_run = [algo for algo in ORDERED_ALGORITHMS if algo in args.algorithms]
        if not to_run:
            log(f"Warning: No valid algorithms matched {args.algorithms}. Available: {ORDERED_ALGORITHMS}")
    else:
        to_run = ORDERED_ALGORITHMS

    log(f"Targeting Algorithms: {to_run}")

    # 3. Execution Loop
    if not args.report_only:
        v3_script = os.path.join(project_root, "benchmark", "benchmark_patterns_v3.py")
        
        for algo in to_run:
            slug = slugify(algo)
            log(f"\n>>> STARTING: {algo}")
            
            csv_path = os.path.join(results_dir_ind, f"benchmark_{slug}.csv")
            report_path = os.path.join(results_dir_ind, f"report_{slug}.md")
            
            # If we are re-running, specific requirement implies "own csv log file".
            # We should overwrite the individual file for this run to keep it clean for *current* state of this algo.
            # However, appending is safer if running multiple files. 
            # Given "each algorithm should get an own csv log file", rewriting seems appropriate if starting a fresh run for that algo.
            if os.path.exists(csv_path):
                 # For this specific suite run, we might want to clear previous results for THIS algorithm 
                 # to avoid duplicating entries if run multiple times.
                 # Let's simple remove it before running.
                 try:
                     os.remove(csv_path)
                     log(f"    Cleaned previous results at {csv_path}")
                 except Exception as e:
                     log(f"    Warning: Could not remove {csv_path}: {e}")

            cmd = [
                sys.executable,
                v3_script,
                "--algorithms", algo,
                "--output-csv", csv_path,
                "--output-report", report_path
            ]
            
            if args.test_run:
                cmd.append("--test-run")
            if args.file:
                cmd.extend(["--file", args.file])

            try:
                subprocess.run(cmd, check=True)
                log(f">>> FINISHED: {algo}")
            except subprocess.CalledProcessError as e:
                log(f"!!! FAILED: {algo} (Exit Code {e.returncode})")
            except Exception as e:
                log(f"!!! ERROR: {algo} - {e}")
    else:
        log("\n>>> REPORT ONLY MODE: Skipping execution loop.")

    # 4. Consolidation
    log("\n=== CONSOLIDATING RESULTS ===")
    
    # Find all CSVs in individual folder (including those not run this time)
    all_csvs = glob.glob(os.path.join(results_dir_ind, "benchmark_*.csv"))
    
    if not all_csvs:
        log("No result CSVs found to consolidate.")
        return

    combined_df = pd.DataFrame()
    for f in all_csvs:
        try:
            df = pd.read_csv(f)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            log(f"  Merged: {os.path.basename(f)} ({len(df)} rows)")
        except Exception as e:
            log(f"  Error reading {f}: {e}")

    if combined_df.empty:
        log("Consolidated DataFrame is empty.")
        return

    # Save Master CSV
    master_csv_path = os.path.join(results_dir_cons, "benchmark_consolidated.csv")
    combined_df.to_csv(master_csv_path, index=False)
    log(f"Saved Consolidated CSV: {master_csv_path}")

    # Generate Master Report
    master_report_path = os.path.join(results_dir_cons, "benchmark_report_consolidated.md")
    
    # We need a dummy timing df or aggregated one. 
    # Since timing info is per-run and potentially lost or scattered, we'll create a simple one or skip detailed file loading table.
    # The generation function handles empty timing df gracefully.
    generate_report(combined_df, pd.DataFrame(), master_report_path)
    log(f"Saved Consolidated Report: {master_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Benchmark Suite (Sequential Isolated Runs)")
    parser.add_argument("--algorithms", nargs="+", choices=ORDERED_ALGORITHMS, help="Specific algorithms to run")
    parser.add_argument("--test-run", action="store_true", help="Quick test run")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing CSVs")
    parser.add_argument("--file", type=str, help="Specific file to benchmark")
    
    args = parser.parse_args()
    
    run_suite(args)
