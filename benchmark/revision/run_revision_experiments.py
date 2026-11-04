"""Quantitative evaluation and sensitivity analysis requested in the revision.

For every event log and sampling strategy (full, optimized, sqrt = Balanced, minimal):
  * performance.csv  - detection time, peak additional memory and number of
                       detected patterns per method (Activity Overview view:
                       X = actual time, Y = activity, color = resource).
  * sens_contamination.csv - Isolation Forest contamination alpha.
  * sens_trend.csv   - Mann-Kendall p-value / Sen's slope, classified offline
                       for several significance levels and effect-size thresholds.
  * sens_eps.csv     - temporal-burst DBSCAN radius as a fraction of the time span.
  * datasets.csv     - size of each log and of each sample.

An extra entry "Road (paper view)" reproduces the case-study view of the paper
(Balanced sampling, seed 42, events from 2009-01-01, color = case id).

Each detection runs in a forked child with a 600 s timeout and a 3 GB cap on
additional address space; memory is the peak RSS of the child above its RSS at
fork time, sampled every 10 ms.

Run from the repository root:
    venv/bin/python benchmark/revision/run_revision_experiments.py [--logs NAME ...]
"""
import argparse
import gc
import resource
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import core.detection.outlier_detection as od
from core.data_processing.loader import load_xes_log
from core.detection.case_arrival_trend_pattern import CaseArrivalTrendPattern
from core.detection.cluster_pattern import ClusterPattern
from core.detection.gap_pattern import GapPattern
from core.detection.sequence_detector import HorizontalSequencePatternDetector
from core.detection.temporal_cluster import TemporalClusterPattern
from core.utils.demo_sampling import SamplingMode, sample_eventlog_variant_aware

OUT = Path(__file__).resolve().parent / "results"
TIMEOUT = 600  # seconds per method
MEM_LIMIT_MB = 3072  # extra address space per detection run

LOGS = {
    "Sepsis": ROOT / "data" / "Sepsis Cases - Event Log.xes" / "Sepsis Cases - Event Log.xes",
    "BPI2020 Domestic": ROOT / "data" / "BPI_Challange_2020" / "DomesticDeclarations.xes",
    "BPI2020 International": ROOT / "data" / "BPI_Challange_2020" / "InternationalDeclarations.xes",
    "Hospital": ROOT / "data" / "Hospital_log.xes",
    "BPI2012": ROOT / "data" / "BPI_Challenge_2012" / "BPI_Challenge_2012.xes",
    "Road Traffic Fine": ROOT / "data" / "Road.xes",
}
PAPER_VIEW = "Road (paper view)"
SAMPLINGS = [SamplingMode.FULL, SamplingMode.OPTIMIZED, SamplingMode.SQRT, SamplingMode.MINIMAL]
SAMPLING_LABEL = {"full": "Full", "optimized": "Optimized", "sqrt": "Balanced (sqrt)", "minimal": "Minimal"}

METHODS = ["Gap", "Case arrival trend", "Temporal clustering", "Spatial clustering (OPTICS)",
           "Outlier detection", "Sequential patterns"]
CONTAMINATIONS = [0.01, 0.025, 0.05, 0.10, 0.15]
SIGNIFICANCE = [0.01, 0.05, 0.10]
EFFECT_THRESHOLDS = [0.0, 0.25, 0.5, 1.0, 2.0]  # % of mean weekly arrivals
EPS_FRACTIONS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
DEFAULT_EPS_FRACTION = 0.02


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


# -----------------------------------------------------------------------------
# Detection (one method, returns pattern counts)
# -----------------------------------------------------------------------------

def detect(method, df, view):
    x, y, color = view
    if method == "Gap":
        det = GapPattern(view_config={"x": x, "y": y}, gap_mode="transition", y_is_categorical=True)
        det.detect(df)
        gaps = det.detected["abnormal_gaps"] if det.detected else []
        severe = sum(g["severity"] >= 2 for g in gaps)
        return {"patterns": len(gaps), "detail": f"{severe} severe (>=2x), {len(gaps) - severe} mild"}
    if method == "Case arrival trend":
        det = CaseArrivalTrendPattern(view_config={"x": x})
        det.detect(df)
        s = det.get_summary()
        return {"patterns": s["count"],
                "detail": f"{s.get('direction')}, slope {s.get('slope_percent', 0):+.2f}%, p={s.get('p_value', 1):.3g}"}
    if method == "Temporal clustering":
        det = TemporalClusterPattern(df=df, x_axis=x, y_axis=y, color=color)
        det.detect()
        bursts = len(det.clusters.get("temporal_bursts", []))
        return {"patterns": bursts, "detail": f"eps={det.temporal_eps / 86400:.1f} d, m_min={det.min_cluster_size}"}
    if method == "Spatial clustering (OPTICS)":
        det = ClusterPattern(view_config={"x": x, "y": y}, algorithm="optics")
        det.detect(df)
        return {"patterns": det.get_summary().get("count", 0), "detail": ""}
    if method == "Outlier detection":
        det = od.OutlierDetectionPattern(df=df, view_config={"x": x, "y": y, "color": color})
        det.detect()
        n_if = len(det.outlier_indices)
        n_multi = len(det.get_multi_reason_outliers())
        return {"patterns": min(n_multi, 500),
                "detail": f"{n_if} Isolation Forest outliers, {n_multi} multi-reason"}
    if method == "Sequential patterns":
        det = HorizontalSequencePatternDetector(x_axis=x, y_axis=y, dot_color=color, df=df, min_support=30)
        det.detect()
        return {"patterns": det.get_summary().get("count", 0), "detail": "min_support=30%"}
    raise ValueError(method)


def _child(fn, args, conn):
    # Cap the address space so that a runaway allocation raises MemoryError
    # instead of triggering the OOM killer on the whole machine.
    with open("/proc/self/statm") as f:
        vsz = int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    resource.setrlimit(resource.RLIMIT_AS, (vsz + MEM_LIMIT_MB * 2**20, resource.RLIM_INFINITY))
    base = rss_mb()
    peak = [base]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], rss_mb())
            time.sleep(0.01)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    try:
        t0 = time.perf_counter()
        value = fn(*args)
        out = {"status": "ok", "value": value, "time_s": time.perf_counter() - t0}
    except MemoryError:
        out = {"status": f"out of memory (> {MEM_LIMIT_MB / 1024:.0f} GB)", "value": None, "time_s": np.nan}
    except Exception as e:
        out = {"status": f"error: {str(e)[:80]}", "value": None, "time_s": np.nan}
    stop.set()
    th.join()
    out["peak_mem_mb"] = max(peak[0], rss_mb()) - base
    conn.send(out)
    conn.close()


def run_isolated(fn, *args):
    """Run fn(*args) in a forked child with a timeout and a memory cap."""
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(fn, args, child))
    p.start()
    child.close()
    try:
        out = parent.recv() if parent.poll(TIMEOUT) else None
    except EOFError:
        out = {"status": "killed", "value": None, "time_s": np.nan, "peak_mem_mb": np.nan}
    if out is None:
        p.kill()
        out = {"status": f"timeout (> {TIMEOUT} s)", "value": None, "time_s": np.nan, "peak_mem_mb": np.nan}
    p.join()
    return out


def run_method(method, df, view):
    out = run_isolated(detect, method, df, view)
    res = out.pop("value") or {"patterns": np.nan, "detail": ""}
    return {**res, **out}


# -----------------------------------------------------------------------------
# Sensitivity analyses (each setting in its own child)
# -----------------------------------------------------------------------------

def _outliers(df, view, c):
    x, y, color = view
    orig = od.IsolationForest
    od.IsolationForest = lambda **kw: orig(**{**kw, "contamination": c})
    det = od.OutlierDetectionPattern(df=df, view_config={"x": x, "y": y, "color": color})
    det.detect()
    multi = det.get_multi_reason_outliers()
    top5 = sorted(multi, key=lambda i: -det.outlier_scores.get(i, 0))[:5]
    row = {"contamination": c, "if_outliers": len(det.outlier_indices),
           "if_outliers_pct": 100 * len(det.outlier_indices) / len(df),
           "multi_reason": len(multi), "shown": min(len(multi), 500),
           "shown_pct": 100 * min(len(multi), 500) / len(df)}
    return row, set(multi), top5


def sens_contamination(df, view):
    rows, sets = [], {}
    for c in CONTAMINATIONS:
        out = run_isolated(_outliers, df, view, c)
        if out["value"] is None:
            rows.append({"contamination": c, "status": out["status"]})
            continue
        row, multi, top5 = out["value"]
        sets[c] = (multi, top5)
        rows.append({**row, "status": "ok", "time_s": out["time_s"], "peak_mem_mb": out["peak_mem_mb"]})
    ref, ref_top5 = sets.get(0.05, (None, None))
    for r in rows:
        if ref is None or r["contamination"] not in sets:
            continue
        s, top5 = sets[r["contamination"]]
        r["jaccard_multi_vs_default"] = len(s & ref) / len(s | ref) if s | ref else 1.0
        r["default_multi_retained_pct"] = 100 * len(s & ref) / len(ref) if ref else np.nan
        r["top5_overlap_with_default"] = len(set(top5) & set(ref_top5))
    return rows


def classify(p, slope_pct, alpha, thr):
    if p <= alpha and abs(slope_pct) >= thr:
        return "increasing" if slope_pct > 0 else "decreasing"
    return "stable" if p <= alpha else "no_trend"


def _trend(df, view):
    det = CaseArrivalTrendPattern(view_config={"x": view[0]}, use_prophet=False)
    det.detect(df)
    return det.trend_result or {}


def sens_trend(df, view):
    tr = run_isolated(_trend, df, view)["value"] or {}
    p, slope = tr.get("p_value", np.nan), tr.get("slope_percent", np.nan)
    rows = []
    for a in SIGNIFICANCE:
        for thr in EFFECT_THRESHOLDS:
            rows.append({"p_value": p, "slope_pct": slope, "cases": tr.get("total_cases", np.nan),
                         "alpha": a, "effect_threshold_pct": thr,
                         "direction": classify(p, slope, a, thr) if tr else "error"})
    return rows


def _bursts(df, view, eps):
    x, y, color = view
    det = TemporalClusterPattern(df=df, x_axis=x, y_axis=y, color=color, temporal_eps=eps)
    det._detect_temporal_bursts()
    lab = pd.Series(-1, index=df.index)
    for k, ix in det.cluster_point_indices.items():
        lab.loc[ix] = k
    bursts = det.clusters.get("temporal_bursts", [])
    sizes = [b.get("event_count", 0) for b in bursts] if isinstance(bursts, list) else []
    return {"m_min": det.min_cluster_size, "bursts": len(bursts),
            "largest_burst_events": max(sizes, default=0)}, lab.to_numpy()


def sens_eps(df, view):
    t = pd.to_datetime(df[view[0]])
    span = (t.max() - t.min()).total_seconds()
    labels, rows = {}, []
    for f in EPS_FRACTIONS:
        out = run_isolated(_bursts, df, view, f * span)
        row = {"eps_fraction": f, "eps_days": f * span / 86400, "status": out["status"],
               "time_s": out["time_s"], "peak_mem_mb": out["peak_mem_mb"]}
        if out["value"] is not None:
            info, labels[f] = out["value"]
            row.update(info, events_in_bursts_pct=100 * (labels[f] >= 0).mean())
        rows.append(row)
    for r in rows:
        if DEFAULT_EPS_FRACTION in labels and r["eps_fraction"] in labels:
            r["ari_vs_default"] = adjusted_rand_score(labels[DEFAULT_EPS_FRACTION], labels[r["eps_fraction"]])
    return rows


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def append(name, rows):
    if not rows:
        return
    path = OUT / name
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def describe(df):
    span = df["actual_time"].max() - df["actual_time"].min()
    variants = df.sort_values(["case_id", "actual_time"]).groupby("case_id")["activity"].agg(tuple).nunique()
    return {"events": len(df), "cases": df["case_id"].nunique(), "activities": df["activity"].nunique(),
            "resources": df["resource"].nunique() if "resource" in df else np.nan,
            "variants": variants, "span_days": span.total_seconds() / 86400,
            "df_memory_mb": df.memory_usage(deep=True).sum() / 2**20}


def run_view(log_name, sampling, df, view, sampling_time, full_df):
    meta = {"log": log_name, "sampling": SAMPLING_LABEL[sampling.value]}
    append("datasets.csv", [{**meta, **describe(df), "sampling_time_s": sampling_time,
                             "events_retained_pct": 100 * len(df) / len(full_df),
                             "cases_retained_pct": 100 * df["case_id"].nunique() / full_df["case_id"].nunique()}])
    for m in METHODS:
        gc.collect()
        res = run_method(m, df, view)
        log(f"  {log_name} / {sampling.value} / {m}: {res['status']}, {res['patterns']} patterns, "
            f"{res['time_s']:.2f} s, {res['peak_mem_mb']:.0f} MB")
        append("performance.csv", [{**meta, "events": len(df), "method": m, **res}])
    for name, fn in [("sens_contamination.csv", sens_contamination), ("sens_trend.csv", sens_trend),
                     ("sens_eps.csv", sens_eps)]:
        try:
            append(name, [{**meta, **r} for r in fn(df, view)])
        except Exception as e:
            log(f"  {name} failed: {e}")
    log(f"  {log_name} / {sampling.value}: sensitivity done")


def prepare(df):
    df = df.dropna(subset=["actual_time", "activity"]).reset_index(drop=True)
    df["actual_time"] = pd.to_datetime(df["actual_time"], utc=True).dt.tz_localize(None)
    return df


def run_log(log_name):
    path = LOGS["Road Traffic Fine" if log_name == PAPER_VIEW else log_name]
    t0 = time.perf_counter()
    full = load_xes_log(str(path))
    load_time = time.perf_counter() - t0
    log(f"{log_name}: loaded {len(full)} events in {load_time:.1f} s")
    append("load_times.csv", [{"log": log_name, "load_time_s": load_time, "file_mb": path.stat().st_size / 2**20}])

    if log_name == PAPER_VIEW:
        t0 = time.perf_counter()
        df, _ = sample_eventlog_variant_aware(full, mode=SamplingMode.SQRT, case_col="case_id",
                                              activity_col="activity", time_col="actual_time", random_state=42)
        st = time.perf_counter() - t0
        df = df.dropna(subset=["actual_time", "activity"]).reset_index(drop=True)
        df = df[df["actual_time"] >= pd.Timestamp("2009-01-01", tz="UTC")]
        df["actual_time"] = df["actual_time"].dt.tz_localize(None)
        run_view(log_name, SamplingMode.SQRT, df, ("actual_time", "activity", "case_id"), st, prepare(full))
        return

    for s in SAMPLINGS:
        t0 = time.perf_counter()
        if s == SamplingMode.FULL:
            df = full
        else:
            df, _ = sample_eventlog_variant_aware(full, mode=s, case_col="case_id", activity_col="activity",
                                                  time_col="actual_time", random_state=42)
        st = time.perf_counter() - t0
        run_view(log_name, s, prepare(df), ("actual_time", "activity", "resource"), st, full)
        del df
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=[*LOGS, PAPER_VIEW])
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        run_log(args.child)
        return
    # One process per log, so that memory is released between logs
    import subprocess
    for name in args.logs:
        log(f"=== {name} ===")
        subprocess.run([sys.executable, __file__, "--child", name], check=False)


if __name__ == "__main__":
    main()
