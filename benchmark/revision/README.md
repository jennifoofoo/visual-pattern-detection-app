# Quantitative Evaluation (SoftwareX paper, Section 3.2)

Scripts and results of the quantitative evaluation and sensitivity analysis reported in Section 3.2 (Tables 2–4) of the SoftwareX paper.

## Contents

| File | Description |
|---|---|
| `run_revision_experiments.py` | Runs all experiments and writes the CSV files in `results/` |
| `revision_tables.xlsx` | Tables of the paper, ready to inspect |
| `results/datasets.csv` | Size of each log and of each sample (Table 2) |
| `results/load_times.csv` | XES file size and loading time (Table 2) |
| `results/performance.csv` | Time, peak additional memory, and detected patterns per method (Table 3) |
| `results/sens_contamination.csv` | Isolation Forest contamination α (Table 4a) |
| `results/sens_trend.csv` | Mann–Kendall p-value and Sen's slope for each trend threshold (Table 4b) |
| `results/sens_eps.csv` | Burst radius ε as a fraction of the time span (Table 4c) |

## Setup

- **View configuration:** *Activity overview* (X = `actual_time`, Y = `activity`, Color = `resource`), for which all methods are fully applicable.
- **Sampling strategies:** Full, Optimized (≤ 50,000 events), Balanced / `SQRT` (≤ 20,000 events), Minimal (≤ 5,000 events).
- **Limits:** each detection runs in a separate process with a 600 s timeout and a 3 GB cap on additional memory; memory is the peak RSS above the RSS at fork time, sampled every 10 ms.
- **Hardware used in the paper:** laptop with Intel Core Ultra 7 256V CPU and 16 GB RAM.
- **Parameter grids:**
  - contamination α ∈ {0.01, 0.025, **0.05**, 0.10, 0.15}
  - significance level ∈ {0.01, **0.05**, 0.10}; effect-size threshold ∈ {0, 0.25, **0.5**, 1, 2}% of mean weekly arrivals
  - ε ∈ {0.5, 1, 1.5, **2**, 3, 5}% of the time span (defaults in bold)

An additional entry, `Road (paper view)`, reproduces the case-study view of Section 3.1 (Balanced sampling, seed 42, events from 2009-01-01, Color = `case_id`).

## Event logs

The logs are public and are not included in the repository. Download them from the 4TU.ResearchData repository and place them in `data/` as follows:

| Log | Expected path |
|---|---|
| Sepsis Cases | `data/Sepsis Cases - Event Log.xes/Sepsis Cases - Event Log.xes` |
| BPI Challenge 2020 – Domestic Declarations | `data/BPI_Challange_2020/DomesticDeclarations.xes` |
| BPI Challenge 2020 – International Declarations | `data/BPI_Challange_2020/InternationalDeclarations.xes` |
| BPI Challenge 2011 (Hospital log) | `data/Hospital_log.xes` |
| BPI Challenge 2012 | `data/BPI_Challenge_2012/BPI_Challenge_2012.xes` |
| Road Traffic Fine Management Process | `data/Road.xes` |

## Running

From the repository root:

```bash
# all logs (slow on the complete logs, because of the 600 s timeout per method)
venv/bin/python benchmark/revision/run_revision_experiments.py

# selected logs only
venv/bin/python benchmark/revision/run_revision_experiments.py --logs Sepsis "BPI2012"
```
