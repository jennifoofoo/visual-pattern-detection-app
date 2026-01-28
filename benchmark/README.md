# Benchmark Suite

This directory contains the benchmark suite for the Visual Pattern Detection App. The main entry point for running benchmarks is `run_benchmark_suite.py`.

## Running the Benchmark Suite

To run the full benchmark suite on all available `.xes` files in the data directory:

```bash
python run_benchmark_suite.py
```

### Options

**Run specific algorithms:**
You can specify which algorithms to benchmark (must be quoted if they contain spaces):
```bash
python run_benchmark_suite.py --algorithms "Sequence Detection" "Gap (Transition)"
```

**Run a quick test:**
Use `--test-run` to run on a small subset of the data for verification:
```bash
python run_benchmark_suite.py --test-run
```

**Benchmark a specific file:**
To benchmark a single file instead of all files in the data directory:
```bash
python run_benchmark_suite.py --file "path/to/your/file.xes"
```

**Generate report only:**
If you have already run the benchmarks and just want to regenerate the consolidated report from existing CSV results:
```bash
python run_benchmark_suite.py --report-only
```

## Results

Results are generated in the `results/` directory:
- `results/individual/`: Contains individual CSV logs and reports for each algorithm run.
- `results/consolidated/`: Contains the consolidated CSV (`benchmark_consolidated.csv`) and the main benchmark report (`benchmark_report_consolidated.md`).

Change the directory name afterwards. 