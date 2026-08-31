---
license: apache-2.0
pretty_name: Spooky MAPF Benchmark Sweeps
tags:
  - robotics
  - multi-agent-path-finding
  - path-planning
  - quantum-computing
  - qubo
  - qaoa
  - quantum-annealing
  - benchmark
  - optimization
---

# Spooky MAPF Benchmark Sweeps

Aggregated benchmark results for **[Spooky](https://github.com/JavideuS/Spooky)**,
a hybrid quantum–classical framework for Multi-Agent Path Finding (MAPF) via
QUBO. Each entry here is one *sweep* — a full
`instance × problem × solver × ablation` matrix run through Spooky's
`BenchmarkRunner` and aggregated into CSV tables.

- **Code:** <https://github.com/JavideuS/Spooky>
- **Paper:** *Scalable Multi-Robot Path Planning via Quadratic Unconstrained
  Binary Optimization*, arXiv:[2602.14799](https://arxiv.org/abs/2602.14799)
- **Served by:** the Spooky FastAPI Space, whose `/v1/analysis/*` endpoints
  `snapshot_download` this repo (`allow_patterns=["sweeps/**", "published.json"]`)
  and render Plotly figures from the CSVs on demand.

Solvers compared across sweeps: `ilp` (HiGHS), `cbs` (Conflict-Based Search),
`dwave` (simulated annealing via `neal`, no QPU), `pennylane` (simulated
QAOA), and, where a sweep enabled hardware, real QPU / IBM backends.

## Repository layout

```
published.json                       ledger — one entry per published sweep
sweeps/
└── <sweep_id>/
    ├── index.json                   the run plan: one entry per matrix cell
    ├── manifest.json                reproducibility record (git, deps, hardware, seed)
    └── analysis/
        ├── runs_long.csv            one row per individual solver run
        ├── summary_by_solver.csv    success rate + variable-reduction means
        ├── statistical_tests.csv    paired Wilcoxon signed-rank tests
        ├── energy_diagnostics.csv   per-config energy-landscape health
        ├── failure_causes.csv       invalid runs split by cause
        ├── robot_statistics_long.csv one row per (run, robot); empty if the
        │                             sweep ran below BenchmarkRunner level 2
        └── benchmark_table.tex      paper-style LaTeX table snippet
```

### Not included

- **Raw per-cell `benchmark_*.json`** — they carry full decoded robot paths
  and run to tens of MB per sweep. The CSVs are sufficient to serve and to
  reproduce every figure.
- **Rendered plots (`analysis/plots/`)** — see *Regenerating the figures*
  below. A ~30 MB sweep directory ships here as a few hundred KB.

### `published.json`

Object keyed by `sweep_id`. Each value:

| Field | Meaning |
|---|---|
| `commit`, `branch` | Spooky git state the sweep ran against. |
| `published_at` | UTC ISO timestamp of the upload. |
| `start_time`, `end_time` | From the sweep's `manifest.json`. A `null` `end_time` means the sweep was interrupted and its matrix is incomplete. |
| `n_completed` | Matrix cells with a completed benchmark JSON. |
| `solvers`, `problems`, `grid_sizes` | Coverage of the sweep. |
| `dropped_solvers` | Solvers filtered out of the CSVs before upload (`--drop-solver`), or `null`. |

Convention: normally **one sweep per Spooky commit**. Two different sweep
*configs* on one commit is legitimate and they publish side by side; a
re-run of the same config replaces the prior sweep for that commit.

## How to read the tables

Full column dictionaries and methodology are in the code repo:

- **`quantum/benchmark/analysis/README.md`** — per-column reference for every
  CSV here.
- **`quantum/benchmark/README.md`** — why the paired Wilcoxon signed-rank
  test, the log transform on time metrics, the rank-biserial effect size,
  Benjamini–Hochberg correction, and the survivorship/censoring accounting.

Pin those to the sweep's own commit — `sweeps/<sweep_id>/manifest.json` →
`git.commit` — for the doc version that matches the numbers:
`https://github.com/JavideuS/Spooky/blob/<commit>/quantum/benchmark/analysis/README.md`

**Before trusting any row of `statistical_tests.csv`:**

- Use **`p_value_bh`** (FDR-adjusted across the whole family of tests), not
  the raw `p_value`.
- Read it next to `summary_by_solver.csv`. A paired test only compares
  instances *both* solvers solved, so a solver that only succeeds on easy
  instances is compared only on easy instances. Check `n_dropped_invalid`,
  `n_dropped_unshared`, `min_valid_runs`, `n_censored` and `success_rate`.
- `n_effective` (not `n_pairs`) is the test's real sample size; both must be
  ≥ 6 or there is no p-value.
- Energy is **not** in the cross-solver tests — a QUBO Hamiltonian value, an
  ILP objective and a CBS sum-of-costs are unrelated quantities. Use the
  within-solver `energy_excess` column in `runs_long.csv` for energy.

## Regenerating the figures

Only the tables are stored. Every figure is a pure function of
`runs_long.csv` (+ `robot_statistics_long.csv`). **No re-solving is
needed**.

```bash
# 1. Install Spooky's analysis + plotting extras (pin to the sweep's commit)
pip install "spooky[benchmark,visualizer] @ git+https://github.com/JavideuS/Spooky@<commit>"

# 2. Pull one sweep directory from this dataset
hf download JavideuS/Spooky-benchmark --repo-type dataset \
    --include "sweeps/<sweep_id>/**" --local-dir ./spooky-sweeps

# 3. Render the plots into <sweep_dir>/analysis/plots/
python -m quantum.benchmark.analysis.run_plots \
    -d ./spooky-sweeps/sweeps/<sweep_id>
```

Output (HTML always; PNG/PDF/SVG when `kaleido` is present — it is in the
`visualizer` extra):

| File | Content |
|---|---|
| `scaling.{html,png,pdf,svg}` | Execution time vs problem size, per solver. |
| `success_rate.{…}` | Valid-solution rate per solver × instance. |
| `variable_reduction.{…}` | Pre-processing variable reduction per solver. |
| `energy_excess.{…}` | Within-solver energy spread above best (QUBO backends). |
| `path_efficiency.{…}` | Per-robot path efficiency — only if `robot_statistics_long.csv` is non-empty (sweep run at level ≥ 2). |

`python -m quantum.benchmark.analysis.run_report -d <sweep_dir>` also
re-derives `benchmark_table.tex` alongside the plots.

## Reproducing a sweep from scratch

`sweeps/<sweep_id>/manifest.json` embeds the entire sweep config under
`sweep_config`, plus the git commit, package versions, hardware, and
`global_seed`.

```bash
git clone https://github.com/JavideuS/Spooky && cd Spooky
git checkout <commit-from-manifest>
pip install -e ".[benchmark,classical,visualizer]"
spooky-sweep --config sweep_configs/<name>.yaml     # same seed → same run
python -m quantum.benchmark.analysis.run_aggregate -d results/sweeps/<new_id>
```

Classical-only configs (`ilp`, `cbs`, simulated `dwave`/`pennylane`) need no
tokens or quota and are fully reproducible from the seed. QPU/IBM rows are
not bit-reproducible and are gated behind explicit flags in `spooky-sweep`.

## License

Apache-2.0, matching the Spooky code repository.

## Citation

```bibtex
@misc{gonzalezvillasmil2026scalablemultirobotpathplanning,
      title={Scalable Multi-Robot Path Planning via Quadratic Unconstrained Binary Optimization},
      author={Javier González Villasmil et al.},
      year={2026},
      eprint={2602.14799},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.14799},
}
```
