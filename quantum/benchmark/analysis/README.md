# Sweep analysis

Post-processing for a completed sweep: reads the per-cell benchmark JSONs
listed in `<sweep_dir>/index.json` and writes CSV tables, Plotly figures, a
LaTeX snippet, and (optionally) a Hugging Face dataset upload.

**Methodology (why the paired Wilcoxon test, why energy is within-solver
only, how `manifest.json` works, what a sweep config looks like) is in the
parent [`../README.md`](../README.md).** This file is the per-column
reference for the output.

## Pipeline

```
index.json + benchmark_*.json
      │  load_sweep()            → runs_long           (one row per solver run)
      │  load_robot_statistics() → robot_statistics_long (one row per run×robot)
      ▼
compute_energy_excess()          adds within-solver energy_excess columns
compute_success_rate() + compute_variable_reduction_stats() → summary_by_solver
run_statistical_tests()          → statistical_tests
compute_energy_diagnostics()     → energy_diagnostics
compute_failure_causes()         → failure_causes
      ▼
generate_all_plots()   → analysis/plots/*.{html,png}
export_benchmark_table() → analysis/benchmark_table.tex
```

`aggregate_sweep(sweep_dir)` runs everything down to the CSVs and returns the
DataFrames too. Each function is independently importable (no CLI required),
`run_*.py` are thin argparse wrappers.

| Module                  | CLI                         | Output                                                                                                                                                                                                                                                                            |
| ----------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aggregate.py`        | `run_aggregate.py`        | the six CSVs below                                                                                                                                                                                                                                                                |
| `plots.py`            | `run_plots.py`            | `plots/{scaling,success_rate,variable_reduction,energy_excess,path_efficiency}.{html,png}`                                                                                                                                                                                      |
| `latex_export.py`     | (via`run_report.py`)      | `benchmark_table.tex` — mean time + mean reduction per (problem, grid, robots, solver); `\input{}`-able into `paper.tex`                                                                                                                                                   |
| `variable_scaling.py` | `run_variable_scaling.py` | QUBO size per preprocess mode, no solving                                                                                                                                                                                                                                         |
| `publish.py`          | `python -m …publish`     | uploads`index.json` + `manifest.json` + `analysis/*.csv` + `.tex` to a HF dataset (not the raw JSON, not the plots). The dataset card's canonical source is [`DATASET_CARD.md`](DATASET_CARD.md) |
| —                      | `run_report.py`           | aggregate → plots → LaTeX in one call                                                                                                                                                                                                                                           |

Missing-input handling is graceful: no `energy_excess` column → energy plot
skipped; sweep run at `BenchmarkRunner` level < 2 → `robot_statistics_long`
is empty and the path-efficiency plot and metric are skipped.

`_resolve_artifact()` lets a finished sweep directory be moved.
Paths in `index.json` are tried first, then the stable
`<sweep_dir>/<cell_dir>/<file>` layout.

---

## `runs_long.csv` — one row per individual solver run

The base table; everything else is derived from it.

| Column                                                                                                                            | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `instance_map`, `grid_size`, `problem_name`, `num_robots`                                                                 | Instance identity.`grid_size` is `MxN`, or `<k>nodes` for graph problems.                                                                                                                                                                                                                                                                                                                                                                                              |
| `solver_name`, `backend`, `device`, `penalty_set`                                                                         | Solver identity.`solver_name` is the sweep-config label and is the unit that `run_statistical_tests` pairs on. `penalty_set` is `None` for `ilp`/`cbs`.                                                                                                                                                                                                                                                                                                          |
| `preprocess`                                                                                                                    | Ablation arm.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `run_id`, `timestamp`                                                                                                         | Within the cell's`num_runs`.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `valid`                                                                                                                         | Path passed full validation. May be`None` on pre-`valid`-key JSONs — treat as `False` (`_valid_mask`).                                                                                                                                                                                                                                                                                                                                                              |
| `invalid_cause`                                                                                                                 | `pre_processing` \| `solver_sampling` \| `None`. See `failure_causes.csv`.                                                                                                                                                                                                                                                                                                                                                                                           |
| `energy`                                                                                                                        | Solver's own raw objective.**Only comparable within `(instance_map, problem_name, solver_name, penalty_set, preprocess)`.** `0.0` can be a "no window contributed an energy" sentinel when BFS consumed every variable — `_solver_ran_mask` filters those.                                                                                                                                                                                                      |
| `execution_time_sec`                                                                                                            | Wall clock. A censored lower bound when`termination_condition` is a time/node limit.                                                                                                                                                                                                                                                                                                                                                                                       |
| `num_windows`, `total_initial_variables`, `total_variables_reduced`, `total_final_variables`, `average_reduction_ratio` | Windowing / pre-processing size stats. Absent for`ilp`/`cbs`.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `termination_condition`                                                                                                         | e.g.`optimal`, `time_limit_exceeded`, `maxIterations`. Drives censoring.                                                                                                                                                                                                                                                                                                                                                                                               |
| `avg_path_efficiency`, `min_path_efficiency`, `robot_success_rate`                                                          | From decoded paths;**commensurable across solvers**. Need level ≥ 2.                                                                                                                                                                                                                                                                                                                                                                                                  |
| `reference_energy`, `energy_excess`, `reference_missing`, `energy_scale_mismatch`                                         | Added by`compute_energy_excess`. `energy_excess = energy − (min valid energy in the same config)`, in raw units: `0` for that config's best run, `> 0` for worse ones, `NaN` if the config has no valid run. A **difference, not a ratio**, a QUBO Hamiltonian has no meaningful zero. `energy_scale_mismatch = True` when runs in the config used differing window counts (different dropped constants → not strictly comparable even within-config). |

---

## `summary_by_solver.csv`

`compute_success_rate` ⋈ `compute_variable_reduction_stats`, keyed by
`(instance_map, problem_name, solver_name, preprocess)`.

| Column                                               | Notes                                                                                                   |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `success_rate`                                     | Mean of`valid` over the cell's runs. **Read this next to every `statistical_tests.csv` row.** |
| `mean_reduction_ratio`                             | Mean`average_reduction_ratio`. High ⇒ the cell is mostly BFS output.                                 |
| `mean_initial_variables`, `mean_final_variables` | QUBO size before/after pre-processing.                                                                  |

---

## `statistical_tests.csv` — paired Wilcoxon signed-rank

One row per `(solver_a, solver_b, metric)`; default pairs `(candidate, baseline)` with `ilp`/`cbs` as baselines. Every directional column is
`solver_a` **relative to** `solver_b`. Full rationale in `../README.md`.

| Column                                                                   | Notes                                                                                                                                                                                        |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `metric`                                                               | `execution_time_sec`, `avg_path_efficiency`, … Values are each solver's **per-instance mean over its valid runs**, then paired by instance.                                       |
| `transform`                                                            | `log` for time metrics (Wilcoxon ranks absolute differences → without it the biggest instances decide the ranking). `None` otherwise.                                                   |
| `n_pairs`                                                              | Instances where both solvers have a metric value.                                                                                                                                            |
| `n_effective`                                                          | `n_pairs` minus exact ties (scipy `zero_method="wilcox"` discards zero differences). The test's real *n*. Guard: both must be ≥ 6 (`_MIN_PAIRS_FOR_WILCOXON`).                      |
| `n_dropped_unshared` / `n_dropped_invalid` / `n_dropped_no_metric` | Survivorship accounting -> why instances fell out of the pairing (one solver didn't attempt / no valid run / metric not recorded).                                                           |
| `min_valid_runs`                                                       | Fewest valid runs behind any paired cell.`1` = some instance rests on one lucky run.                                                                                                       |
| `n_censored`                                                           | Paired instances where either solver hit a time/node limit → test understates the true difference.                                                                                          |
| `median_diff`                                                          | Median of the (possibly log) paired differences.                                                                                                                                             |
| `median_ratio`                                                         | `exp(median_diff)` for log metrics -> "solver_a takes N× as long". `NaN` otherwise.                                                                                                     |
| `rank_biserial`                                                        | Matched-pairs effect size`(W⁺−W⁻)/(W⁺+W⁻)` ∈ [−1, 1]. The companion to the p-value.                                                                                                 |
| `favors`                                                               | `rank_biserial` resolved to a solver name via the metric's `higher_is_better`; `None` on a tie.                                                                                        |
| `statistic`, `p_value`                                               | scipy`wilcoxon` output. `p_value` is **raw**.                                                                                                                                      |
| `p_value_bh`                                                           | Benjamini–Hochberg FDR-adjusted across all real-p-value rows in the file.**Use this for significance claims**, not `p_value`.                                                       |
| `note`                                                                 | Why a row has no p-value (`insufficient_data`, `insufficient_effective_pairs`, `log_transform_invalid`, `not_comparable_across_solvers`, `metric_missing`) or a censoring warning. |

Energy metrics are refused here (`_WITHIN_SOLVER_ONLY_METRICS`) — a QUBO
energy, an ILP objective and a CBS sum-of-costs are unrelated quantities.

---

## `energy_diagnostics.csv` — per energy class

One row per `(instance_map, problem_name, solver_name, penalty_set, preprocess)`: is this configuration's energy landscape actually healthy.
Full column semantics in `../README.md` (§ Energy-landscape diagnostics).
Quick read:

- `separated = False` / `n_inversions > 0` → an invalid run scored below a
  valid one → the violated constraint's penalty is **too weak**.
- `quality_rho ≥ 0` → lower energy is selecting **worse** paths → recalibrate
  the penalty set. (Want it clearly negative.)
- `median_reduction_ratio ≥ 0.9` or `n_solver_skipped > 0` → the rows are
  mostly **BFS**, not the solver → run the `preprocess` ablation.
- `hit_rate_best`, `energy_iqr`, `max_excess` → stochastic-sampler
  consistency (tuning, not formulation).
- All magnitude columns are raw units — **never cross-configuration**.

---

    

## `failure_causes.csv`

Invalid runs per `(instance, problem, solver, preprocess)` by `invalid_cause`:

| Cause               | Meaning                                                                                                     | Fix                                    |
| ------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `pre_processing`  | BFS / diagonal fixing pinned a conflict before the solver ran; the variables to avoid it were already gone. | Formulation / windowing — not tuning. |
| `solver_sampling` | Solver returned a bitstring violating a penalty that*was* in the Hamiltonian.                             | Solver params or penalty weights.      |
| `not_recorded`    | Older JSON without the cause field.                                                                         | —                                     |

Empty (correctly-shaped) if the sweep had no invalid runs.

---

## `robot_statistics_long.csv` — one row per (run, robot)

The per-robot detail behind `runs_long`'s `avg_/min_path_efficiency`. Empty
unless the sweep ran at `BenchmarkRunner` level ≥ 2. Columns: `path_length`,
`moves_taken`, `optimal_path_length`, `path_efficiency`, `goal_reached`,
`validation_passed`, `priority`, plus the run/instance/solver identity.
