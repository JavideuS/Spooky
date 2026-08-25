"""
pandas/scipy aggregation for a sweep's results.

Reads index.json, flattens every individual run into one row of
a long-format DataFrame, and computes the derived columns/tables
that later the plots and LaTeX export read from.

Every function here is a plain, independently-importable function (not
CLI-only) so a future FastAPI endpoint could call aggregate_sweep()
directly — see run_aggregate.py for the CLI wrapper.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

# Wilcoxon signed-rank has essentially no discriminatory power below this
# many paired instances (commonly cited minimum for the test to be
# meaningfully computable at all)
_MIN_PAIRS_FOR_WILCOXON = 6


def _robot_efficiency_summary(solution_statistics: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """(avg, min) path_efficiency across a run's robots, or (None, None) if
    solution_statistics wasn't recorded (needs BenchmarkRunner level>=2 —
    see quantum.benchmark.benchmark._compute_solution_statistics)."""
    robot_stats = (solution_statistics or {}).get("robot_statistics") or {}
    efficiencies = [
        rs["path_efficiency"] for rs in robot_stats.values() if "path_efficiency" in rs
    ]
    if not efficiencies:
        return None, None
    return sum(efficiencies) / len(efficiencies), min(efficiencies)


def _resolve_artifact(sweep_dir: Path, recorded: str) -> Path:
    """Locate a benchmark JSON named in index.json.

    The index records the path each artifact had when the sweep ran. Filing a
    finished sweep somewhere else afterwards (results/sweeps/CML/<id>/, an
    archive, another machine) invalidates every one of those paths, and the
    whole aggregation then dies on the first entry. The layout inside a sweep
    directory is stable — <sweep_dir>/<combination_dir>/<file> — so fall back
    to that before giving up."""
    recorded_path = Path(recorded)
    if recorded_path.exists():
        return recorded_path
    relocated = sweep_dir / recorded_path.parent.name / recorded_path.name
    if relocated.exists():
        return relocated
    raise FileNotFoundError(
        f"{recorded} is missing, and so is {relocated} — index.json points "
        f"outside {sweep_dir} and the artifact is not where a moved sweep "
        "would have put it either."
    )


def load_sweep(sweep_dir: str) -> pd.DataFrame:
    """Long-format DataFrame, one row per individual solver run."""
    sweep_dir = Path(sweep_dir)
    index_path = sweep_dir / "index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    rows: List[Dict[str, Any]] = []
    for entry in index:
        if entry.get("dry_run") or not entry.get("benchmark_json"):
            continue
        with open(
            _resolve_artifact(sweep_dir, entry["benchmark_json"]), "r", encoding="utf-8"
        ) as f:
            data = json.load(f)

        problem_meta = data["metadata"]["problem"]
        num_robots = len(problem_meta.get("robots", {}) or {})
        grid = problem_meta.get("grid")
        graph = problem_meta.get("graph")
        if grid:
            grid_size = f"{grid['M']}x{grid['N']}"
        elif graph:
            grid_size = f"{len(graph.get('nodes', []) or [])}nodes"
        else:
            grid_size = None

        for run in data["runs"]:
            var_stats = run.get("variable_stats", {}) or {}
            solution_stats = run.get("solution_statistics") or {}
            avg_efficiency, min_efficiency = _robot_efficiency_summary(solution_stats)
            rows.append(
                {
                    "sweep_id": sweep_dir.name,
                    "instance_map": entry["instance"],
                    "grid_size": grid_size,
                    "problem_name": entry["problem"],
                    "num_robots": num_robots,
                    "solver_name": entry["solver"],
                    "backend": entry["backend"],
                    "device": entry.get("device"),
                    "penalty_set": entry.get("penalty_set"),
                    "preprocess": entry["preprocess"],
                    "run_id": run.get("run_id"),
                    "valid": run.get("valid"),
                    # "pre_processing" when BFS/diag fixing forced the
                    # conflict before the solver ran, "solver_sampling" when
                    # the solver actually returned a violating bitstring. The
                    # difference decides whether to fix the formulation or the
                    # sampler, so it belongs in the table, not only the JSON.
                    "invalid_cause": (run.get("invalid_cause") or {}).get("origin"),
                    "energy": run.get("energy"),
                    "execution_time_sec": run.get("execution_time_sec"),
                    "num_windows": var_stats.get("num_windows"),
                    "total_initial_variables": var_stats.get("total_initial_variables"),
                    "total_variables_reduced": var_stats.get("total_variables_reduced"),
                    "total_final_variables": var_stats.get("total_final_variables"),
                    "average_reduction_ratio": var_stats.get("average_reduction_ratio"),
                    "termination_condition": run.get("termination_condition"),
                    "avg_path_efficiency": avg_efficiency,
                    "min_path_efficiency": min_efficiency,
                    "robot_success_rate": solution_stats.get("success_rate"),
                    "timestamp": run.get("timestamp"),
                }
            )

    return pd.DataFrame(rows)


def load_robot_statistics(sweep_dir: str) -> pd.DataFrame:
    """Long-format DataFrame, one row per (run, robot) — the per-robot
    detail behind load_sweep()'s avg_path_efficiency/min_path_efficiency
    summary columns (see quantum.benchmark.benchmark._compute_solution_statistics).
    Empty (but correctly shaped) if no run in the sweep was benchmarked at
    level>=2, since solution_statistics only exists there."""
    sweep_dir = Path(sweep_dir)
    index_path = sweep_dir / "index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    columns = [
        "sweep_id", "instance_map", "problem_name", "solver_name", "backend",
        "preprocess", "run_id", "robot_id", "path_length", "moves_taken",
        "optimal_path_length", "path_efficiency", "goal_reached",
        "validation_passed", "priority",
    ]
    rows: List[Dict[str, Any]] = []
    for entry in index:
        if entry.get("dry_run") or not entry.get("benchmark_json"):
            continue
        with open(
            _resolve_artifact(sweep_dir, entry["benchmark_json"]), "r", encoding="utf-8"
        ) as f:
            data = json.load(f)

        for run in data["runs"]:
            robot_stats = (run.get("solution_statistics") or {}).get(
                "robot_statistics"
            ) or {}
            for robot_id, rs in robot_stats.items():
                rows.append(
                    {
                        "sweep_id": sweep_dir.name,
                        "instance_map": entry["instance"],
                        "problem_name": entry["problem"],
                        "solver_name": entry["solver"],
                        "backend": entry["backend"],
                        "preprocess": entry["preprocess"],
                        "run_id": run.get("run_id"),
                        "robot_id": robot_id,
                        "path_length": rs.get("path_length"),
                        "moves_taken": rs.get("moves_taken"),
                        "optimal_path_length": rs.get("optimal_path_length"),
                        "path_efficiency": rs.get("path_efficiency"),
                        "goal_reached": rs.get("goal_reached"),
                        "validation_passed": rs.get("validation_passed"),
                        "priority": rs.get("priority"),
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    """`valid` coerced to a real bool Series.

    A benchmark JSON that predates the 'valid' key (or any run record
    missing it) yields None from run.get("valid"), which makes the whole
    column object-dtype — and boolean indexing on an object column raises
    rather than treating the None as False. Everything downstream wants
    "not known to be valid" == "not valid"."""
    if "valid" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    return df["valid"].apply(lambda v: bool(v) if pd.notna(v) else False).astype(bool)


def compute_optimality_gap(
    df: pd.DataFrame, reference_solver: str = "ilp"
) -> pd.DataFrame:
    """Removed — energy is not comparable across solvers. Use
    compute_energy_excess() (within-solver) for energy, and the path-derived
    columns (avg_path_efficiency / min_path_efficiency) for anything
    cross-solver."""
    raise NotImplementedError(
        "compute_optimality_gap() compared raw 'energy' across solvers, which is "
        "not a meaningful quantity: a QUBO solver's energy is its Hamiltonian "
        "value including every penalty term, scaled by that solver's "
        "normalize_scale and summed per window, while ILPBuilder's objective is "
        "a plain 'timesteps away from goal' count. Use compute_energy_excess(df) "
        "for the within-solver energy gap, and avg_path_efficiency for "
        "cross-solver solution quality."
    )


# An energy is only comparable to another energy produced from the *same*
# Hamiltonian by the *same* backend: penalty_set changes the Hamiltonian's
# terms, preprocess changes the windowing (and therefore how many per-window
# energies get summed), and each solver applies its own normalize_scale.
_ENERGY_CLASS_KEYS = [
    "instance_map",
    "problem_name",
    "solver_name",
    "penalty_set",
    "preprocess",
]


def _energy_class_keys(df: pd.DataFrame) -> List[str]:
    return [c for c in _ENERGY_CLASS_KEYS if c in df.columns]


def _solver_ran_mask(df: pd.DataFrame) -> pd.Series:
    """False for runs where pre-processing consumed every variable and the
    solver was never invoked.

    solve_qubo_smart()'s BFS/dead-end pruning can reduce a window to zero
    variables ("Window N: 31 -> 0 vars, skipping solver"). When that happens
    for every window the recorded energy is 0.0 — a sentinel meaning "no
    window contributed an energy", not a Hamiltonian value. Letting it
    through is actively harmful: on a valid run 0.0 would almost always be
    the smallest number in its class and would become the reference,
    poisoning every other run's excess.

    total_final_variables is absent for ILP/CBS (they have no variable_stats),
    so a missing value means "ran", not "skipped"."""
    if "total_final_variables" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    final_vars = pd.to_numeric(df["total_final_variables"], errors="coerce")
    return ~(final_vars == 0)


def compute_energy_excess(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'reference_energy', 'energy_excess', 'reference_missing' and
    'energy_scale_mismatch' columns.

    energy_excess = energy - reference, in the solver's own raw energy units.
    0 for that configuration's best run, > 0 for every worse one.

    Why a difference and not a percentage
    -------------------------------------
    A QUBO Hamiltonian has no meaningful zero. apply_one_hot() writes the
    -K_hot diagonal and +2*K_hot quadratic of K*(sum(x) - 1)^2 but drops the
    +K_hot constant, once per one-hot constraint, and the other penalty terms
    drop their constants the same way. On a 3x3 baseline at K_hot=9 that
    omitted offset is ~108 while a solved run's reported energy is ~-1.8, so
    a ratio (energy - ref)/|ref| is dividing almost entirely by offset:
    keeping the constants instead would turn the same pair of solutions from
    a 43% gap into a 0.7% one. The *difference* is invariant under that
    choice, the ratio is not. Energy is an interval scale, not a ratio scale,
    so it gets a difference.

    Why within-solver only
    ----------------------
    A QUBO solver reports its own (normalized, per-window-summed) Hamiltonian
    value; ILP reports "timesteps away from goal"; CBS reports sum-of-costs.
    Those are unrelated quantities. reference is therefore the best (minimum)
    energy that same configuration — (instance_map, problem_name,
    solver_name, penalty_set, preprocess) — reached on that instance across
    its *valid* runs. Configurations with no valid run get
    reference_missing=True and a NaN excess. For cross-solver quality use
    avg_path_efficiency / min_path_efficiency, which come from decoded paths
    and are commensurable.

    energy_scale_mismatch
    ---------------------
    energy is summed over windows (BaseSolver's per-window energy sum), and
    each window drops its own set of constants. Two runs of the same
    configuration that used a different number of windows therefore carry
    different offsets and their excesses are not strictly comparable either.
    That is flagged rather than silently tolerated.
    """
    df = df.copy()
    # a run whose solver never ran has no energy to speak of, valid or not
    usable = _valid_mask(df) & _solver_ran_mask(df)
    keys = _energy_class_keys(df)

    energy_when_valid = pd.to_numeric(df["energy"], errors="coerce").where(usable)
    # groupby-transform rather than groupby+merge: penalty_set is None for
    # every non-QUBO solver, and groupby's default dropna=True would delete
    # that whole group before the merge ever ran, leaving those rows with no
    # reference at all.
    grouper = [df[c] for c in keys]
    df["reference_energy"] = energy_when_valid.groupby(
        grouper, dropna=False
    ).transform("min")
    df["reference_missing"] = df["reference_energy"].isna()

    df["energy_excess"] = np.nan
    comparable = usable & ~df["reference_missing"]
    df.loc[comparable, "energy_excess"] = (
        energy_when_valid[comparable] - df.loc[comparable, "reference_energy"]
    )

    if "num_windows" in df.columns:
        distinct_windows = (
            df["num_windows"].where(usable).groupby(grouper, dropna=False).transform("nunique")
        )
        df["energy_scale_mismatch"] = distinct_windows > 1
    else:
        df["energy_scale_mismatch"] = False
    return df


# Two valid runs of one configuration can land on energies that differ by
# less than this and still be "the same answer" — floating-point summation
# over windows is not bit-reproducible.
_ENERGY_TIE_TOL = 1e-9


def _separation(valid_energies: np.ndarray, invalid_energies: np.ndarray) -> Dict[str, Any]:
    """How cleanly a configuration's energy separates its valid runs from its
    invalid ones.

    This is the sharp penalty-balance test. An invalid run's energy is a
    perfectly well-defined evaluation of the same Hamiltonian — it is just
    the energy of a bitstring that violates a constraint. If any such
    bitstring scores *lower* than a valid one, the penalty enforcing that
    constraint is provably too weak (the K_crash <= K_adj relationship in
    config.yaml is exactly this kind of balance). No calibration judgment is
    needed to read it: one inversion is one counterexample.

    separation_auc = P(invalid energy > valid energy), ties counted as 1/2.
    1.0 is perfect separation, 0.5 is energy telling you nothing about
    validity, below 0.5 means energy is actively misleading.
    """
    if valid_energies.size == 0 or invalid_energies.size == 0:
        return {
            "separation_auc": np.nan,
            "n_inversions": np.nan,
            "separated": None,
        }
    difference = invalid_energies[:, None] - valid_energies[None, :]
    n_inversions = int((difference < -_ENERGY_TIE_TOL).sum())
    n_ties = int((np.abs(difference) <= _ENERGY_TIE_TOL).sum())
    n_above = int((difference > _ENERGY_TIE_TOL).sum())
    return {
        "separation_auc": (n_above + 0.5 * n_ties) / difference.size,
        "n_inversions": n_inversions,
        # bool(), not the numpy scalar: `separated is False` is how the
        # caller tests it, and np.False_ is not False.
        "separated": bool(n_inversions == 0),
    }


def _energy_quality_correlation(
    energies: np.ndarray, efficiencies: np.ndarray
) -> Dict[str, Any]:
    """Spearman rank correlation between energy and path efficiency across a
    configuration's valid runs.

    The direct statement of "better energy should mean a better response":
    lower energy is better and higher efficiency is better, so a well-balanced
    QUBO gives a *negative* rho. Around zero means the energy landscape is not
    tracking solution quality and the penalty set needs recalibrating.

    Rank-based on purpose — it is invariant to the arbitrary additive constant
    and to normalize_scale, both of which make the energy's magnitude
    uninterpretable but leave its ordering intact.
    """
    usable = np.isfinite(energies) & np.isfinite(efficiencies)
    energies, efficiencies = energies[usable], efficiencies[usable]
    if energies.size < 3:
        return {"quality_rho": np.nan, "quality_p": np.nan, "quality_n": int(energies.size)}
    # spearmanr returns NaN (with a warning) if either input is constant, which
    # happens whenever a solver hits the same path on every run
    if np.ptp(energies) == 0 or np.ptp(efficiencies) == 0:
        return {"quality_rho": np.nan, "quality_p": np.nan, "quality_n": int(energies.size)}
    rho, p_value = stats.spearmanr(energies, efficiencies)
    return {
        "quality_rho": float(rho),
        "quality_p": float(p_value),
        "quality_n": int(energies.size),
    }


def compute_energy_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per energy class — (instance_map, problem_name, solver_name,
    penalty_set, preprocess) — describing what that configuration's energies
    actually tell you.

    Built for stochastic backends. A single run's energy says very little;
    what matters for D-Wave or QAOA is the shape of the distribution over
    repeated runs on the same instance, and whether that distribution lines
    up with solution quality:

      n_energy_runs     valid runs that produced a solver energy. Reads 0
                        when BFS solved everything, even though those runs are
                        valid — this table is about energy, not success.
      best_energy, median_energy, energy_iqr, max_excess
                        the spread. A tight distribution means the sampler
                        lands on the same basin repeatedly; a long tail means
                        it doesn't, which is a num_reads / layers / opt_steps
                        problem rather than a formulation one.
      hit_rate_best     fraction of valid runs that reach the best energy
                        found. The direct "how often does it actually work"
                        number for a stochastic solver.
      separation_auc, n_inversions, separated
                        does energy separate valid runs from invalid ones —
                        see _separation(). A formulation problem.
      quality_rho, quality_p, quality_n
                        does energy rank valid solutions by path quality —
                        see _energy_quality_correlation(). Also a formulation
                        problem, and the one that says whether "lower energy"
                        means anything at all.
      windows_consistent
                        False when runs used differing window counts, which
                        makes even the within-configuration energy comparison
                        unsound (see compute_energy_excess).

    All magnitude columns are in the solver's own raw energy units and must
    not be compared between configurations.
    """
    keys = _energy_class_keys(df)
    solver_ran = _solver_ran_mask(df)
    valid = _valid_mask(df) & solver_ran
    energy = pd.to_numeric(df["energy"], errors="coerce")
    reduction = (
        pd.to_numeric(df["average_reduction_ratio"], errors="coerce")
        if "average_reduction_ratio" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    efficiency = (
        pd.to_numeric(df["avg_path_efficiency"], errors="coerce")
        if "avg_path_efficiency" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    rows: List[Dict[str, Any]] = []
    for group_key, group in df.groupby(keys, dropna=False, sort=False):
        idx = group.index
        is_valid = valid.loc[idx]
        valid_energy = energy.loc[idx][is_valid].dropna().to_numpy(dtype=float)
        invalid_energy = energy.loc[idx][~is_valid].dropna().to_numpy(dtype=float)

        row: Dict[str, Any] = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,)))
        row.update(
            {
                "n_runs": int(len(group)),
                # valid runs whose solver actually ran — NOT the same as the
                # success rate. A run can have a perfectly valid path from BFS
                # alone and contribute no energy at all, so this column can
                # read 0 while compute_success_rate() reports 100%.
                "n_energy_runs": int(is_valid.sum()),
                "n_invalid": int((~is_valid).sum()),
                "n_solver_skipped": int((~solver_ran.loc[idx]).sum()),
                "median_reduction_ratio": (
                    float(reduction.loc[idx].median())
                    if reduction.loc[idx].notna().any()
                    else np.nan
                ),
            }
        )

        if valid_energy.size:
            best = float(valid_energy.min())
            excess = valid_energy - best
            row.update(
                {
                    "best_energy": best,
                    "median_energy": float(np.median(valid_energy)),
                    "energy_iqr": float(
                        np.percentile(valid_energy, 75) - np.percentile(valid_energy, 25)
                    ),
                    "median_excess": float(np.median(excess)),
                    "max_excess": float(excess.max()),
                    "hit_rate_best": float((excess <= _ENERGY_TIE_TOL).mean()),
                }
            )
        else:
            row.update(
                {
                    "best_energy": np.nan,
                    "median_energy": np.nan,
                    "energy_iqr": np.nan,
                    "median_excess": np.nan,
                    "max_excess": np.nan,
                    "hit_rate_best": np.nan,
                }
            )

        row.update(_separation(valid_energy, invalid_energy))
        row.update(
            _energy_quality_correlation(
                energy.loc[idx][is_valid].to_numpy(dtype=float),
                efficiency.loc[idx][is_valid].to_numpy(dtype=float),
            )
        )

        if "num_windows" in group.columns:
            row["windows_consistent"] = bool(
                group.loc[is_valid[is_valid].index, "num_windows"].nunique(dropna=True) <= 1
            )
        else:
            row["windows_consistent"] = True

        notes = []
        if row["n_energy_runs"] == 0:
            notes.append(
                "no_energy_runs (no valid run produced a solver energy — this "
                "says nothing about the success rate, see compute_success_rate)"
            )
        if row["n_solver_skipped"]:
            notes.append(
                f"{row['n_solver_skipped']} run(s) were fully pre-processed — the "
                "solver never ran and their energy of 0.0 is a sentinel, not a "
                "measurement; they are excluded from every column here"
            )
        if pd.notna(row["median_reduction_ratio"]) and row["median_reduction_ratio"] >= 0.9:
            notes.append(
                f"pre-processing removed a median {row['median_reduction_ratio']:.1%} "
                "of variables — this is mostly measuring BFS, not the solver; "
                "run the preprocess ablation to separate them"
            )
        if not row["windows_consistent"]:
            notes.append(
                "window_count_varies (energies carry different dropped constants "
                "and are not comparable even within this configuration)"
            )
        if row["separated"] is False:
            notes.append(
                f"{row['n_inversions']} invalid run(s) score below a valid one — "
                "the penalty for the violated constraint is too weak"
            )
        if row["quality_n"] >= 3 and pd.notna(row["quality_rho"]) and row["quality_rho"] > 0:
            notes.append(
                "energy is positively correlated with path efficiency — lower "
                "energy is selecting *worse* paths"
            )
        row["note"] = "; ".join(notes) or None
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=keys
        + [
            "n_runs",
            "n_energy_runs",
            "n_invalid",
            "n_solver_skipped",
            "median_reduction_ratio",
            "best_energy",
            "median_energy",
            "energy_iqr",
            "median_excess",
            "max_excess",
            "hit_rate_best",
            "separation_auc",
            "n_inversions",
            "separated",
            "quality_rho",
            "quality_p",
            "quality_n",
            "windows_consistent",
            "note",
        ],
    )


def compute_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(valid=_valid_mask(df))
        .groupby(["instance_map", "problem_name", "solver_name", "preprocess"])["valid"]
        .mean()
        .rename("success_rate")
        .reset_index()
    )


def compute_failure_causes(df: pd.DataFrame) -> pd.DataFrame:
    """Invalid runs per (instance, solver, preprocess), split by what caused
    them — the first question to ask of any failing cell.

    pre_processing means BFS or diagonal fixing pinned the robots into a
    conflict before the solver was invoked; no amount of solver tuning or
    penalty reweighting can recover that, because the variables needed to
    avoid it were already removed. solver_sampling means the solver returned
    a bitstring that violates a penalty that was actually present, which is a
    sampler or penalty-weight problem.

    Empty (but correctly shaped) if the sweep recorded no invalid runs."""
    columns = [
        "instance_map",
        "problem_name",
        "solver_name",
        "preprocess",
        "invalid_cause",
        "n_runs",
    ]
    if "invalid_cause" not in df.columns:
        return pd.DataFrame(columns=columns)
    invalid = df[~_valid_mask(df)]
    if invalid.empty:
        return pd.DataFrame(columns=columns)
    return (
        invalid.assign(
            invalid_cause=invalid["invalid_cause"].fillna("not_recorded")
        )
        .groupby(
            ["instance_map", "problem_name", "solver_name", "preprocess", "invalid_cause"],
            dropna=False,
        )
        .size()
        .rename("n_runs")
        .reset_index()
        .sort_values("n_runs", ascending=False)
        .reset_index(drop=True)
    )


def compute_variable_reduction_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Groups the already-computed average_reduction_ratio column — no new
    computation, BenchmarkRunner already aggregates this per run from each
    solver's own window_stats/bfs_stats."""
    return (
        df.groupby(["instance_map", "problem_name", "solver_name", "preprocess"])
        .agg(
            mean_reduction_ratio=("average_reduction_ratio", "mean"),
            mean_initial_variables=("total_initial_variables", "mean"),
            mean_final_variables=("total_final_variables", "mean"),
        )
        .reset_index()
    )


# termination_condition values that mean execution_time_sec is a *censored*
# lower bound — the solver was cut off, not finished. CBS emits the first
# two (quantum.solvers.cbs_algorithm), Pyomo the rest via ILP_solver.
_CENSORED_TERMINATIONS = frozenset(
    {
        "time_limit_exceeded",
        "node_limit_exceeded",
        "maxTimeLimit",
        "maxIterations",
        "maxEvaluations",
    }
)

# Per-metric handling for the paired test.
#   transform="log": test log(a) - log(b) instead of a - b. Wilcoxon ranks
#     *absolute* differences, so on a sweep spanning 0.1s to 600s the raw
#     ranking is decided entirely by the biggest instances — a 2x speedup on
#     a small instance ranks below a 5% wobble on a large one. Testing the
#     log ratio puts every instance on equal footing and makes the test's
#     symmetry assumption far more plausible.
#   higher_is_better: which direction of the effect size counts as a win.
#   censorable: whether a cut-off run's value is a censored lower bound.
_METRIC_SPECS = {
    "execution_time_sec": {
        "transform": "log",
        "higher_is_better": False,
        "censorable": True,
    },
    "build_time_sec": {
        "transform": "log",
        "higher_is_better": False,
        "censorable": False,
    },
    "avg_path_efficiency": {
        "transform": None,
        "higher_is_better": True,
        "censorable": False,
    },
    "min_path_efficiency": {
        "transform": None,
        "higher_is_better": True,
        "censorable": False,
    },
    "robot_success_rate": {
        "transform": None,
        "higher_is_better": True,
        "censorable": False,
    },
}
_DEFAULT_METRIC_SPEC = {
    "transform": None,
    "higher_is_better": False,
    "censorable": False,
}

# Metrics that only mean something relative to other runs of the same solver
# (see compute_energy_excess). A cross-solver paired test on these is refused
# rather than computed-and-reported, because the resulting p-value would look
# exactly like a real one.
_WITHIN_SOLVER_ONLY_METRICS = frozenset(
    # energy_gap and optimality_gap are the names this column had before it
    # became energy_excess. They are kept here so a df loaded from an older
    # runs_long.csv is refused rather than quietly tested.
    {"energy", "energy_excess", "energy_gap", "reference_energy", "optimality_gap"}
)

_RESULT_COLUMNS = [
    "solver_a",
    "solver_b",
    "metric",
    "transform",
    "n_pairs",
    "n_effective",
    "n_dropped_unshared",
    "n_dropped_invalid",
    "n_dropped_no_metric",
    "n_censored",
    "min_valid_runs",
    "median_diff",
    "median_ratio",
    "rank_biserial",
    "favors",
    "statistic",
    "p_value",
    "p_value_bh",
    "note",
]


def _default_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """(candidate, baseline) pairs, in that order.

    The order is load-bearing: every directional column in the result
    (median_diff, rank_biserial, favors) is defined as solver_a relative to
    solver_b, so the baseline has to stay on the b side. Sorting each pair
    alphabetically instead would make "which one is the baseline"
    unrecoverable from the output."""
    solvers = sorted(df["solver_name"].unique())
    backends = set(df["backend"].unique())
    baselines = [s for s in ("ilp", "cbs") if s in backends]
    baseline_names = sorted(
        df[df["backend"].isin(baselines)]["solver_name"].unique().tolist()
    )

    pairs: List[Tuple[str, str]] = []
    seen = set()
    for baseline in baseline_names:
        for candidate in solvers:
            key = frozenset((candidate, baseline))
            if candidate == baseline or key in seen:
                continue
            seen.add(key)
            pairs.append((candidate, baseline))
    return pairs


def _rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: (W+ - W-) / (W+ + W-).

    Bounded in [-1, 1], positive when solver_a's values are the larger ones.
    This is the effect size that goes with Wilcoxon the way Cohen's d goes
    with a t-test — the p-value alone says only "some difference exists"."""
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = ranks[nonzero > 0].sum()
    w_minus = ranks[nonzero < 0].sum()
    total = w_plus + w_minus
    if total == 0:
        return float("nan")
    return float((w_plus - w_minus) / total)


def run_statistical_tests(
    df: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
    metrics: Tuple[str, ...] = ("execution_time_sec", "avg_path_efficiency"),
) -> pd.DataFrame:
    """Wilcoxon signed-rank test per (solver_a, solver_b, metric), with an
    effect size, explicit drop accounting, and Benjamini-Hochberg FDR
    correction across the whole family of tests run in this call.

    Pairing
    -------
    Pairs *by instance*, not by raw run: for each solver, takes its
    per-instance mean of the metric across its valid runs, then tests the
    paired per-instance values. Raw per-run values from a stochastic solver
    on one instance are not independent draws suitable for a cross-solver
    paired test.

    Direction is preserved: every directional column is solver_a relative to
    solver_b, and the default pairs are (candidate, baseline).

    Survivorship accounting
    -----------------------
    The paired population is conditioned on *both* solvers having produced a
    usable value, so it silently shifts from pair to pair: a solver that only
    succeeds on easy instances gets compared only on easy instances and looks
    both fast and accurate. That bias cannot be removed here — it is a
    property of what a paired test is — so it is instead made visible:

      n_dropped_unshared   only one solver attempted the instance at all
      n_dropped_invalid    both attempted, at least one had no valid run
      n_dropped_no_metric  both had valid runs, the metric wasn't recorded
                           (e.g. path efficiency needs BenchmarkRunner
                           level>=2)
      min_valid_runs       fewest valid runs behind any single paired cell —
                           1 here means some instance is represented by one
                           lucky run out of many attempts

    Read those alongside compute_success_rate() before believing any row.

    Effective sample size
    ---------------------
    scipy's default zero_method="wilcox" discards zero differences, so the
    test's real n is n_effective, not n_pairs — an instance where both
    solvers tie contributes nothing. Both are reported, and the minimum-pairs
    guard is applied to n_effective as well as n_pairs.

    Censoring
    ---------
    n_censored counts paired instances where either solver hit a time/node
    limit. Those execution times are lower bounds, so a test over them
    understates the true difference.

    Metrics
    -------
    Metrics in _WITHIN_SOLVER_ONLY_METRICS (raw energy and anything derived
    from it) are refused: see compute_energy_excess() for why an energy is not
    comparable between two solvers.

    p-values
    --------
    p_value is raw; p_value_bh is Benjamini-Hochberg-adjusted over every row
    in this result that has a real p-value. Use p_value_bh for any "is this
    significant" claim.
    """
    metrics = tuple(metrics)
    if pairs is None:
        pairs = _default_pairs(df)

    valid = _valid_mask(df)
    group_keys = ["instance_map", "problem_name", "solver_name"]

    # Every instance x solver table is built ONCE here. The previous version
    # rebuilt the groupby inside the pair x metric loop, recomputing the same
    # aggregation len(pairs) * len(metrics) times.
    attempted = df.groupby(group_keys).size().unstack("solver_name")
    index = attempted.index

    valid_df = df[valid]
    valid_counts = (
        valid_df.groupby(group_keys).size().unstack("solver_name").reindex(index)
    )

    if "termination_condition" in df.columns:
        censored = (
            df.assign(_censored=df["termination_condition"].isin(_CENSORED_TERMINATIONS))
            .groupby(group_keys)["_censored"]
            .any()
            .unstack("solver_name")
            .reindex(index)
        )
    else:
        censored = pd.DataFrame(False, index=index, columns=attempted.columns)

    values: Dict[str, pd.DataFrame] = {}
    metric_runs: Dict[str, pd.DataFrame] = {}
    for metric in metrics:
        if metric not in df.columns or metric in _WITHIN_SOLVER_ONLY_METRICS:
            continue
        grouped = valid_df.groupby(group_keys)[metric]
        values[metric] = grouped.mean().unstack("solver_name").reindex(index)
        metric_runs[metric] = grouped.count().unstack("solver_name").reindex(index)

    def _column(table: pd.DataFrame, solver: str, fill) -> pd.Series:
        if table is None or solver not in table.columns:
            return pd.Series(fill, index=index)
        return table[solver].reindex(index)

    rows: List[Dict[str, Any]] = []
    for solver_a, solver_b in pairs:
        for metric in metrics:
            spec = _METRIC_SPECS.get(metric, _DEFAULT_METRIC_SPEC)
            row: Dict[str, Any] = {
                "solver_a": solver_a,
                "solver_b": solver_b,
                "metric": metric,
                "transform": spec["transform"],
            }

            if metric in _WITHIN_SOLVER_ONLY_METRICS:
                rows.append(
                    {
                        **row,
                        "note": (
                            f"not_comparable_across_solvers ('{metric}' is only "
                            "meaningful against other runs of the same solver — "
                            "see compute_energy_excess)"
                        ),
                    }
                )
                continue
            if metric not in values:
                rows.append({**row, "note": f"metric_missing ('{metric}' not in df)"})
                continue

            table = values[metric]
            a_attempted = _column(attempted, solver_a, np.nan).notna()
            b_attempted = _column(attempted, solver_b, np.nan).notna()
            if not a_attempted.any() or not b_attempted.any():
                continue  # one of the two solvers isn't in this sweep at all

            a_valid_runs = _column(valid_counts, solver_a, np.nan).fillna(0)
            b_valid_runs = _column(valid_counts, solver_b, np.nan).fillna(0)
            va = _column(table, solver_a, np.nan)
            vb = _column(table, solver_b, np.nan)

            both_attempted = a_attempted & b_attempted
            both_have_valid = both_attempted & (a_valid_runs > 0) & (b_valid_runs > 0)
            paired = va.notna() & vb.notna()

            row.update(
                {
                    "n_pairs": int(paired.sum()),
                    "n_dropped_unshared": int(
                        ((a_attempted | b_attempted) & ~both_attempted).sum()
                    ),
                    "n_dropped_invalid": int(
                        (both_attempted & ~both_have_valid).sum()
                    ),
                    "n_dropped_no_metric": int((both_have_valid & ~paired).sum()),
                }
            )

            runs_a = _column(metric_runs[metric], solver_a, np.nan)[paired]
            runs_b = _column(metric_runs[metric], solver_b, np.nan)[paired]
            row["min_valid_runs"] = (
                int(min(runs_a.min(), runs_b.min())) if row["n_pairs"] else 0
            )
            if spec["censorable"]:
                # .eq(True) rather than .fillna(False).astype(bool): an
                # instance a solver never attempted is NaN here, and fillna on
                # an object column is a deprecated downcast.
                ca = _column(censored, solver_a, False).eq(True)
                cb = _column(censored, solver_b, False).eq(True)
                row["n_censored"] = int((paired & (ca | cb)).sum())
            else:
                row["n_censored"] = 0

            if row["n_pairs"] < _MIN_PAIRS_FOR_WILCOXON:
                rows.append(
                    {
                        **row,
                        "note": (
                            f"insufficient_data (need >= {_MIN_PAIRS_FOR_WILCOXON} "
                            "paired instances)"
                        ),
                    }
                )
                continue

            xa = va[paired].to_numpy(dtype=float)
            xb = vb[paired].to_numpy(dtype=float)
            if spec["transform"] == "log":
                if (xa <= 0).any() or (xb <= 0).any():
                    rows.append(
                        {
                            **row,
                            "note": (
                                "log_transform_invalid (non-positive values — a "
                                "recorded time of 0 means the timer resolution is "
                                "too coarse for this instance)"
                            ),
                        }
                    )
                    continue
                ya, yb = np.log(xa), np.log(xb)
            else:
                ya, yb = xa, xb

            differences = ya - yb
            n_effective = int(np.count_nonzero(differences))
            median_diff = float(np.median(differences))
            rank_biserial = _rank_biserial(differences)

            row.update(
                {
                    "n_effective": n_effective,
                    "median_diff": median_diff,
                    # median of the log differences exponentiated = the median
                    # a/b ratio, i.e. "solver_a takes N x as long as solver_b"
                    "median_ratio": (
                        float(np.exp(median_diff))
                        if spec["transform"] == "log"
                        else np.nan
                    ),
                    "rank_biserial": rank_biserial,
                    "favors": (
                        None
                        if not np.isfinite(rank_biserial) or rank_biserial == 0
                        else (
                            solver_a
                            if (rank_biserial > 0) == spec["higher_is_better"]
                            else solver_b
                        )
                    ),
                }
            )

            if n_effective < _MIN_PAIRS_FOR_WILCOXON:
                rows.append(
                    {
                        **row,
                        "note": (
                            f"insufficient_effective_pairs ({row['n_pairs'] - n_effective} "
                            f"of {row['n_pairs']} paired differences are exactly zero and "
                            "are discarded by zero_method='wilcox'; effect size above is "
                            "still computed over all pairs)"
                        ),
                    }
                )
                continue

            # No try/except here on purpose: the all-zero-differences case that
            # the old bare `except ValueError` was documenting is now caught by
            # the n_effective guard above, and every other ValueError scipy can
            # raise is a real bug that should not be reported as p = 1.0.
            statistic, p_value = stats.wilcoxon(ya, yb)
            rows.append(
                {
                    **row,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "note": (
                        None
                        if not row["n_censored"]
                        else (
                            f"{row['n_censored']} of {row['n_pairs']} paired instances "
                            "hit a time/node limit; those times are censored lower "
                            "bounds and the true difference is larger than reported"
                        )
                    ),
                }
            )

    result = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    computed = result["p_value"].notna()
    if computed.any():
        result.loc[computed, "p_value_bh"] = false_discovery_control(
            result.loc[computed, "p_value"].to_numpy(), method="bh"
        )
    return result


def aggregate_sweep(
    sweep_dir: str, output_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Runs the full aggregation pipeline and writes CSVs to
    <sweep_dir>/analysis/ (or output_dir if given). Returns the DataFrames
    too, so callers (CLI, future FastAPI endpoint) can use either the
    files or the in-memory result."""
    sweep_dir = Path(sweep_dir)
    out = Path(output_dir) if output_dir else sweep_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    runs_long = load_sweep(sweep_dir)
    if runs_long.empty:
        raise ValueError(f"No completed runs found in {sweep_dir}/index.json")

    runs_long = compute_energy_excess(runs_long)
    summary_by_solver = compute_success_rate(runs_long).merge(
        compute_variable_reduction_stats(runs_long),
        on=["instance_map", "problem_name", "solver_name", "preprocess"],
    )
    statistical_tests = run_statistical_tests(runs_long)
    energy_diagnostics = compute_energy_diagnostics(runs_long)
    failure_causes = compute_failure_causes(runs_long)
    robot_statistics_long = load_robot_statistics(sweep_dir)

    runs_long.to_csv(out / "runs_long.csv", index=False)
    summary_by_solver.to_csv(out / "summary_by_solver.csv", index=False)
    statistical_tests.to_csv(out / "statistical_tests.csv", index=False)
    energy_diagnostics.to_csv(out / "energy_diagnostics.csv", index=False)
    failure_causes.to_csv(out / "failure_causes.csv", index=False)
    robot_statistics_long.to_csv(out / "robot_statistics_long.csv", index=False)

    return {
        "runs_long": runs_long,
        "summary_by_solver": summary_by_solver,
        "statistical_tests": statistical_tests,
        "energy_diagnostics": energy_diagnostics,
        "failure_causes": failure_causes,
        "robot_statistics_long": robot_statistics_long,
    }
