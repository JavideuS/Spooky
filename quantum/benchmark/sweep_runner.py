"""
Runs a matrix of (instance x solver x ablation) benchmark configs defined
in a YAML sweep config, driving the existing BenchmarkRunner unchanged for
each combination — see quantum/qubo_cli.py's main() for the canonical
problem -> builder -> solver -> runner wiring this replicates per config
entry. Produces a reproducibility manifest (manifest.py) and an index of
every benchmark JSON it produced, so Phase 3's aggregation reads a known
list instead of globbing results/benchmarks/ (which also holds unrelated
ad hoc runs).
"""

import json
import contextlib
import signal
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import quantum.config.parser as config_parser
from quantum.utils.paths import RESULTS_DIR
from quantum.pathFormulation import PathfindingProblem
from quantum.solvers import SolverFactory
from quantum.builder import (
    QUBOBuilder,
    GraphQUBO,
    GridILPBuilder,
    GraphILPBuilder,
    GridCBSBuilder,
    GraphCBSBuilder,
)
from quantum.benchmark.benchmark import BenchmarkRunner
from quantum.benchmark.manifest import (
    build_manifest,
    checkpoint_manifest,
    finalize_manifest,
    record_resume_event,
)
from quantum.utils.logger import get_logger, set_verbose_level
from quantum.utils import preprocess as preprocess_modes

# Two independent flags must both be set before any hardware-gated solver
# entry runs — see SweepRunner.__init__. Deliberately not a single boolean:
# makes it hard to trigger a quota-costing real-hardware run by accident
# (stray flag, copy-pasted command, muscle memory from a previous invocation).
HARDWARE_CONFIRM_PHRASE = "yes-spend-quota"

# Sentinel passed as resume_id when the user ran --resume with no explicit
# id — load() resolves it to the latest incomplete sweep on disk before
# doing anything else. Not prefixed with an underscore despite being an
# internal implementation detail: run_sweep.py imports it directly across
# the module boundary (as the argparse `const` for bare --resume), so it's
# part of the real contract between the two files, same as
# HARDWARE_CONFIRM_PHRASE.
AUTO_RESUME = "__auto__"

# ILP/CBS have no penalty weights, var_limit, or windowing — mirrors
# qubo_cli.py's own `if args.solver == "ilp": ... builder_kwargs don't apply`
# branching.
_BUILDER_FREE_BACKENDS = {"ilp", "cbs"}

_CONFIG_YAML = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


class SweepConfigError(ValueError):
    """Raised by SweepRunner.load() on any config problem — validation
    happens entirely before any compute, so a typo doesn't surface 20
    minutes into a sweep."""


class RunTimeout(Exception):
    """A single (instance x solver x ablation) run exceeded run_timeout_sec."""


# How often to re-raise after the initial deadline. A single one-shot alarm
# gets exactly one chance to land, and it can land somewhere the exception is
# discarded -- observed in a sweep where SIGALRM arrived inside a garbage
# collection callback ("Exception ignored in: <function _xla_gc_callback>"),
# so the run carried on past its deadline and had to be killed by hand.
_TIMEOUT_RETRY_SECONDS = 5


@contextlib.contextmanager
def _run_timeout(seconds):
    """Abort a run that overruns, so one hung combination cannot block a sweep.

    Uses a repeating SIGALRM. Signals are only delivered between Python
    bytecodes, so a call already inside a C extension (neal's sampler,
    lightning's statevector) is not interrupted until it returns -- and even
    once delivered, an exception raised in a context that swallows it (a GC
    callback, a __del__) is lost. Re-arming every _TIMEOUT_RETRY_SECONDS means
    a swallowed delivery is retried rather than being the only attempt.

    Still best-effort: a run can overshoot its deadline, and one that never
    returns to interpretable Python cannot be stopped this way at all.
    """
    if not seconds or not hasattr(signal, "setitimer"):
        yield
        return

    def _fire(signum, frame):
        raise RunTimeout(f"exceeded run_timeout_sec={seconds}")

    previous = signal.signal(signal.SIGALRM, _fire)
    # (initial delay, repeat interval) -- the repeat is what makes this
    # survive a swallowed delivery
    signal.setitimer(signal.ITIMER_REAL, float(seconds), _TIMEOUT_RETRY_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class SweepRunner:
    def __init__(
        self,
        config_path: str,
        enable_hardware: bool = False,
        confirm_hardware: str = "",
        dry_run: bool = False,
        only_instances: Optional[List[str]] = None,
        only_solvers: Optional[List[str]] = None,
        verbose_level: int = 2,
        resume_id: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        self.enable_hardware = (
            enable_hardware and confirm_hardware == HARDWARE_CONFIRM_PHRASE
        )
        self.dry_run = dry_run
        self.only_instances = set(only_instances) if only_instances else None
        self.only_solvers = set(only_solvers) if only_solvers else None
        self.logger = get_logger()
        set_verbose_level(verbose_level)

        # resume_id forces sweep_id instead of auto-generating a fresh one,
        # and switches _run_one() to skip any combo that already has a
        # completed benchmark JSON on disk under that sweep's output_dir.
        self.resume_id = resume_id
        self.resuming = resume_id is not None

        self.config: Dict[str, Any] = {}
        self.sweep_id = ""
        self.output_dir = (
            RESULTS_DIR / "sweeps"
        )  # replaced with output_root/<id> in load()
        self.index: List[Dict[str, Any]] = []
        self.manifest: Dict[str, Any] = {}
        self._penalty_sets: Dict[str, Any] = {}
        self.global_seed: Optional[int] = None

    @staticmethod
    def _find_latest_incomplete_sweep(
        output_root: Path, current_config: Dict[str, Any]
    ) -> Optional[str]:
        """Scans output_root for sweep directories whose manifest.json has
        end_time=null (interrupted or still running) AND whose stored
        sweep_config matches current_config exactly, returning the sweep_id
        of the one most recently checkpointed among those — or None if no
        matching candidates exist. Used by load() when resume_id==AUTO_RESUME.

        The config match is required, not optional: without it, bare
        --resume would happily grab the most recently-checkpointed
        incomplete sweep from a *different* config file (e.g. two separate
        sweeps interrupted around the same time) and silently run the
        current config's instances into that other sweep's directory —
        confirmed as a real failure mode, not just a theoretical one, before
        this check was added."""
        candidates = []
        for manifest_path in output_root.glob("*/manifest.json"):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("end_time") is not None:
                    continue  # already finished cleanly — skip
                if data.get("sweep_config") != current_config:
                    continue  # belongs to a different sweep config entirely
                sweep_id = data.get("sweep_id") or manifest_path.parent.name
                # Use last_checkpoint if available, else start_time, else mtime
                ts = data.get("last_checkpoint") or data.get("start_time")
                sort_key = ts or str(manifest_path.stat().st_mtime)
                candidates.append((sort_key, sweep_id))
            except Exception:
                continue  # corrupt/unreadable manifest — skip silently
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def load(self) -> "SweepRunner":
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        sweep_meta = self.config.get("sweep", {})
        # sweep lands in the same place wherever it's launched from.
        # Unset -> <repo>/results/sweeps.
        configured_root = sweep_meta.get("output_root")
        if configured_root:
            output_root = Path(configured_root)
            if not output_root.is_absolute():
                output_root = RESULTS_DIR.parent / output_root
        else:
            output_root = RESULTS_DIR / "sweeps"

        # Resolve auto-resume: scan output_root for the latest incomplete
        # sweep whose stored config matches this one — never a different
        # sweep's leftover directory, even if it was checkpointed more
        # recently (see _find_latest_incomplete_sweep's docstring).
        if self.resume_id == AUTO_RESUME:
            found = self._find_latest_incomplete_sweep(output_root, self.config)
            if found:
                self.resume_id = found
                self.logger.minimal(
                    f"[sweep] --resume (auto): found incomplete sweep '{found}' "
                    f"matching this config"
                )
            else:
                self.resume_id = None
                self.resuming = False
                self.logger.minimal(
                    "[sweep] --resume (auto): no incomplete sweep matching this "
                    f"config found under '{output_root}' — starting a fresh sweep."
                )

        self.sweep_id = (
            self.resume_id
            or sweep_meta.get("id")
            or (f"sweep_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}")
        )
        self.output_dir = output_root / self.sweep_id
        self.global_seed = sweep_meta.get("seed")

        instances = self.config.get("instances", [])
        solvers = self.config.get("solvers", [])
        if not instances:
            raise SweepConfigError("Sweep config has no 'instances' entries.")
        if not solvers:
            raise SweepConfigError("Sweep config has no 'solvers' entries.")

        available_solvers = set(SolverFactory.get_available_solvers())
        self._penalty_sets = (
            config_parser.load_config(str(_CONFIG_YAML), sections=["penalty_sets"]).get(
                "penalty_sets", {}
            )
            or {}
        )

        for instance in instances:
            map_path = instance.get("map")
            if not map_path:
                raise SweepConfigError(f"Instance entry missing 'map': {instance}")
            problems = instance.get("problems", [])
            if not problems:
                raise SweepConfigError(
                    f"Instance '{map_path}' has no 'problems' listed."
                )
            yaml_path = f"{map_path}.yaml"
            if not Path(yaml_path).exists():
                raise SweepConfigError(f"Map YAML not found: {yaml_path}")
            defined = (
                config_parser.load_config(yaml_path, sections=["problems"]).get(
                    "problems", {}
                )
                or {}
            )
            for problem_name in problems:
                if problem_name not in defined:
                    raise SweepConfigError(
                        f"Problem '{problem_name}' not found in {yaml_path}. "
                        f"Available: {list(defined.keys())}"
                    )
            builder_kind = instance.get("builder", "grid")
            if builder_kind not in ("grid", "graph"):
                raise SweepConfigError(
                    f"Instance '{map_path}': builder must be 'grid' or 'graph', got '{builder_kind}'"
                )

        for solver_cfg in solvers:
            name = solver_cfg.get("name", "<unnamed>")
            backend = solver_cfg.get("backend")
            if backend not in available_solvers:
                raise SweepConfigError(
                    f"Solver entry '{name}': backend '{backend}' not registered. "
                    f"Available: {sorted(available_solvers)}"
                )
            if backend not in _BUILDER_FREE_BACKENDS:
                pset = solver_cfg.get("penalty_set")
                if pset and pset not in self._penalty_sets:
                    raise SweepConfigError(
                        f"Solver entry '{name}': penalty_set '{pset}' not found in "
                        f"{_CONFIG_YAML}. Available: {sorted(self._penalty_sets)}"
                    )

        manifest_path = self.output_dir / "manifest.json"
        if self.resuming and manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            self.manifest = record_resume_event(self.manifest)
            self.logger.minimal(
                f"[sweep] Resuming {self.sweep_id} — "
                f"{len(self.manifest.get('resume_events', []))} prior resume(s)."
            )
        else:
            if self.resuming:
                self.logger.minimal(
                    f"[sweep] --resume {self.sweep_id} given but no existing manifest found "
                    f"at {manifest_path} — starting fresh under that id."
                )
            self.manifest = build_manifest(self.config, self.sweep_id)
        return self

    def _build_problem_and_builder(
        self, map_path, problem_name, builder_kind, backend, solver_cfg
    ):
        problem = PathfindingProblem.from_map_config(map_path, problem_name)

        if backend in _BUILDER_FREE_BACKENDS:
            builder_cls = {
                ("ilp", "grid"): GridILPBuilder,
                ("ilp", "graph"): GraphILPBuilder,
                ("cbs", "grid"): GridCBSBuilder,
                ("cbs", "graph"): GraphCBSBuilder,
            }[(backend, builder_kind)]
            p = (
                problem.as_grid_only()
                if builder_kind == "grid"
                else problem.as_graph_only()
            )
            return p, builder_cls(p, name=problem_name)

        penalties = dict(self._penalty_sets[solver_cfg["penalty_set"]])
        penalties.setdefault("name", solver_cfg["penalty_set"])
        builder_kwargs = {"penalties": penalties, "name": problem_name}
        if "var_limit" in solver_cfg:
            builder_kwargs["var_limit"] = solver_cfg["var_limit"]
        if builder_kind == "grid":
            p = problem.as_grid_only()
            return p, QUBOBuilder(p, **builder_kwargs)
        p = problem.as_graph_only()
        return p, GraphQUBO(p, **builder_kwargs)

    def run(self) -> List[Dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._persist()

        if self.global_seed is not None:
            # Seeds numpy's global RNG once so anything drawn at
            # problem/builder/solver construction is reproducible. Per-run
            # solver randomness (QAOA angles, neal seed) is handled
            # separately: BenchmarkRunner re-seeds from global_seed + run_id
            # each run (see BenchmarkRunner._reseed_run), so the num_runs
            # loop actually varies instead of replaying one fixed init.
            import numpy as np

            np.random.seed(self.global_seed)

        execution = self.config.get("execution", {})
        self.run_timeout_sec = execution.get("run_timeout_sec")
        preprocess_default = preprocess_modes.normalize(
            execution.get("preprocess_default", True)
        )
        fail_fast = execution.get("fail_fast", False)

        for instance in self.config["instances"]:
            map_path = instance["map"]
            if self.only_instances and map_path not in self.only_instances:
                continue
            builder_kind = instance.get("builder", "grid")

            for problem_name in instance["problems"]:
                for solver_cfg in self.config["solvers"]:
                    solver_name = solver_cfg["name"]
                    if self.only_solvers and solver_name not in self.only_solvers:
                        continue

                    if solver_cfg.get("hardware", False) and not self.enable_hardware:
                        self.manifest["skipped"].append(
                            {
                                "instance": map_path,
                                "problem": problem_name,
                                "solver": solver_name,
                                "reason": "hardware_gated_not_enabled",
                            }
                        )
                        self._persist()
                        continue

                    ablation = solver_cfg.get("ablation", {})
                    # ablation entries may be mode strings or legacy bools;
                    # normalize so index.json records one vocabulary
                    preprocess_values = [
                        preprocess_modes.normalize(v)
                        for v in ablation.get("preprocess", [preprocess_default])
                    ]
                    num_runs = solver_cfg.get("num_runs", 1)

                    for preprocess in preprocess_values:
                        self._run_one(
                            map_path,
                            problem_name,
                            builder_kind,
                            solver_cfg,
                            preprocess,
                            num_runs,
                            fail_fast,
                        )

        # The loop reached its natural end (not a fail_fast exception or a
        # process kill) — this is what "the sweep finished" actually means,
        # so end_time gets set exactly here, nowhere else. Every other
        # _persist() call during the run only touches last_checkpoint.
        self.manifest = finalize_manifest(self.manifest)
        self._persist()
        return self.index

    def _run_one(
        self,
        map_path,
        problem_name,
        builder_kind,
        solver_cfg,
        preprocess,
        num_runs,
        fail_fast,
    ):
        solver_name = solver_cfg["name"]
        instance_slug = map_path.replace("/", "_")
        run_dir = (
            self.output_dir
            / f"{instance_slug}__{problem_name}__{solver_name}__preprocess_{preprocess}"
        )

        device = solver_cfg.get("params", {}).get("device")
        penalty_set = solver_cfg.get("penalty_set")

        if self.dry_run:
            self.index.append(
                {
                    "instance": map_path,
                    "problem": problem_name,
                    "solver": solver_name,
                    "backend": solver_cfg["backend"],
                    "device": device,
                    "penalty_set": penalty_set,
                    "preprocess": preprocess,
                    "num_runs": num_runs,
                    "output_dir": str(run_dir),
                    "benchmark_json": None,
                    "dry_run": True,
                }
            )
            return

        # Resume support: a combo is "done" if run_dir already holds a
        # completed benchmark JSON from a prior (possibly interrupted) run
        # of this same sweep_id — checked against the directory on disk,
        # not a loaded index.json, so it self-heals even if the index was
        # itself lost/corrupted (the deterministic run_dir naming is the
        # real source of truth). A combo that crashed mid-solve leaves no
        # JSON here and is correctly retried, not skipped.
        if self.resuming and run_dir.exists():
            existing = sorted(run_dir.glob("benchmark_*.json"))
            if existing:
                self.index.append(
                    {
                        "instance": map_path,
                        "problem": problem_name,
                        "solver": solver_name,
                        "backend": solver_cfg["backend"],
                        "device": device,
                        "penalty_set": penalty_set,
                        "preprocess": preprocess,
                        "num_runs": num_runs,
                        "output_dir": str(run_dir),
                        "benchmark_json": str(existing[-1]),
                        "resumed": True,
                    }
                )
                self.logger.minimal(
                    f"[sweep] SKIP (already done): {instance_slug}/{problem_name}/"
                    f"{solver_name}/preprocess_{preprocess}"
                )
                self._persist()
                return

        try:
            problem, builder = self._build_problem_and_builder(
                map_path, problem_name, builder_kind, solver_cfg["backend"], solver_cfg
            )
            params = dict(solver_cfg.get("params", {}))
            if (
                solver_cfg["backend"] == "dwave"
                and "seed" not in params
                and self.global_seed is not None
            ):
                params["seed"] = self.global_seed
            solver = SolverFactory.create_solver(solver_cfg["backend"], **params)

            runner = BenchmarkRunner(
                builder,
                solver,
                num_runs=num_runs,
                output_dir=str(run_dir),
                level=2,
                preprocess=preprocess,
                # Per-run reseed: run k re-draws the solver's random init from
                # global_seed + k, so the num_runs loop varies instead of
                # replaying one fixed init. Same base across combos => paired
                # inits (run k identical everywhere), which is the comparison
                # we want. None => entropy per run, still logged.
                seed=self.global_seed,
            )
            with _run_timeout(self.run_timeout_sec):
                runner.run_build()  # writes its own benchmark_*.json into run_dir

            json_files = sorted(run_dir.glob("benchmark_*.json"))
            json_path = str(json_files[-1]) if json_files else None

            self.index.append(
                {
                    "instance": map_path,
                    "problem": problem_name,
                    "solver": solver_name,
                    "backend": solver_cfg["backend"],
                    "device": device,
                    "penalty_set": penalty_set,
                    "preprocess": preprocess,
                    "num_runs": num_runs,
                    "output_dir": str(run_dir),
                    "benchmark_json": json_path,
                }
            )
        except Exception as exc:
            self.manifest["failures"].append(
                {
                    "instance": map_path,
                    "problem": problem_name,
                    "solver": solver_name,
                    "preprocess": preprocess,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            self.logger.minimal(
                f"[sweep] FAILED {instance_slug}/{problem_name}/{solver_name}"
                f"(preprocess={preprocess}): {exc}"
            )
            self._persist()
            if fail_fast:
                raise
            return

        self._persist()

    def _persist(self):
        """Writes index.json + manifest.json to disk immediately — called
        after every single combo (success, failure, or skip), not just at
        the end, so a killed/crashed sweep leaves a resumable record of
        exactly what's done so far (see the resuming check in _run_one).
        Only updates last_checkpoint, never end_time — see
        checkpoint_manifest()/finalize_manifest() docstrings for why that
        split matters: a manifest read while the sweep is still running (or
        after it was interrupted) must show a null end_time, not one that
        looks like the sweep finished just because a checkpoint happened."""
        self.manifest = checkpoint_manifest(self.manifest)
        self._write_manifest()
        self._write_index()

    def _write_manifest(self):
        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def _write_index(self):
        with open(self.output_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, default=str)
