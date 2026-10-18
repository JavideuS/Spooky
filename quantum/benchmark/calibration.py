"""
Calibrate the solvers' fixed defaults on a family of problems: QAOA's angles
and "auto" shot table, and SA's "auto" read rule.

Both solvers run with fixed settings chosen offline, so the settings have to
work across the windows a problem family produces, not on one instance:
QAOA angles fitted to a few Hamiltonians can cancel the minimum on others,
and generous shot/read counts hide that by brute force. Every command writes
a JSON report (git commit, arguments, machine, results) next to its output,
so a number in the code can be traced to the run that produced it.
Recalibrate whenever the goal term, penalties or preprocessing change, and
check new problem families with --split size (train small maps, test large).

  record    solve every grid instance of a sweep config with SA and save each
            window QUBO as the sampler receives it
  angles    tune 2-layer QAOA angles on recorded windows: maximise how much the
            circuit concentrates each window's exact minimum over uniform
            sampling (mean log10 ratio, clipped); held-out problems
  shots     QAOA shots per qubit-count bin to hit the minimum with the target
            probability; suggested table = next power of two of margin x the
            bin's worst window
  sa-reads  SA: per recorded window, the chance one read reaches a strong
            anneal's best energy (the curve behind the rule), then end to end
            the validity and time of reads = max(4, ceil(c * vars)) for each c

QAOA statevectors are exact (H^n, then per layer exp(-i g E) exp(-i b sum X),
E the QUBO energy normalised to max |coef| = 1, which is the Ising cost
Hamiltonian up to a constant), so those windows are limited to --max-qubits.

    python -m quantum.benchmark.calibration record --config sweep_configs/synthetic_full.yaml
    python -m quantum.benchmark.calibration angles --split size
    python -m quantum.benchmark.calibration shots --angles 0.969 2.528 -0.396 -0.233
    python -m quantum.benchmark.calibration record --config sweep_configs/synthetic_full.yaml --preprocess bfs_safe
    python -m quantum.benchmark.calibration sa-reads --config sweep_configs/synthetic_full.yaml

Heavy commands use a process per core they are allowed (os.sched_getaffinity):
pin them, e.g. `taskset -c 0-11 python -m ...`, to leave the machine usable.
"""

import argparse
import contextlib
import datetime
import glob
import io
import json
import math
import os
import pickle
import platform
import subprocess
from multiprocessing import Pool

import numpy as np
import yaml

from quantum.utils.paths import RESULTS_DIR

OUT_ROOT = RESULTS_DIR / "calibration"
CONFIG_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


def _cores():
    return len(os.sched_getaffinity(0))


def windows_dir(preprocess):
    return OUT_ROOT / f"windows_{preprocess}"


def write_report(cmd, args, results):
    """JSON next to the outputs: what produced these numbers."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True,
                                  cwd=os.path.dirname(__file__)).stdout.strip()
        except OSError:
            return None
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_ROOT / f"{cmd}_{stamp}.json"
    report = {"command": cmd, "args": {k: v for k, v in vars(args).items() if k != "func"},
              "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "-uno")),
              "machine": {"host": platform.node(), "cpu": platform.processor(), "cores_used": _cores()},
              "timestamp": stamp, "results": results}
    path.write_text(json.dumps(report, indent=2, default=float))
    print(f"report: {path}")


def _instances(config, max_cells):
    """(map, problem, cells) for every grid instance of a sweep config."""
    from quantum.pathFormulation import PathfindingProblem

    out = []
    for entry in yaml.safe_load(open(config))["instances"]:
        if entry.get("builder", "grid") != "grid":
            continue
        for prob in entry["problems"]:
            p = PathfindingProblem.from_map_config(entry["map"], prob).as_grid_only()
            cells = p.grid.M * p.grid.N
            if cells <= max_cells:
                out.append((entry["map"], prob, cells))
    return out


def _penalties(name):
    pen = dict(yaml.safe_load(open(CONFIG_YAML))["penalty_sets"][name])
    pen["name"] = name
    return pen


# --- record ------------------------------------------------------------------


def record(config, out_dir, penalty_set="trailing", preprocess="full", runs=2, seed=42, max_cells=100):
    """Solve every grid instance of a sweep config with SA and pickle each
    window QUBO the sampler receives. Instances larger than max_cells are
    skipped."""
    import quantum.solvers.DWave_solver as dwave_module
    from quantum.builder import QUBOBuilder
    from quantum.pathFormulation import PathfindingProblem
    from quantum.solvers.DWave_solver import DWaveSolver
    from quantum.utils.logger import set_verbose_level

    set_verbose_level(0)
    os.makedirs(out_dir, exist_ok=True)
    penalties = _penalties(penalty_set)
    base = dwave_module.SimulatedAnnealingSampler
    current = {}

    class Recording(base):
        def sample(self, bqm, **kw):
            Q, offset = bqm.to_qubo()
            name = f"{current['inst']}__r{current['run']}__{current['k']:03d}.pkl"
            with open(os.path.join(out_dir, name), "wb") as fh:
                pickle.dump({"inst": current["inst"], "cells": current["cells"], "Q": Q, "offset": offset}, fh)
            current["k"] += 1
            return super().sample(bqm, **kw)

    dwave_module.SimulatedAnnealingSampler = Recording
    try:
        for map_path, prob, cells in _instances(config, max_cells):
            inst = f"{os.path.basename(map_path)}:{prob}"
            for run in range(runs):
                problem = PathfindingProblem.from_map_config(map_path, prob).as_grid_only()
                builder = QUBOBuilder(problem, penalties=dict(penalties), name=prob)
                solver = DWaveSolver(normalize_scale=1, num_reads="auto", seed=seed + run)
                current.update(inst=inst, cells=cells, run=run, k=0)
                with contextlib.redirect_stdout(io.StringIO()):
                    solver.solve(builder, preprocess=preprocess)
            print(f"recorded {inst}")
    finally:
        dwave_module.SimulatedAnnealingSampler = base


def load_raw(path):
    """Recorded windows as {inst, cells, Q} (Q normalised to max |coef| = 1)."""
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.pkl"))):
        w = pickle.load(open(f, "rb"))
        scale = max((abs(v) for v in w["Q"].values()), default=0)
        if scale:
            out.append({"inst": w["inst"], "cells": w.get("cells"),
                        "Q": {k: v / scale for k, v in w["Q"].items()}})
    return out


def split(wins, how, small_cells=25):
    """(train, test): alternate problems, or by map size (train on maps of at
    most small_cells cells, test on larger ones -- an out-of-family check)."""
    if how == "size":
        return ([w for w in wins if w["cells"] <= small_cells],
                [w for w in wins if w["cells"] > small_cells])
    insts = sorted({w["inst"] for w in wins})
    test_insts = set(insts[1::2])
    return ([w for w in wins if w["inst"] not in test_insts], [w for w in wins if w["inst"] in test_insts])


# --- QAOA: exact statevectors on recorded windows ------------------------------


def load_windows(path, max_qubits=20):
    wins = []
    for w in load_raw(path):
        Q = w["Q"]
        var = sorted({i for k in Q for i in k})
        n = len(var)
        if not 2 <= n <= max_qubits:
            continue
        pos = {v: k for k, v in enumerate(var)}
        bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.float64)
        E = np.zeros(2**n)
        for (i, j), c in Q.items():
            E += c * bits[:, pos[i]] * (bits[:, pos[j]] if i != j else 1)
        is_min = E <= E.min() + 1e-9
        wins.append({**w, "n": n, "E": E, "is_min": is_min, "p_uniform": is_min.mean()})
    return wins


def p_min(w, angles):
    n, E = w["n"], w["E"]
    psi = np.full(2**n, 2 ** (-n / 2), dtype=np.complex128)
    for g, b in zip(angles[0], angles[1]):
        psi = psi * np.exp(-1j * g * E)
        c, s = np.cos(b), -1j * np.sin(b)
        t = psi.reshape((2,) * n)
        for ax in range(n):
            t = c * t + s * np.flip(t, axis=ax)
        psi = t.reshape(-1)
    return float((np.abs(psi[w["is_min"]]) ** 2).sum())


def amplification(wins, angles, floor=-3.0):
    """Mean log10(p_min / p_min under uniform sampling), clipped at floor so a
    window where the circuit cancels the minimum costs a fixed amount."""
    return float(np.mean([
        max(np.log10(max(p_min(w, angles), 1e-300) / w["p_uniform"]), floor) for w in wins
    ]))


def shots_needed(p, target):
    p = min(max(p, 1e-300), 1 - 1e-15)
    return math.log(1 - target) / math.log(1 - p)


def _as_angles(x):
    return np.array([[x[0], x[1]], [x[2], x[3]]])


def _optimise(args):
    from scipy.optimize import minimize

    wins, x0 = args
    r = minimize(lambda x: -amplification(wins, _as_angles(x)), x0, method="Nelder-Mead",
                 options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4})
    return r.x, -r.fun


def tune_angles(wins, how="alternate", starts=16, seed=0, train_max_qubits=18):
    """Multi-start Nelder-Mead on the train split; [(angles, train, test)] best first."""
    train, test = split(wins, how)
    train = [w for w in train if w["n"] <= train_max_qubits]
    rng = np.random.default_rng(seed)
    x0s = [rng.uniform([0, 0, -1.6, -1.6], [3.2, 3.2, 1.6, 1.6]) for _ in range(starts)]
    with Pool(min(starts, _cores())) as pool:
        res = pool.map(_optimise, [(train, x0) for x0 in x0s])
    out = [(_as_angles(x), s, amplification(test, _as_angles(x))) for x, s in res]
    return sorted(out, key=lambda r: -r[2]), len(train), len(test)


QAOA_BINS = ((2, 9), (10, 12), (13, 16), (17, 20))


def shot_table(wins, angles, target=0.99, margin=3.0):
    rows = []
    for lo, hi in QAOA_BINS:
        ws = [w for w in wins if lo <= w["n"] <= hi]
        if not ws:
            continue
        need = [shots_needed(p_min(w, angles), target) for w in ws]
        uniform = [shots_needed(w["p_uniform"], target) for w in ws]
        rows.append({"qubits": f"{lo}-{hi}", "windows": len(ws), "median": int(math.ceil(np.median(need))),
                     "p95": int(math.ceil(np.percentile(need, 95))), "max": int(math.ceil(max(need))),
                     "uniform_median": int(math.ceil(np.median(uniform))),
                     "suggested": int(2 ** math.ceil(math.log2(max(margin * max(need), 1))))})
    return rows


# --- SA reads ------------------------------------------------------------------

SA_BINS = ((2, 20), (21, 40), (41, 60), (61, 100), (101, 160), (161, 10_000))


def _sa_hit_rate(args):
    """One window: reference = best of a strong anneal; p = fraction of reads at
    neal's default sweeps that reach it."""
    import dimod
    from neal import SimulatedAnnealingSampler

    w, ref_reads, ref_sweeps, reads = args
    bqm = dimod.BinaryQuadraticModel.from_qubo(w["Q"])
    sa = SimulatedAnnealingSampler()
    ref = sa.sample(bqm, num_reads=ref_reads, num_sweeps=ref_sweeps, seed=7).first.energy
    r = sa.sample(bqm, num_reads=reads, seed=11)
    return len(bqm.variables), float((r.record.energy <= ref + 1e-9).mean())


def _sa_run(args):
    from quantum.benchmark.benchmark import BenchmarkRunner
    from quantum.builder import QUBOBuilder
    from quantum.pathFormulation import PathfindingProblem
    from quantum.solvers.DWave_solver import DWaveSolver
    from quantum.utils.logger import set_verbose_level

    map_path, prob, cells, factor, runs, penalty_set, preprocess, seed = args
    set_verbose_level(0)
    problem = PathfindingProblem.from_map_config(map_path, prob).as_grid_only()
    builder = QUBOBuilder(problem, penalties=_penalties(penalty_set), name=prob)
    solver = DWaveSolver(normalize_scale=1, num_reads="auto", seed=seed)
    solver.AUTO_READS_PER_VAR = factor
    out = str(OUT_ROOT / "sa_reads_runs" / f"{os.path.basename(map_path)}_{prob}_c{factor}")
    with contextlib.redirect_stdout(io.StringIO()):
        res = BenchmarkRunner(builder, solver, num_runs=runs, output_dir=out, level=1,
                              preprocess=preprocess, seed=seed).run_build()
    return {"inst": f"{os.path.basename(map_path)}:{prob}", "cells": cells, "factor": factor,
            "valid": sum(r["valid"] for r in res["runs"]), "runs": len(res["runs"]),
            "time": float(np.mean([r["execution_time_sec"] for r in res["runs"]]))}


def sa_reads(args):
    results = {}
    # 1. per-window hit rate: the shape of the rule
    raw = load_raw(args.windows) if os.path.isdir(args.windows) else []
    if raw:
        seen, uniq = set(), []
        for w in raw:
            key = repr(sorted(w["Q"].items()))
            if key not in seen:
                seen.add(key)
                uniq.append(w)
        with Pool(_cores()) as pool:
            hits = pool.map(_sa_hit_rate, [(w, 256, 10_000, 128) for w in uniq], chunksize=4)
        print(f"per window ({len(uniq)} distinct): one read at default sweeps reaching a strong anneal's best")
        rows = []
        for lo, hi in SA_BINS:
            ps = [p for n, p in hits if lo <= n <= hi]
            if ps:
                need = [shots_needed(p, 0.99) for p in ps if p > 0]
                rows.append({"vars": f"{lo}-{hi}", "windows": len(ps), "mean_p": float(np.mean(ps)),
                             "never_hit": sum(p == 0 for p in ps),
                             "median_reads_99": int(math.ceil(np.median(need))) if need else None})
                print("  {vars:>9} vars  {windows:4d} windows  mean p {mean_p:.3f}  "
                      "never hit in 128 {never_hit:3d}  median reads for 99% {median_reads_99}".format(**rows[-1]))
        results["per_window"] = rows
    else:
        print(f"no recorded windows in {args.windows}; skipping the per-window curve")
    # 2. end to end: validity and time per factor
    jobs = [(m, p, cells, c, args.runs, args.penalty_set, args.preprocess, args.seed)
            for m, p, cells in _instances(args.config, args.max_cells) for c in args.factors]
    with Pool(_cores(), maxtasksperchild=1) as pool:
        rows = pool.map(_sa_run, jobs)
    print(f"end to end ({args.preprocess}, {args.runs} runs per problem): valid / runs, time per run")
    summary = []
    for c in args.factors:
        rs = [r for r in rows if r["factor"] == c]
        small = [r for r in rs if r["cells"] <= 25]
        large = [r for r in rs if r["cells"] > 25]
        row = {"factor": c, "valid": sum(r["valid"] for r in rs), "runs": sum(r["runs"] for r in rs),
               "valid_small_maps": sum(r["valid"] for r in small), "runs_small_maps": sum(r["runs"] for r in small),
               "valid_large_maps": sum(r["valid"] for r in large), "runs_large_maps": sum(r["runs"] for r in large),
               "time": float(np.mean([r["time"] for r in rs]))}
        summary.append(row)
        print("  c={factor:<6} valid {valid}/{runs}  (maps <=25 cells {valid_small_maps}/{runs_small_maps}, "
              "larger {valid_large_maps}/{runs_large_maps})  {time:.2f} s".format(**row))
    best = max(r["valid"] for r in summary)
    noise = 2 * math.sqrt(summary[0]["runs"] * 0.25)  # ~2 sigma of a binomial total
    pick = min((r for r in summary if r["valid"] >= best - noise), key=lambda r: r["factor"])
    print(f"smallest factor within ~2 sigma ({noise:.0f}) of the best: c = {pick['factor']}")
    results.update(end_to_end=summary, per_problem=rows, suggested_factor=pick["factor"])
    write_report("sa_reads", args, results)


# --- CLI -----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--config", required=True, help="sweep config whose instances form the family")
    r.add_argument("--preprocess", default="full")
    r.add_argument("--out", default=None, help="default: results/calibration/windows_<preprocess>")
    r.add_argument("--penalty-set", default="trailing")
    r.add_argument("--runs", type=int, default=2)
    r.add_argument("--max-cells", type=int, default=100)
    for name in ("angles", "shots"):
        p = sub.add_parser(name)
        p.add_argument("--windows", default=str(windows_dir("full")))
        p.add_argument("--max-qubits", type=int, default=20)
    sub.choices["angles"].add_argument("--starts", type=int, default=16)
    sub.choices["angles"].add_argument("--split", choices=["alternate", "size"], default="alternate")
    s = sub.choices["shots"]
    s.add_argument("--angles", nargs=4, type=float, metavar=("G1", "G2", "B1", "B2"), required=True)
    s.add_argument("--target", type=float, default=0.99)
    s.add_argument("--margin", type=float, default=3.0)
    sr = sub.add_parser("sa-reads")
    sr.add_argument("--config", required=True)
    sr.add_argument("--preprocess", default="bfs_safe")
    sr.add_argument("--windows", default=None, help="default: results/calibration/windows_<preprocess>")
    sr.add_argument("--factors", nargs="+", type=float, default=[0.125, 0.25, 0.5, 1.0])
    sr.add_argument("--runs", type=int, default=5)
    sr.add_argument("--penalty-set", default="trailing")
    sr.add_argument("--max-cells", type=int, default=100)
    sr.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.cmd == "record":
        record(a.config, a.out or str(windows_dir(a.preprocess)), a.penalty_set, a.preprocess, a.runs,
               max_cells=a.max_cells)
        return
    if a.cmd == "sa-reads":
        a.windows = a.windows or str(windows_dir(a.preprocess))
        sa_reads(a)
        return
    wins = load_windows(a.windows, a.max_qubits)
    print(f"{len(wins)} windows (2-{a.max_qubits} qubits) from {len({w['inst'] for w in wins})} problems")
    if a.cmd == "angles":
        res, n_train, n_test = tune_angles(wins, a.split, a.starts)
        print(f"split {a.split}: train {n_train} windows, test {n_test} windows; log10 amplification:")
        for ang, tr, te in res[:4]:
            print(f"  gammas {np.round(ang[0], 3)} betas {np.round(ang[1], 3)}  train {tr:+.3f}  test {te:+.3f}")
        write_report("angles", a, [{"angles": ang.tolist(), "train": tr, "test": te} for ang, tr, te in res])
    else:
        angles = _as_angles(a.angles)
        amp = amplification(wins, angles)
        print(f"log10 amplification {amp:+.3f}; shots to hit the minimum with p >= {a.target} (margin {a.margin}x):")
        rows = shot_table(wins, angles, a.target, a.margin)
        for row in rows:
            print("  {qubits:>6} qubits  {windows:3d} windows  median {median:6d}  p95 {p95:7d}  "
                  "max {max:7d}  uniform median {uniform_median:8d}  -> suggested {suggested}".format(**row))
        write_report("shots", a, {"amplification": amp, "table": rows})


if __name__ == "__main__":
    main()
