"""Report-grade figures and tables for the internship report's Benchmarking section.

Reads finished sweeps (their analysis/runs_long.csv, produced by
`python -m quantum.benchmark.analysis.run_report -d <sweep>`) plus the raw
hardware-run JSONs for QPU telemetry, and writes PDFs + \\input-able .tex
tables into --out.

    python -m quantum.paper.report_figures --out <overleaf>/figures/benchmark \\
        [--full-sweep <sweep_id>]   # e.g. a rerun with trailing penalties
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantum.benchmark.analysis.aggregate import load_sweep  # noqa: E402
from quantum.utils.paths import RESULTS_DIR  # noqa: E402

SWEEPS = Path(RESULTS_DIR) / "sweeps"

# Full synthetic sweep: 17 maps / 44 problems, trailing penalties, preprocess=full
# (+ an SA preprocess=bfs_safe arm). The earlier crash-penalty run: sweep_20260927_183106_fa53bb94.
FULL_SWEEP = "sweep_20260928_013052_c905c219"
# Hardware-sized instances, trailing penalties, preprocess=bfs_safe.
MATCHED_CLASSICAL = "sweep_20260928_000237_88c2ab01"  # also has a preprocess=full arm
HARDWARE_SWEEPS = [
    "sweep_20260927_221319_f8945177",  # IBM + local SA/QAOA
    "sweep_20260927_224931_ad1a6b7e",  # IQM Garnet
    "sweep_20260927_234037_56bd4241",  # IQM Emerald
    "sweep_20260927_235007_fed04f6a",  # 2-robot: IBM + Garnet
]

SOLVER_LABEL = {
    "ilp_highs": "ILP (HiGHS)",
    "cbs": "CBS",
    "sa_neal": "QUBO + SA",
    "qaoa_sim": "QUBO + QAOA (sim)",
    "sa_neal_bfs_safe": "QUBO + SA (bfs_safe)",
}
COLORS = {
    "ilp_highs": "#d95f02", "cbs": "#1b9e77", "sa_neal": "#7570b3", "sa_neal_bfs_safe": "#a6a1d6", "qaoa_sim": "#e7298a",
}


def _arm(solver, preprocess):
    """A solver + its non-default preprocess mode is its own arm (e.g. sa_neal_bfs_safe)."""
    return solver if preprocess in ("full", None) or pd.isna(preprocess) else f"{solver}_{preprocess}"


def _order(r):
    return [a for a in SOLVER_LABEL if a in set(r.arm)]

plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"}
)


def _runs(sweep):
    r = pd.read_csv(SWEEPS / sweep / "analysis" / "runs_long.csv")
    r["valid"] = r["valid"].fillna(False).astype(bool)
    r["map"] = r.instance_map.str.split("/").str[-1]
    r["arm"] = [_arm(s, p) for s, p in zip(r.solver_name, r.preprocess)]
    return r


def _timeouts(sweep):
    m = json.loads((SWEEPS / sweep / "manifest.json").read_text())
    return {
        (f["instance"].split("/")[-1], f["problem"], _arm(f["solver"], f.get("preprocess")))
        for f in m["failures"]
        if "run_timeout" in str(f.get("error", ""))
    }


def _grid_cells(g):
    m, n = g.split("x")
    return int(m) * int(n)


# ---------------------------------------------------------------------------
def success_heatmap(out):
    r = _runs(FULL_SWEEP)
    tos = _timeouts(FULL_SWEEP)
    inst = (
        r[["map", "problem_name", "grid_size", "num_robots"]]
        .drop_duplicates(["map", "problem_name"])
        .assign(cells=lambda d: d.grid_size.map(_grid_cells))
    )
    # timed-out cells never wrote a JSON, so add their instances back from the manifest
    extra = {(m, p) for m, p, _ in tos} - set(zip(inst["map"], inst.problem_name))
    assert not extra, f"instances only present as timeouts: {extra}"
    inst = inst.sort_values(["cells", "num_robots", "map", "problem_name"])
    rate = r.groupby(["map", "problem_name", "arm"]).valid.mean()
    SOLVER_ORDER = _order(r)

    M = np.full((len(inst), len(SOLVER_ORDER)), np.nan)
    txt = [["" for _ in SOLVER_ORDER] for _ in range(len(inst))]
    for i, (m, p) in enumerate(zip(inst["map"], inst.problem_name)):
        for j, s in enumerate(SOLVER_ORDER):
            if (m, p, s) in rate.index:
                M[i, j] = rate[(m, p, s)]
                txt[i][j] = f"{rate[(m, p, s)]:.0%}".replace("%", "")
            elif (m, p, s) in tos:
                txt[i][j] = "T/O"

    fig, ax = plt.subplots(figsize=(1.0 * len(SOLVER_ORDER) + 2.2, 0.19 * len(inst) + 0.9))
    ax.imshow(np.ma.masked_invalid(M), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_facecolor("#dddddd")
    for i in range(len(inst)):
        for j in range(len(SOLVER_ORDER)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(len(SOLVER_ORDER)), [SOLVER_LABEL[s] for s in SOLVER_ORDER], rotation=25, ha="right")
    ylab =[f"{m}: {p} ({n}R)" for m, p, n in zip(inst["map"], inst.problem_name, inst.num_robots)]
    ax.set_yticks(range(len(inst)), ylab, fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Success rate (%) per instance, full synthetic sweep", fontsize=9)
    fig.savefig(out / "success_heatmap.pdf")
    plt.close(fig)


def time_scaling(out):
    r = _runs(FULL_SWEEP)
    v = r[r.valid]
    grids = ["5x5", "10x10", "50x50"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    for ax, g in zip(axes, grids):
        d = v[v.grid_size == g]
        for s in _order(r):
            ds = d[d.arm == s].groupby("num_robots").execution_time_sec.median()
            if len(ds):
                ls = "--" if s.endswith("bfs_safe") else "-"
                ax.plot(ds.index, ds.values, "o" + ls, ms=3, lw=1.2, color=COLORS[s], label=SOLVER_LABEL[s])
        ax.set_yscale("log")
        ax.set_title(f"{g} grids", fontsize=9)
        ax.set_xlabel("robots")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3, which="major")
    axes[0].set_ylabel("median solve time [s]")
    axes[-1].axhline(300, ls=":", c="gray", lw=1)
    axes[-1].text(1, 330, "300 s timeout", fontsize=7, color="gray")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Solve time of valid runs (median over maps/runs)", fontsize=9, y=1.03)
    fig.savefig(out / "time_scaling.pdf")
    plt.close(fig)


def solver_summary_table(out):
    r = _runs(FULL_SWEEP)
    tos = _timeouts(FULL_SWEEP)
    rows = []
    for s in _order(r):
        d = r[r.arm == s]
        per_inst = d.groupby(["map", "problem_name"]).valid.mean()
        n_to = sum(1 for t in tos if t[2] == s)
        v = d[d.valid]
        rows.append(
            f"{SOLVER_LABEL[s].replace('_', '\\_')} & {len(per_inst) + n_to} & {(per_inst == 1).sum()} & {n_to} & "
            f"{d.valid.mean():.0%} & {v.execution_time_sec.median():.3f} & "
            f"{v.avg_path_efficiency.mean():.2f} \\\\".replace("%", "\\%")
        )
    (out / "solver_summary.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\hline\n"
        "Solver & Inst. & Always solved & Timeouts & Run success & Median $t$ [s] & Path eff. \\\\\n"
        "\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n"
    )


# ---------------------------------------------------------------------------
QPU_CALIBRATION = Path(RESULTS_DIR) / "qpu_calibration.json"
FAMILIES = ["IBM Heron", "IQM-garnet", "IQM-emerald"]
# IQM exposes no CLOPS via API; Garnet's is published on iqmacademy.com/qpu/garnet (2026-09).
IQM_PUBLISHED_CLOPS = {"IQM-garnet": 2600}
FAMILY_LABEL = {"IBM Heron": "IBM Heron", "IQM-garnet": "IQM Garnet", "IQM-emerald": "IQM Emerald"}


def _iqm_depths():
    """IQM job JSONs don't carry the 2Q depth; qpu_calibration.json does, keyed by the same
    measured execution time (unique to the microsecond)."""
    cal = json.loads(QPU_CALIBRATION.read_text())
    return {round(e["execution_time_sec"], 6): e["two_qubit_depth"] for v in cal.values() for e in v}


def _hardware_jobs():
    """One row per QPU job (= one window of one run).

    qpu_sec is the best available *device* time per job: IQM's measured execution span,
    IBM's gate-model estimate (IBM only exposes whole-second billed usage per job).
    """
    depths = _iqm_depths()
    rows = []
    for s in HARDWARE_SWEEPS:
        for f in glob.glob(str(SWEEPS / s / "*" / "benchmark_*.json")):
            if "qaoa_ibm" not in f and "qaoa_iqm" not in f:
                continue
            inst = f.split("synthetic_")[1].split("__qaoa")[0].split("_", 1)[1].replace("__", ":")
            for run in json.loads(Path(f).read_text())["runs"]:
                for b in run.get("qpu_time_estimate_breakdown") or []:
                    iqm = b.get("iqm_timing")
                    row = dict(
                        sweep=s, file=f, run=run["run_id"], inst=inst, backend=b["backend"],
                        family=b["backend"] if iqm else "IBM Heron",
                        valid=run.get("valid") in (True, "True"),
                        shots=b.get("shots"), wall=b.get("wall_clock_sec"),
                    )
                    if iqm:
                        depth = depths.get(round(iqm["execution_time_sec"], 6))
                        clops = IQM_PUBLISHED_CLOPS.get(b["backend"])
                        row.update(qpu_sec=iqm["execution_time_sec"], billed=iqm["job_total_sec"], depth2q=depth,
                                   clops_sec=b["shots"] * depth / clops if clops and depth else np.nan)
                    else:
                        row.update(qpu_sec=b["gate_model"]["total_estimate_sec"],
                                   clops_sec=b["clops_model"]["total_estimate_sec"],
                                   depth2q=b["clops_model"]["two_qubit_depth"],
                                   billed=b["billed_usage"]["usage"])
                    rows.append(row)
    return pd.DataFrame(rows)


def _tex_inst(i):
    return "\\texttt{" + i.replace("_", "\\_") + "}"


def hardware_table(out):
    jobs = _hardware_jobs()
    runs = jobs.groupby(["sweep", "file", "run", "inst", "family"], as_index=False).agg(
        valid=("valid", "first"), jobs=("shots", "size"), qpu=("qpu_sec", "sum"), wall=("wall", "sum")
    )
    sim = load_sweep(SWEEPS / MATCHED_CLASSICAL)
    sim["valid"] = sim["valid"].fillna(False).astype(bool)
    sim = sim[(sim.preprocess == "bfs_safe") & sim.solver_name.isin(["sa_neal", "qaoa_sim"])]
    sim["inst"] = sim.instance_map.str.split("/").str[-1] + ":" + sim.problem_name
    insts = sorted(set(runs.inst) | set(sim.inst), key=lambda x: (x.count("two"), x))

    # --- success
    lines = []
    for i in insts:
        cells = []
        for s in ["sa_neal", "qaoa_sim"]:
            d = sim[(sim.inst == i) & (sim.solver_name == s)]
            cells.append(f"{d.valid.sum()}/{len(d)}" if len(d) else "--")
        for fam in FAMILIES:
            d = runs[(runs.inst == i) & (runs.family == fam)]
            cells.append(f"{d.valid.sum()}/{len(d)}" if len(d) else "--")
        lines.append(_tex_inst(i) + " & " + " & ".join(cells) + " \\\\")
    (out / "hardware_success.tex").write_text(
        "\\begin{tabular}{lccccc}\n\\hline\n"
        "Instance & SA & QAOA sim & IBM Heron & IQM Garnet & IQM Emerald \\\\\n\\hline\n"
        + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n"
    )

    # --- per-job timing: device time (measured / modelled) vs what is billed vs wall clock
    lines = []
    for fam in FAMILIES:
        d = jobs[jobs.family == fam]
        if d.empty:
            continue
        clops = f"{d.clops_sec.median():.2f}" if d.clops_sec.notna().any() else "--"
        lines.append(
            f"{FAMILY_LABEL[fam]} & {len(d)} & {d.shots.median():.0f} & {d.depth2q.median():.0f} & "
            f"{d.qpu_sec.median():.2f} & {clops} & {d.billed.median():.2f} & {d.wall.median():.1f} \\\\"
        )
    (out / "hardware_timing.tex").write_text(
        "\\begin{tabular}{lrrrrrrr}\n\\hline\n"
        "Backend & Jobs & Shots & 2Q depth & QPU [s] & CLOPS est. [s] & Server/billed [s] & Wall [s] \\\\\n"
        "\\hline\n" + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n"
    )

    # --- per-run time-to-solution on the instances every backend ran
    sim_t = sim[sim.valid].groupby(["inst", "solver_name"]).execution_time_sec.median()
    rq = runs.groupby(["inst", "family"])[["qpu", "wall"]].median()
    lines = []
    for i in insts:
        cells = [f"{sim_t.get((i, s), np.nan):.3f}" if (i, s) in sim_t.index else "--" for s in ["sa_neal", "qaoa_sim"]]
        for fam in FAMILIES:
            cells += [f"{rq.loc[(i, fam)].qpu:.2f}", f"{rq.loc[(i, fam)].wall:.0f}"] if (i, fam) in rq.index else ["--", "--"]
        lines.append(_tex_inst(i) + " & " + " & ".join(cells) + " \\\\")
    (out / "hardware_tts.tex").write_text(
        "\\begin{tabular}{lcc|cc|cc|cc}\n\\hline\n"
        " & & & \\multicolumn{2}{c|}{IBM Heron} & \\multicolumn{2}{c|}{IQM Garnet} & \\multicolumn{2}{c}{IQM Emerald} \\\\\n"
        "Instance & SA & QAOA sim & device & wall & device & wall & device & wall \\\\\n\\hline\n"
        + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n"
    )
    jobs.drop(columns=["file"]).to_csv(out / "hardware_jobs.csv", index=False)


def preprocess_table(out):
    r = load_sweep(SWEEPS / MATCHED_CLASSICAL)
    r["valid"] = r["valid"].fillna(False).astype(bool)
    r = r[r.solver_name.isin(["sa_neal", "qaoa_sim"])]
    r["inst"] = r.instance_map.str.split("/").str[-1] + ":" + r.problem_name
    lines = []
    for i, d in r.groupby("inst", sort=True):
        cells = []
        for pp in ["bfs_safe", "full"]:
            dp = d[d.preprocess == pp]
            q = dp.max_solver_variables.max()
            cells.append(f"{int(q)}")
            for s in ["sa_neal", "qaoa_sim"]:
                ds = dp[dp.solver_name == s]
                cells.append(f"{ds.valid.mean():.0%}".replace("%", "\\%"))
        lines.append(f"\\texttt{{{i.replace('_', '\\_')}}} & " + " & ".join(cells) + " \\\\")
    (out / "preprocess_ablation.tex").write_text(
        "\\begin{tabular}{l|ccc|ccc}\n\\hline\n"
        " & \\multicolumn{3}{c|}{\\texttt{bfs\\_safe}} & \\multicolumn{3}{c}{\\texttt{full}} \\\\\n"
        "Instance & max qubits & SA & QAOA & max qubits & SA & QAOA \\\\\n\\hline\n"
        + "\n".join(lines)
        + "\n\\hline\n\\end{tabular}\n"
    )


def main():
    global FULL_SWEEP, MATCHED_CLASSICAL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--full-sweep", default=FULL_SWEEP, help="sweep id for the synthetic sweep figures")
    ap.add_argument("--matched-sweep", default=MATCHED_CLASSICAL, help="sweep id for the preprocess ablation")
    args = ap.parse_args()
    FULL_SWEEP, MATCHED_CLASSICAL = args.full_sweep, args.matched_sweep
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for fn in (success_heatmap, time_scaling, solver_summary_table, hardware_table, preprocess_table):
        fn(out)
        print(f"[report_figures] {fn.__name__} done")


if __name__ == "__main__":
    main()
